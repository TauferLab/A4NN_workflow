# Callbacks during NAS training
prev_train_file = None
epochs = 10 # The number of epochs must match the epochs in the config files 
train_nafc = 0
served_files = []
def train_callback(vol, rank):

    def afc_cb(name):
        global train_nafc
        global served_files
        global prev_train_file
        if name.endswith("training.h5"):
            def bsa_cb():
                return [3]

            vol.set_serve_indices(bsa_cb)
            if name != prev_train_file and prev_train_file!=None:
                train_nafc = 0
                print("TRAINING: resetting afc_cb counter to ", train_nafc)
            prev_train_file = name
            train_nafc += 1
            if train_nafc > epochs or train_nafc==1 and name in served_files:
                print("TRAINING: skipping serving due to vinarch convergence", name, train_nafc)
            else:
                served_files.append(name)
                if vol.is_passthru(name, "*") == False:
                    vol.serve_all()
                else:
                    vol.serve_all(True, False)
        elif name.endswith("global_vinarch.h5"):
            def bsa_cb():
                return [0]

            vol.set_serve_indices(bsa_cb)
            if vol.is_passthru(name, "*") == False:
                vol.serve_all()
            else:
                vol.serve_all(True, False)  
    def bfo_cb(name):
        if vol.is_passthru(name, "*") == True:
            if name.endswith("penguin.h5"):
                fnames = vol.get_filenames(2)
                print(f"{fnames = }")
                print(fnames[0])
                vol.send_done(2)
            elif name.endswith("local_vinarch.h5"):
                fnames = vol.get_filenames(1)
                print(f"{fnames = }")
                print(fnames[0])
                vol.send_done(1)

    vol.set_before_file_open(bfo_cb)
    vol.set_after_file_close(afc_cb)
    vol.set_keep(True)
    vol.serve_on_close = False

# Callbacks during penguin 
penguin_nafc = 0       # number of times afc_cb was called
penguin_nscf = -1      # number of times scf_cb was called
training_file = None   # needed for resetting the counters after shifting to a new arch
prev_file = None       # needed for resetting the counters after shifting to a new arch
prev_train_file = None
def penguin_callback(vol, rank):

    def afc_cb(name):
        global penguin_nafc
        global prev_file
        global penguin_nscf
        print("penguin_callback afc_cb: name =", name)
        if name.endswith("penguin.h5"):
            if name != prev_file and prev_file!=None:
                penguin_nafc = 0
                penguin_nscf = 0
                print("PENGUIN: resetting afc_cb counter to ", penguin_nafc, "and scf_cb counter to ", penguin_nscf)
            prev_file = name
            if penguin_nafc%2==0:
                print("PENGUIN: serving at", penguin_nafc)
                if vol.is_passthru(name, "*") == False:
                    vol.serve_all()
                else:
                    vol.serve_all(True, False)
            else:
                print("PENGUIN: skipping serving", penguin_nafc)
            penguin_nafc += 1

    def scf_cb():
        global penguin_nscf
        global training_file
        global prev_train_file
        import re
        penguin_nscf += 1
        if (penguin_nscf-1)%3==0 or penguin_nscf==0:
            fnames = vol.get_filenames(1)
            print(f"{fnames = }", penguin_nscf)
            # Filter out filenames that contains "vinarch" and then sort numerically
            sorted_fnames = sorted([f for f in fnames if "vinarch" not in f], 
                                 key=lambda x: int(re.search(r"arch_(\d+)", x).group(1)))
            print(f"{sorted_fnames = }", penguin_nscf)
            training_file = sorted_fnames[-1]
            if training_file != prev_train_file and prev_train_file!=None:
                print("PENGUIN: another reset to penguin_nscf to 0 due to vinarch convergence")
                penguin_nscf = 0
            prev_train_file = training_file
            if vol.is_passthru(training_file, "*") == True:
                vol.send_done(1)
            return training_file
        else:
            penguin_file = training_file.replace("training.h5", "penguin.h5")
            return penguin_file


    # set the callbacks
    vol.set_after_file_close(afc_cb)
    vol.set_consumer_filename(scf_cb)

    vol.set_keep(True)
    vol.serve_on_close = False

# Callbacks during vinarch
def vinarch_callback(vol, rank):
    def bfo_cb(name):
        print("vinarch_callback bfo_cb: name =", name)
        if vol.is_passthru(name, "*") == True:
            if name.endswith("global_vinarch.h5"):
                fnames = vol.get_filenames(0)
                print(f"{fnames = }")
                print(fnames[0])
                vol.send_done(0)


    def afc_cb(name):
        if vol.is_passthru(name, "*") == True:
            if name.endswith("local_vinarch.h5"):
                vol.serve_all(True, False)

    vol.set_before_file_open(bfo_cb)
    vol.set_after_file_close(afc_cb)
