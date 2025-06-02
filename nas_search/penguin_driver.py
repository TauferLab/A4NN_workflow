# -------------------------------------------------- #
# Driver code for the Predictive Engine (Penguin)    #
# -------------------------------------------------- #

import sys
import os

# Update project root path before running
sys.path.insert(0, os.environ['NSGA_NET_PATH']) 

import time
import argparse
import json
import pandas as pd 
import h5py
from pathlib import Path

from penguin.penguino import Penguin


def main(raw_args=None):
    """
    
    :param raw_args: command-line args parser
    """
    # Read configuration file arguments
    parser = argparse.ArgumentParser(description='Penguin Optimization')
    # parser.add_argument('--dirpath', dest='path_to_results', type=Path) # path to file with results from NAS
    parser.add_argument('--config', dest='penguin_config', type=argparse.FileType('r')) # path to configuration file
    config = parser.parse_args(raw_args)

    # Extract arguments
    penguin_config = config.penguin_config # extract penguin configuration
    # path_to_results = config.path_to_results # extract path to file with NAS results

    # Opening JSON configuration file
    penguin_args=json.load(penguin_config) # load the penguin arguments from the path
    num_epochs = penguin_args['train_params']['epochs']

    path_to_results = os.path.join(penguin_args['io_params']['save_dir'], 'arch_1') # path to file with results from NAS

    if "penguin_config" in penguin_args:
        dict_params = penguin_args['penguin_config']
    else:
        dict_params = {}


    training_file = None
    path_to_fitness = os.path.join(path_to_results, "training.h5") # Path to NAS results

    # Open the training file for the current model to retrieve results
    h5f = h5py.File(path_to_fitness,'r')
    fit_ds = h5f['val_accs']
    fitnesses = fit_ds[:].tolist()
    epoch_ds = h5f['epoch']
    epoch = epoch_ds[:][-1]
    training_file = h5f.filename
    h5f.close()
    if epoch==-1:
        print("PENGUIN: converged vinarch, terminating early")
        return 

    peng_res_pth = training_file.replace("training.h5", "penguin.h5")

    # Open the penguin local file to retrieve past predictions
    if epoch>0: 
        peng_df = h5py.File(peng_res_pth,'r')
        pred_ds = peng_df['predictions']
        predictions = pred_ds[:].tolist()
        peng_df.close()
    else:
        predictions = list()

    # Create penguin object
    penguin = Penguin(e_pred=num_epochs, **dict_params)

    # Start penguin's parametric modeling
    peng_start = time.time()
    current_prediction, current_function, fit_params = penguin.predict(epoch, fitnesses) 
    print('penguin prediction after epoch {} is {}.'.format(epoch, current_prediction))
    predictions.append(current_prediction)
    is_converged = penguin.evaluate(predictions) # evaluate if penguin converged
    peng_end = time.time()
    peng_time = peng_end - peng_start

     
    # If first training epoch, create a local penguin file for the current model to store predictions
    if epoch==0: 
        h5f = h5py.File(peng_res_pth, 'w')
        h5f.create_dataset('epoch', data=[epoch], maxshape=(None,))  
        is_converged = False 
        h5f.create_dataset('converged', data=[is_converged], maxshape=(None,)) 
        h5f.create_dataset('predictions', data=[current_prediction], maxshape=(None,))
        h5f.close()
    else: # Otherwise, append new predictions
        h5f = h5py.File(peng_res_pth, 'a')
        epoch_ds = h5f['epoch']
        new_size = epoch_ds.shape[0] + 1
        epoch_ds.resize((new_size,))
        epoch_ds[-1] = epoch

        conv_ds = h5f['converged']
        new_size = conv_ds.shape[0] + 1
        conv_ds.resize((new_size,))
        conv_ds[-1] = is_converged

        pred_ds = h5f['predictions']
        new_size = pred_ds.shape[0] + 1
        pred_ds.resize((new_size,))
        pred_ds[-1] = current_prediction

        h5f.close()


if __name__ == "__main__":
    main()
