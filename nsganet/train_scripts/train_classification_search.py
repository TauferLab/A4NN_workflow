# ---------------------------------------------------------------------------------------------------------
# Script for training the evolutionary NSGA-Net models
# ---------------------------------------------------------------------------------------------------------

import sys
import os
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch
import random

import torch.backends.cudnn as cudnn
import pandas as pd
import h5py
import csv
import copy

import time
from nsganet.misc import utils
from nsganet.misc import save_model

# import utilities to count number of flops
from nsganet.misc.flops_counter import add_flops_counting_methods

# set device to GPU if available
if torch.cuda.is_available():
    device = "cuda"
else:
    device="cpu"


def main(genome, 
         epochs, 
         search_space='macro',
         data_root='/home/username/datasets/', 
         save='arch_1', 
         expr_root='/home/username/models',
         seed=0, 
         gpu=0, 
         init_channels=24,
         layers=11, 
         auxiliary=False, 
         cutout=False, 
         drop_path_prob=0.0, 
         dataset=None, 
         penguin_args=None,
         vinarch_args=None, 
         save_models=False):
    
    """
    Training, evolution and inference script for NNs

    :param genome: the bit-string representing the connections of the network
    :param search_space: either macro or micro search
    :param data_root: path to download(ed) raw data
    :param save: architecture ID
    :param expr_root: path to save models/experiments
    :param seed: 
    :param gpu: the GPU ID
    :param init_channels: the number of hidden channels for the output data
    :param layers: num of layers for micro search
    :param auxiliary: bool
    :param cutout: bool to indicate whether to use gradient clipping
    :param drop_path_prob: dropout probability
    :param dataset: name of the dataset to train on 
    :param penguin_args: corresponding penguin parameters
    :param vinarch_args: corresponding vinarch parameters
    :param save_models: bool to indicate if saving models
    """

    # ---- train logger ----------------- #
    save_pth = os.path.join(expr_root, '{}'.format(save)) # define path to save new architecture
    utils.create_exp_dir(save_pth) # create directories to save experiments

    peng_pth = os.path.join(save_pth, 'penguin.h5') # define the path to the penguin data
    train_res_pth = os.path.join(save_pth, 'training.h5') # define the path to the training results
    global_vinarch_pth = os.path.join(expr_root, 'global_vinarch.h5') # define the path to the global vinarch file
    local_vinarch_pth = os.path.join(save_pth, 'local_vinarch.h5') # define the path to the local vinarch file

    # ---- Define global variables --------#
    train_acc = None # store training acc
    valid_acc = None # store validation acc
    epoch_time = None # store the time to complete each epoch

    # ---- Read Penguin parameters ------ #
    use_penguin = penguin_args['penguin'] # flag whether to use penguin or not
    peng_stop_if_converged = penguin_args['stop_if_converged'] # flag whether to stop training or not
    peng_freq = penguin_args['peng_freq'] # run peng every N iterations
    is_converged = False

    # ---- Read Vinarch parameters ------ #
    use_vinarch = vinarch_args['vinarch'] # flag whether to use Vinarch or not
    vinarch_stop_if_converged = vinarch_args['stop_if_converged'] # flag whether to stop training or not
    is_similar = False

    # ---- parameter values setting ----- #
    batch_size = 128 # the number of batches while training
    learning_rate = 0.025
    momentum = 0.9
    weight_decay = 3e-4
    cutout_length = 16
    auxiliary_weight = 0.4
    grad_clip = 5
    report_freq = 50
    train_params = {
        'auxiliary': auxiliary,
        'auxiliary_weight': auxiliary_weight,
        'grad_clip': grad_clip,
        'report_freq': report_freq,
    }

    # ---- Create data loaders --- # 
    train_queue, valid_queue, data_args = utils.return_dataloaders(dataset, 
                                                                   data_root=data_root,
                                                                   batch_size=batch_size, 
                                                                   cutout=cutout, 
                                                                   cutout_length=cutout_length)

    # ---- Create model object --- #
    model, genotype = utils.return_architecture(search_space,
                                                genome, 
                                                data_args=data_args, 
                                                init_channels=init_channels,
                                                micro_layers=layers, 
                                                micro_auxiliary=auxiliary)

    print(f"Architecture ID = {save}; genome = {genome}")

    # --------- Count number of parameters in the model ---------------- #
    n_params = np.sum(np.prod(v.size()) for v in filter(lambda p: p.requires_grad, model.parameters())) 
    
    # --------- Calculate flops during a forward pass ------------------ #
    flops_fn = os.path.join(save_pth, 'flops.csv') # store flops in a separate file
    flops_model = copy.deepcopy(model)
    n_flops = 0.0
    flops_model = flops_model.to(device)
    flops_model.eval()
    with torch.no_grad():
        flops_model = add_flops_counting_methods(flops_model)
        flops_model.start_flops_count()
        data_shape1, data_shape2 = data_args['data_shape']
        random_data = torch.randn(1, data_args['im_channels'], data_shape1, data_shape2)
        flops_model(torch.autograd.Variable(random_data).to(device))
        n_flops = np.round(flops_model.compute_average_flops_cost() / 1e6, 4)

    with open(flops_fn, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([n_flops])
    print(f"n_flops = {n_flops}")
    del flops_model # release memory for copied model
    del random_data
    torch.cuda.empty_cache()
    
    
    # -------- Send model to Device --------- #
    model = model.to(device) 


    # ------ Define the loss function and optimizers ----------- #
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        learning_rate,
        momentum=momentum,
        weight_decay=weight_decay
    )
    
    # ------ Define a learning rate scheduler --------- #
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(epochs))


    # ------ Save model structure (genome) in a Global file to be processed by Vinarch --------- #
    if save == 'arch_1': # If we are in the first model, then create the global file
        h5f = h5py.File(global_vinarch_pth, 'w')
        h5f.create_dataset('id', data=[save], maxshape=(None,))
        h5f.create_dataset('genome', data=[genome], maxshape=(None,len(genome), len(genome[0])))
        h5f.close()
    else: # Otherwise, append the new model to the global file
        h5f = h5py.File(global_vinarch_pth, 'a')
        new_genome = h5f['genome']
        new_size = new_genome.shape[0] + 1
        new_genome.resize((new_size,len(genome), len(genome[0])))
        new_genome[-1] = genome

        new_id = h5f['id']
        new_size = new_id.shape[0] + 1
        new_id.resize((new_size,))
        new_id[-1] = save

        h5f.close()


    # ------ Start Model Training --------------------- #
    for epoch in range(epochs):
        # --------------------------------------------------#
        # If True, call out the Similarity Engine (Vinarch) #
        # --------------------------------------------------#
        if use_vinarch:
            # Run the similarity engine for all models before training begins, except the first one
            if (save != 'arch_1') and (epoch == 0): 
                # Open the local vinarch file for the current model to retrieve structural similarity data
                h5f = h5py.File(local_vinarch_pth,'r')
                similar_ds = h5f['is_similar'][:]
                similar_ids = h5f['targetID'][:]
                h5f.close()

                # Check if there exist a NN with similar structure than the current one
                similar_ds = np.flatnonzero(similar_ds)
                if len(similar_ds) > 0:
                    is_similar=True
                    similar_idx = similar_ds[0] # get index of architecture
                    similar_id = similar_ids[similar_idx].decode('utf-8') # get the architecture ID
                    print(f"Found similar architecture for {save}: {similar_id}")

                    # Open the training H5 file of the similar architecture to retrieve its validation accuracy and FLOPs
                    similar_res_pth = os.path.join(expr_root, similar_id, 'training.h5')
                    h5f = h5py.File(similar_res_pth, 'r')
                    val_ds = h5f['val_accs'][:].tolist()
                    vinarch_prediction = val_ds[-1]
                    vinarch_train = h5f['train_accs'][:].tolist()[-1]
                    vinarch_flops = h5f['n_flops'][:][-1]
                    vinarch_epoch = h5f['epoch'][:].tolist()[-1]
                    h5f.close()

                    if vinarch_stop_if_converged: # create a training file for the current model using the retrieved information
                        h5f = h5py.File(train_res_pth, 'w')
                        h5f.create_dataset('epoch', data=[-1], maxshape=(None,)) 
                        h5f.create_dataset('train_accs', data=[vinarch_train], maxshape=(None,))
                        h5f.create_dataset('val_accs', data=[vinarch_prediction], maxshape=(None,))
                        h5f.create_dataset('n_flops', data=[vinarch_flops], maxshape=(None,))
                        h5f.close()
                        break # break from training loop
        # --------------------------------------------- #
        # End of Similarity Engine                      #
        # --------------------------------------------- #


        # --------------------------------------------------#
        # If True, call out the Predictive Engine (Penguin) #
        # --------------------------------------------------#
        if use_penguin:
            # Check iteration frequency, run the predictive engine, and skip for the first iteration
            if (epoch % peng_freq == 0) and epoch != 0:

                # Open the local penguin file and retrieve the prediction results for the current model
                h5f = h5py.File(peng_pth,'r')
                pred_ds = h5f['predictions']
                predictions = pred_ds[:].tolist()
                peng_prediction = predictions[-1]
                conv_ds = h5f['converged']
                convs = conv_ds[:].tolist()
                is_converged = convs[-1] # flag that indicates if predictions have converged or not
                h5f.close()

                if is_converged:
                    if peng_stop_if_converged: # stop training if user set the flag to True
                        print('converged and terminated training after epoch {}'.format(str(epoch)))
                        break # break from training loop
                    else: # Otherwise, continue training as usual
                        is_converged=False 
                        print('converged after epoch {}'.format(str(epoch)))
        # ---------------------------------------------#
        # End of Predictive Engine                     #
        # ---------------------------------------------#


        # ---------------------------------------------#
        # Start of Model Training                      #
        # ---------------------------------------------#
        epoch_start = time.perf_counter() # start time of training 

        # Dropout probability 
        model.droprate = drop_path_prob * epoch / epochs

        # Train and Validate the model
        train_acc, train_loss, valid_acc, valid_loss = utils.train_and_val(train_queue,
                                                                           model,
                                                                           criterion, 
                                                                           optimizer, 
                                                                           train_params, 
                                                                           valid_queue, 
                                                                           device)
        
        # step the scheduler
        scheduler.step()

        print('train_acc:', train_acc)
        print('train_loss:', train_loss)
        print('val_acc:', valid_acc)
        print('val_loss:', valid_loss)

        # end time of training
        epoch_end = time.perf_counter()
        epoch_time = epoch_end-epoch_start
        # ---------------------------------------------#
        # End of Model Training                        #
        # ---------------------------------------------#


        # ---------------------------------------------#
        # Save NN training results per epoch           #
        # ---------------------------------------------#
        if save_models:
            save_model.model_package(model, save_pth, save, epoch)


        # Create a file to store training results for the current epoch
        if epoch==0: 
            h5f = h5py.File(train_res_pth, 'w')
            h5f.create_dataset('epoch', data=[epoch], maxshape=(None,))
            h5f.create_dataset('train_accs', data=[train_acc], maxshape=(None,))
            h5f.create_dataset('epoch_time', data=[epoch_time], maxshape=(None,))
            h5f.create_dataset('val_accs', data=[valid_acc], maxshape=(None,))
            h5f.create_dataset('n_flops', data=[n_flops], maxshape=(None,), dtype="float64") # create a field for n_flops
            h5f.close()
        else: # If the file already exists, append a new row of results for the current epoch
            h5f = h5py.File(train_res_pth, 'a')
            epoch_ds = h5f['epoch']
            new_size = epoch_ds.shape[0] + 1
            epoch_ds.resize((new_size,))
            epoch_ds[-1] = epoch

            train_ds = h5f['train_accs']
            new_size = train_ds.shape[0] + 1
            train_ds.resize((new_size,))
            train_ds[-1] = train_acc

            time_ds = h5f['epoch_time']
            new_size = time_ds.shape[0] + 1
            time_ds.resize((new_size,))
            time_ds[-1] = epoch_time

            val_ds = h5f['val_accs']
            new_size = val_ds.shape[0] + 1
            val_ds.resize((new_size,))
            val_ds[-1] = valid_acc

            h5f.close()

    # Read the local penguin file one final time at the last epoch or if the predictive engine fails to converge
    if (not is_converged) and (epoch == (epochs-1)):
        h5f = h5py.File(peng_pth,'r')
        conv_ds = h5f['converged']
        convs = conv_ds[:].tolist()
        is_converged = convs[-1]
        h5f.close()

    # Return Results to NAS
    if (use_vinarch == True) and (is_similar==True):
        return {
            'valid_acc': vinarch_prediction,
            'params': n_params,
            'flops': vinarch_flops
        }
    elif (use_penguin == True) and (peng_stop_if_converged == True) and (is_converged == True):
        return {
                'valid_acc': valid_acc,
                'params': n_params,
                'flops': n_flops
            }
    else:
        return {
            'valid_acc': valid_acc,
            'params': n_params,
            'flops': n_flops
        } 


if __name__ == "__main__":
    DARTS_V2 = [[[[3, 0], [3, 1]], [[3, 0], [3, 1]], [[3, 1], [2, 0]], [[2, 0], [5, 2]]],
               [[[0, 0], [0, 1]], [[2, 2], [0, 1]], [[0, 0], [2, 2]], [[2, 2], [0, 1]]]]
    start = time.time()
    print(main(genome=DARTS_V2, epochs=20, save='DARTS_V2_16', seed=1, init_channels=16,
               auxiliary=False, cutout=False, drop_path_prob=0.0))
    print('Time elapsed = {} mins'.format((time.time() - start)/60))
    # start = time.time()
    # print(main(genome=DARTS_V2, epochs=20, save='DARTS_V2_32', seed=1, init_channels=32))
    # print('Time elapsed = {} mins'.format((time.time() - start) / 60))
