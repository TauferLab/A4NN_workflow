"""
Driver code for NAS
"""

import sys
import os
# update project root path before running
sys.path.insert(0, os.environ['NSGA_NET_PATH']) 

import time
import logging
import argparse
import json
from nsganet.misc import utils
import pandas as pd 

import torch
import numpy as np



# NSGANET libraries
from nsganet.models import ian_whale_nsganet as engine
from pymoo.optimize import minimize
from nsganet.models.nas_problem import NAS, do_every_generations

def main(raw_args=None):
    """
    
    :param args: dict-like object to configure run
    """
    # Read configuration file arguments
    parser = argparse.ArgumentParser(description='Multi-objetive Genetic Algorithm for NAS')
    parser.add_argument('--config', dest='config_file_path', type=argparse.FileType('r'))
    config_json = parser.parse_args(raw_args)

    # Opening JSON file
    args=json.load(config_json.config_file_path) # load the arguments from the path

    args_io = args['io_params']
    args_search = args["search_params"]
    args_macro = args['macro_params']
    args_train = args['train_params']
    args_penguin = args['penguin_params']
    args_vinarch = args['vinarch_params']

    # create unique path to store files
    path_to_save = args_io['save_dir']
    print("PATH TO SAVE", path_to_save)
    utils.create_exp_dir(path_to_save)
    args["io_params"]["save_path"] = path_to_save # create argument to hold path to store models
    

    np.random.seed(args_search['seed'])
    num_gpus = torch.cuda.device_count()
    print("num_gpus =", str(num_gpus))

    # setup NAS search problem using a modified GeneticCNN search space
    n_var = int(((args_macro["n_nodes"]-1)*args_macro["n_nodes"]/2 + 1)*args_macro["n_phases"])
    lb = np.zeros(n_var)
    ub = np.ones(n_var)

    # Instantiate NAS object
    problem = NAS(data_root=args_io["data_root"],
                  model_type='classification', 
                  n_var=n_var,
                  n_phases=args_macro["n_phases"],
                  search_space='macro',
                  elementwise=True,
                  n_obj=2, 
                  n_constr=0, 
                  lb=lb, 
                  ub=ub,
                  init_channels=args_train["init_channels"], 
                  epochs=args_train["epochs"], 
                  save_dir=args_io["save_path"], 
                  dataset=args_io["dataset"], 
                  save_models=args_io["save_models"], 
                  penguin_args=args_penguin, 
                  vinarch_args=args_vinarch)

    # configure the nsga-net method
    method = engine.nsganet(pop_size=args_search["pop_size"],
                            n_offsprings=args_search["n_offspring"],
                            eliminate_duplicates=True, 
                            save_history=True)

    # Run the optimization problem and terminate when we reach the number of generations
    res = minimize(problem,
                   method,
                   seed=args_search['seed'],
                   callback=do_every_generations,
                   termination=('n_gen', args_search["n_gens"]))

    return


if __name__ == "__main__":
    main()
