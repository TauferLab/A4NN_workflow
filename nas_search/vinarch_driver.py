# -------------------------------------------------- #
# Driver code for the Similarity Engine (Vinarch)    #
# -------------------------------------------------- #

import sys
import os
import time
import argparse
import json
import pandas as pd 
import h5py
from pathlib import Path

# Update project root path
sys.path.insert(0, os.environ['NSGA_NET_PATH']) 

from vinarch.graph_converter.nsganet_utils import build_genome_edgelist
from vinarch.graph_converter.convert import from_edgelist_to_nx
from vinarch.graph_distance.utils import flatten_list
from nsganet.models import macro_encoding
from vinarch.graph_distance.compute_distance_metrics import  get_dataframe_for_distance_metrics, get_dataframe_for_graph_kernels, metric_normalization, aggregate_metrics

def main(raw_args=None):

    # Read configuration file arguments
    parser = argparse.ArgumentParser(description='Vinarch Analysis')
    parser.add_argument('--config', dest='vinarch_config', type=argparse.FileType('r')) # path to configuration file
    config = parser.parse_args(raw_args)

    # Extract arguments
    vinarch_config = config.vinarch_config # extract vinarch configuration

    # Opening JSON configuration file
    vinarch_args=json.load(vinarch_config) # load the vinarch arguments from the path
    path_to_results = vinarch_args['io_params']['save_dir']

    if "vinarch_params" in vinarch_args:
        dict_params = vinarch_args['vinarch_params']
        kernel_params = dict_params['graph_kernels']
        distance_params = dict_params['distance_metrics']
        similarity_t = dict_params['threshold']
        is_metric_similarity = dict_params['is_metric_similarity']
        metric_norm = dict_params['metric_norm']
        comp_window = dict_params['comparison_window']
    else:
        raise Exception("Empty Vinarch Parameters")

    # Path to where the global vinarch file is stored
    global_file_path = os.path.join(path_to_results, 'global_vinarch.h5')


    # Open the global vinarch file to retrieve the model structures
    h5f = h5py.File(global_file_path,'r')
    num_genomes = len(h5f['genome']) # Number of saved structures
    
    # Run the similarity engine if we have more than one saved structure
    if num_genomes > 1:
        genomes = h5f['genome'][:].tolist()
        ids = [i.decode('utf-8') for i in h5f['id'][:].tolist()]
        h5f.close() # close global file

        # If -1, then compare current structure to all previously trained models
        if comp_window == -1: 
            source_nn = genomes[-1] # source is the current model structure
            target_nns = genomes[:-1] # targets are all previous models
            target_ids = ids[:-1] 
        else: # Otherwise, compare with the most N recent models
            offset = comp_window + 1
            source_nn = genomes[-1] 
            target_nns = genomes[-offset:-1] 
            target_ids = ids[-offset:-1] 

        source_id = ids[-1] # obtain ID of current model
        
        # Path to store similarity results for the current model
        path_to_local_vinarch = os.path.join(path_to_results, source_id, 'local_vinarch.h5')

        # Start time for running vinarch similarity
        vinarch_start = time.perf_counter()

        
        # Step 1: Decode genome structure to an edge list
        source_nn = macro_encoding.decode(source_nn)
        target_nns = [macro_encoding.decode(i) for i in target_nns]
        # Step 2: convert edge lists to networkx objects
        nx_source = from_edgelist_to_nx(build_genome_edgelist(source_nn), convert_labels_to_ints=True)
        nx_targets = [from_edgelist_to_nx(build_genome_edgelist(i), convert_labels_to_ints=True) for i in target_nns]

        # Step 3: Compute graph distances and string distances, if any
        df_graph_metrics = get_dataframe_for_graph_kernels(nx_source, nx_targets, kernel_params=kernel_params)

        source_flatten = flatten_list(source_nn)
        target_flatten = [flatten_list(i) for i in target_nns]
        df_string_metrics = get_dataframe_for_distance_metrics(source_flatten, target_flatten, distance_params)

        # Step 4: Concatenate results for each metric and normalize, if needed
        df = pd.concat([df_graph_metrics, df_string_metrics], axis=1) 
        df = metric_normalization(df)
        df['targetID'] = target_ids

        if metric_norm:
            use_cols = [i for i in df.columns if 'norm' in i]
        else:
            use_cols = [i for i in df.columns if all(word not in i for word in ['norm', 'ID'])]
        
        sim_results = aggregate_metrics(df, columns=use_cols)
        df['sim_result'] = sim_results

        # Set similarity threshold based on whther the metric is similarity or distance
        if is_metric_similarity: 
            df['is_similar'] = df['sim_result'] >= similarity_t
        else: 
            df['is_similar'] = df['sim_result'] <= similarity_t

        # Sort the results by most similar model
        if is_metric_similarity:
            df = df.sort_values('sim_result', ascending=False) # from most similar to less similar
        else:
            df = df.sort_values('sim_result', ascending=True) # from lower distance to higher distance

        vinarch_end = time.perf_counter() # End time for running vinarch similarity
        vinarch_time = vinarch_end - vinarch_start

        # Step 5: save similarity results to h5 file for the current model
        h5f = h5py.File(path_to_local_vinarch, 'w')
        for col in df.columns: 
            data = df[col].values
            h5f.create_dataset(col, data=data, maxshape=(None,))
        h5f.create_dataset('vinarch_times', data=[vinarch_time], maxshape=(None,))
        h5f.close()

    else:
        h5f.close() # close global file


if __name__ == "__main__":
    main()
