from vinarch.graph_distance.graph_kernels import approximate_graph_edit_distance
from vinarch.graph_distance import string_metrics 

import pandas as pd
import numpy as np
from scipy import stats

def get_dataframe_for_graph_kernels(source_nn, target_nns, kernel_params):
    '''
    Compute graph kernel distances in kernel_params between the source_nn and target_nns

    :param source_nn: networkx object
    :param target_nns: list of networkx objects
    :param kernel_params: dict of graph kernel parameters
    
    :return dataframe where each column is a metric
    '''
    
    data = {}
    for v in kernel_params:
        if v['name'] == 'ged':
            dist_arr = []
            colname = v['name']+'-distance'
            for target in target_nns:
                dist_arr.append(approximate_graph_edit_distance(source_nn, target))
            data[colname] = dist_arr
        else:
            continue
        
    return pd.DataFrame.from_dict(data)


def get_dataframe_for_distance_metrics(source_str, target_strs, metrics):
    '''
    Compute string-related distances between source and targets

    :param source_nn: list of binary digits
    :param target_nns: list of lists of binary digits
    :param metrics: list of strings for metrics
    
    :return dataframe of distance metrics
    '''
    columns = None
    df = pd.DataFrame()
    for metric in metrics:
        if metric == 'euclidean':
            res = string_metrics.compute_euclidean_distance
            columns=metric+'-distance'
        elif metric == 'manhattan':
            res = string_metrics.compute_manhattan_distance
            columns=metric+'-distance'
        elif metric == 'lcs':
            res = string_metrics.longest_common_subsequence
            columns=[metric+'-distance', 'norm_'+metric+'-distance']
        elif metric == 'aligned_lcs':
            res = string_metrics.longest_nonzero_aligned_subsequence
            columns=[metric+'-distance', 'norm_'+metric+'-distance']
        else:
            raise NotImplementedError(f"Metric: {metric} is not implemented")
    
        distances = [res(source_str, target_str) for target_str in target_strs]
        df[columns] = distances    
    return df


def metric_normalization(df):
    '''
    Normalize metrics using specific normalization techniques
    '''
    
    cols = df.columns
    
    for col in cols:
        if ('norm' in col) or ('lcs' in col):
            continue

        if 'ged' in col or 'wlst' in col:
            # normalize using exponential decay to make differences decay slowly
            df['norm_'+col] = np.exp(-0.05 * df[col])
        else:
            # normalize using exponential decay to make differences decay faster
            df['norm_'+col] = np.exp(-0.5 * df[col])
    return df


def aggregate_metrics(df, mode='arithmetic', columns=None):
    '''
    Return aggregated distance metric - higher is more similar

    :param mode: arithmetic or geometric
    '''
    if columns:
        cols = columns
    else:
        cols = [i for i in df.columns if 'norm' in i]
    if mode == 'arithmetic':
        mean = df[cols].mean(axis=1).values
    elif mode == 'geometric':
        mean = stats.gmean(df[cols], axis=1)
    else:
        raise NotImplementedError(f"Mode: {mode} not implemented")
    
    return mean