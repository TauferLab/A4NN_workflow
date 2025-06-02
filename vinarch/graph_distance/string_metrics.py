import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances


def compute_aligned_subsequences(arr1, arr2, min_len=1, return_df=False):
    '''
    Returns all aligned subsequence (i.e., a subsequence that appears at the 
    same positions) between two binary lists that are greater than or equal to
    the min_len argument
    
    :param arr1: list of binary values
    :param arr2: list of binary values
    :param min_len: int to return only sequences >= than min_len
    :param return_df: bool if True, then return a pandas dataframe otherwise return a nested list
    
    :return int: start index; int: sequence length, list: common sequence
    '''
    curr_seq_length = 0 # store the current sequence length
    start_index = 0 # store the start index of the subsequence
    tracking = False # flag to check if we are in a subsequence
    
    all_sequence_lengths = [] # store start index, seq length, actual sequence
    
    min_list_len = min(len(arr1), len(arr2)) # store the min length between arrays
    
    for i in range(min_list_len):
        if arr1[i] == arr2[i]:
            if tracking == False:
                start_index = i
            tracking = True 
            curr_seq_length += 1
        else:
            tracking = False
            if (curr_seq_length != 0) and (curr_seq_length >= min_len):
                all_sequence_lengths.append([start_index, curr_seq_length, arr1[start_index:start_index+curr_seq_length], curr_seq_length/min_list_len])
            # reset current sequence length
            curr_seq_length = 0
            
    if (curr_seq_length != 0) and (curr_seq_length >= min_len):
        all_sequence_lengths.append([start_index, curr_seq_length, arr1[start_index:start_index+curr_seq_length], curr_seq_length/min_list_len])

    if return_df:
        df = pd.DataFrame(all_sequence_lengths, columns=['start_index', 'seq_len', 'common_seq', 'norm_seq_len'])
        return df
    return all_sequence_lengths


def longest_nonzero_aligned_subsequence(arr1, arr2):
    '''
    Return a dictionary with information about the longest nonzero common subsequence
    between two arrays
    
    :param arr1: list 
    :param arr2: list
    :return int, float: the length of aligned subsequence and the normalized length
    '''
    # compute all subsequences
    df = compute_aligned_subsequences(arr1, arr2, min_len=1, return_df=True)
    df['sum'] = df['common_seq'].apply(sum) # count number of connections
    row = df[df['sum']>0].sort_values('seq_len', ascending=False)
    if len(row) > 0:
        dict_res = row.head(1).to_dict('records')[0]
        return dict_res['seq_len'], dict_res['norm_seq_len']
    return (0, 0)


def longest_common_subsequence(arr1, arr2):
    '''
    Return the longest common consecutive subsequence between the two lists.
    The subsequence does not have to occur at the same positions
    
    :param arr1, arr2: list of binary digits
    :return int, float: the length of the longest common subsequence and the normalized length
    '''
    
    len_arr1 = len(arr1)
    len_arr2 = len(arr2)
    max_len = max(len_arr1, len_arr2)
    
    # initialize array to store dynamic programming results
    arr = [0] * (len_arr2+1)
    
    # length of max common consecutive subsequence
    maxm = 0 
    
    # build LCS in bottom up fashion
    for i in range(len_arr1 - 1, -1, -1): # iterate over rows
        prev = 0
        for j in range(len_arr2 - 1, -1, -1): # iterate over columns
            temp = arr[j]
            
            if arr1[i] == arr2[j]:
                arr[j] = prev + 1
                maxm = max(maxm, arr[j])
            else:
                arr[j] = 0
            prev = temp
    
    norm_maxm = maxm/max_len
    return maxm, norm_maxm


def compute_euclidean_distance(arr1, arr2):
    '''
    Compute the euclidean distance between two lists

    :param arr1, arr2: flatten lists
    '''
    X = np.array(arr1).reshape(1, -1)
    Y = np.array(arr2).reshape(1, -1)
    return euclidean_distances(X,Y).item()


def compute_manhattan_distance(arr1, arr2):
    '''
    Compute the manhattan distance between two lists

    :param arr1, arr2: flatten lists
    '''
    X = np.array(arr1).reshape(1, -1)
    Y = np.array(arr2).reshape(1, -1)
    return manhattan_distances(X,Y).item()
