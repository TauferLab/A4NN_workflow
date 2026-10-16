import igraph
import numpy as np
# import graphkernels.kernels as gk
import networkx as nx
import itertools
from scipy.spatial.distance import squareform

def node_name_match(n1, n2):
    """
    :param n1: dictionary of attributes for node 1
    :param n2: dictionary of attributes for node 2
    :return True if same name otherwise False
    """
    return n1['name'] == n2['name']

def approximate_graph_edit_distance(G1, G2):
    """
    Returns an approximation of the graph edit distance
    :param G1: networkx object
    :param G2: networkx object
    :return approximated graph edit distance
    """

    # node_match takes a function that returns whether two nodes should be considered equal
    # Since we have a DAG with orderings we can compare their name attributes
    return next(nx.optimize_graph_edit_distance(G1, G2, node_match=node_name_match))

def compute_ged_kernel_distance(graphs):
    """
    Computes the graph edit distance and returns a kernel distance matrix

    :param graphs: list of networkx objects
    :return a numpy array for kernel distances
    """

    # iterate over pairwise combinations of graphs
    distances = []
    for g1, g2 in itertools.combinations(graphs, 2):
        distances.append(approximate_graph_edit_distance(g1, g2))

    # Return the square matrix of the condensed-vector of distances
    return squareform(distances)


def relabel_for_wlst_kernel( graph, label="dummy" ):
    """
    Relabel a igraph object for comparison using WLST kernel from graphkernels library

    :param graph: igraph object
    :param label: name of node attribute to use for the graph
    :return a relabeled igraph object
    """
    
    # Copy the graph
    n_vertices = len(graph.vs[:])
    relabeled_graph = igraph.Graph(n_vertices, directed=graph.is_directed())
    # If a label was specified, use it. Otherwise, put no label so that the 
    # WL kernel impl. supplies its own labels
    if label == "dummy":
        labels = [ 0 ] * len(graph.vs[:])
    elif label == "random":
        labels = [ np.random.randint(len(graph.vs[:])) ] * len(graph.vs[:])
    else:
        # Weisfeiler-Lehman kernel impl. in graphkernels only accepts 
        # integer-valued labels
        try:
            label_val = int(graph.vs[0][label])
            # Apply labels directly to the copy's vertices
            labels = [ int(label_val) for label_val in graph.vs[:][label] ]
        except:
            # Translate label values of other types to ints first
            label_set = set(graph.vs[:][label])
            label_to_int = { L:I for L,I in zip(label_set,range(len(label_set))) }
            labels = [ label_to_int[label_val] for label_val in graph.vs[:][label] ]
    relabeled_graph.vs[:]["label"] = labels
    # Add edges
    edges = [ (e.source, e.target) for e in graph.es[:] ]
    relabeled_graph.add_edges( edges )
    return relabeled_graph


def relabel_for_sp_kernel( graph, label="dummy" ):
    """
    Relabel a igraph object for comparison using Shortest path kernel from graphkernels library

    :param graph: igraph object
    :param label: name of node attribute to use for the graph
    :return a relabeled igraph object
    """
    
    # Copy the graph
    n_vertices = len(graph.vs[:])
    # convert graph to undirected
    relabeled_graph = igraph.Graph(n_vertices, directed=False)
    # If a label was specified, use it. Otherwise, put no label so that the 
    # WL kernel impl. supplies its own labels
    if label == "dummy":
        labels = [ 0 ] * len(graph.vs[:])
    elif label == "random":
        labels = [ np.random.randint(len(graph.vs[:])) ] * len(graph.vs[:])
    else:
        # kernel impl. in graphkernels only accepts 
        # integer-valued labels
        try:
            label_val = int(graph.vs[0][label])
            # Apply labels directly to the copy's vertices
            labels = [ int(label_val) for label_val in graph.vs[:][label] ]
        except:
            # Translate label values of other types to ints first
            label_set = set(graph.vs[:][label])
            label_to_int = { L:I for L,I in zip(label_set,range(len(label_set))) }
            labels = [ label_to_int[label_val] for label_val in graph.vs[:][label] ]
    relabeled_graph.vs[:]["label"] = labels
    # Add edges
    edges = [ (e.source, e.target) for e in graph.es[:] ]
    relabeled_graph.add_edges( edges )
    return relabeled_graph


def get_relabeled_graphs(graphs, kernels):
    """
    Returns a dictionary of relabeled graphs for each kernel supported

    :param graphs: list of igraph objects
    :param kernels: a list of dictionary objects including graph kernel and corresponding parameters
    :return a dictionary of relabeled graphs
    """
    all_relabeled_graphs = {}
    for kernel in kernels:
        # Name of graph kernel
        name = kernel["name"]
        # Which label are we using?
        try:
            label = kernel["params"]["label"]
        except:
            err_msg = f"Requested re-labeling for kernel: {name} not possible, label not specified"
            raise KeyError (err_msg)
        # Define a key that uniquely identifies this kernel / label pair
        key = ( name, label )
        # Only generate a relabeled set of graphs if needed
        # The list of kernels may include multiple distinct kernels that can 
        # use the same relabeled graphs (e.g., multiple WL kernels with the same
        # label but different numbers of WL iterations)
        if key not in all_relabeled_graphs:
            # Relabel for Weisfeiler-Lehman Subtree-Pattern kernel
            if name == "wlst":
                relabeled_graphs = [relabel_for_wlst_kernel(g, label) for g in graphs]
            # Relabel for Shortest-path length kernel
            if name == "sp": 
                relabeled_graphs = [relabel_for_sp_kernel(g, label) for g in graphs]
            all_relabeled_graphs[key] = relabeled_graphs
    
    return all_relabeled_graphs


def convert_to_distance_matrix( kernel_mat ):
    """
    Convert a similarity matrix to a kernel distance matrix
    The kernel distance matrix is computed based on the self-similarities and the cross-similarities of the similarity matrix
    https://arxiv.org/pdf/1103.1625

    :param kernel_mat: similarity matrix
    :return dist_mat: distance matrix
    """
    n_rows = len( kernel_mat )
    n_cols = len( kernel_mat[0] )
    assert( n_rows == n_cols )
    dist_mat = np.zeros( ( n_rows, n_cols ) )
    for i in range(n_rows):
        for j in range(n_cols):
            self_similarity = kernel_mat[i][i] + kernel_mat[j][j]
            cross_similarity = 2*kernel_mat[i][j]
            kernel_distance = np.sqrt( self_similarity - cross_similarity )
            dist_mat[i][j] = kernel_distance
    return dist_mat


def compute_kernel_distance_matrices(graphs, kernel_params):
    """
    Return the kernel distance matrices

    :param graphs: list of igraph objects 
    :param kernel_params: a list of dictionary objects including graph kernel and corresponding parameters
    :return a dictionary of kernel distance matrices
    """

    # relabel graphs based on requested graph kernels
    kernel_label_pair_to_relabeled_graphs = get_relabeled_graphs(graphs, kernel_params)

    # To store the kernel distance matrices
    kernel_to_distance_matrix = {}
    for kp in kernel_params:
        kernel = kp["name"]
        params = kp["params"]
        # Compute Weisfeiler-Lehman subtree pattern kernel
        if kernel == "wlst":
            n_iters = params["n_iters"]
            label   = params["label"]
            kernel_label_pair = ( kernel, label )
            relabeled_graphs = kernel_label_pair_to_relabeled_graphs[ kernel_label_pair ]
            kernel_mat = gk.CalculateWLKernel( relabeled_graphs, n_iters )
            distance_mat = convert_to_distance_matrix( kernel_mat )
            kernel_params_key = ( kernel, label, n_iters )
            kernel_to_distance_matrix[ kernel_params_key ] = distance_mat
        elif kernel == "sp":
            label   = params["label"]
            kernel_label_pair = ( kernel, label )
            relabeled_graphs = kernel_label_pair_to_relabeled_graphs[ kernel_label_pair ]
            kernel_mat = gk.CalculateShortestPathKernel( relabeled_graphs )
            distance_mat = convert_to_distance_matrix( kernel_mat )
            kernel_params_key = ( kernel, label )
            kernel_to_distance_matrix[ kernel_params_key ] = distance_mat
        else:
            raise NotImplementedError(f"Kernel: {kernel} not supported")
    return kernel_to_distance_matrix


