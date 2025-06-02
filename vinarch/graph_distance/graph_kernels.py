
import numpy as np
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
    Computes the graph edit distance and returns a kernel distance matrix - lower is more similar

    :param graphs: list of networkx objects
    :return a numpy array for kernel distances
    """

    # iterate over pairwise combinations of graphs
    distances = []
    for g1, g2 in itertools.combinations(graphs, 2):
        distances.append(approximate_graph_edit_distance(g1, g2))

    # Return the square matrix of the condensed-vector of distances
    return squareform(distances)