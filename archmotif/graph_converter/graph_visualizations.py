import pydot
from IPython.display import Image, display
from scipy.cluster.hierarchy import dendrogram
import numpy as np


def view_pydot(pdot):
    """
    Returns a rendered image of a Dot (graphviz) object

    :param pdot: graph object
    """
    plt = Image(pdot.create_png())
    display(plt) # use display function from IPython to render the image inside the notebook


def save_pydot(pdot, filename='', format="png"):
    """
    Save a Dot Graph visualization

    :param pdot: Dot Object
    :param filename: path and file name to store the visualization
    :param format: supported image extension
    """

    OUTPUT_FORMATS = {"jpe","jpeg","jpg","pdf","png","svg"}
    if format not in OUTPUT_FORMATS:
        raise NotImplementedError(f"Format {format} is not supported")

    if filename:
        if not filename.endswith(f".{format}"):
            filename += f".{format}"
        
        dot.write(filename, format=format)
    else:
        print("Filename was not provided")


def plot_dendrogram(model, **kwargs):
    """
    code example from: https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    """
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)