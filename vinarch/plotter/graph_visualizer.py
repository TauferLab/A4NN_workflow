
import pydot
from IPython.display import Image, display
from scipy.cluster.hierarchy import dendrogram
import numpy as np


def view_pydot(pdot):
    """
    Returns a rendered image of a Dot (graphviz) object

    :param pdot: pydot graph object
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

