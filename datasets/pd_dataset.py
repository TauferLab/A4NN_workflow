import os
import torch
import numpy as np
from torch.utils.data import Dataset
import glob

from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt

class ProteinDataset(Dataset):
    """
    Dataset class for the protein diffraction dataset
    """

    def __init__(self, root_dir, train=True):
        """
        :param root_dir: the path to the directory where data is stored
        :param train: boolean - if True then load train split, else load test split
        
        """
        self.root_dir = root_dir
        self.train = train
        self.labels_map = {"1n0u":0, "1n0vc":1}

        if train:
            # load the train data
            self.data_path = os.path.join(root_dir, "images/trainset/")
        else:
            # load the test data
            self.data_path = os.path.join(root_dir, "images/testset/")

        # load paths to images
        self.image_list = self.__getDataFiles__()
    
    def __getitem__(self, index):
        
        single_image_path = self.image_list[index] # extract image path
        
        # Extract (label, image ID)
        conf_label, image_id = single_image_path.split("ef2_")[1].split("_ptm")
        
        # open image
        im = Image.open(single_image_path)
        im_array = np.array(im) # convert to numpy array
        
        # Returns F, which means the image is 32-bit floating point pixels
        c = len(im.getbands())
        
        # Resize image matrix to include the color channel. PD Images are greyscale, so there is only one channel
        # Output size is (1x127x127)
        # If color images, then output size must be (3x127x127) assuming original size is (127x127x3)
        im_array = np.expand_dims(im_array, axis=0) if c == 1 else np.transpose(im_array, axes=(-1, 0, 1))

        # Transform numpy to tensor of type float32
        im_tensor = torch.from_numpy(im_array).float()
        
        # Get label and transform to a tensor
        label = int(self.labels_map[conf_label])

        return im_tensor, label

    def __len__(self):
        return len(self.image_list)


    def __getDataFiles__(self, EXT='tiff'):
        """ 
        Returns a list of paths to images
        
        :param EXT: image file extension
        :return a list of paths to images
        """
        datafiles = glob.glob(os.path.join(self.data_path, '*.'+EXT))
        datafiles = [path for path in datafiles if not path.split("/")[-1].startswith(".")]

        return datafiles

    def show_example(self, index, figsize=(6,5)):
        """
        Show an example in the dataset
        
        :param index: the example index
        :return prints the image with corresponding label and ID
        """

        single_image_path = self.image_list[index] # extract image path

        # Extract (label, image ID)
        conf_label, image_id = single_image_path.split("ef2_")[1].split("_ptm")

        # open image
        im = Image.open(single_image_path)
        im_array = np.array(im) # convert to numpy array

        fig, ax = plt.subplots(1,figsize=figsize)
        ax.imshow(im_array)