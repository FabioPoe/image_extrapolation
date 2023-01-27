"""
Author: Fabio Pöschko
Matr.Nr.: K11905017
Exercise 5
"""

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from matplotlib.image import imread

# importing the ex4 again
def ex4(image_array: np.ndarray, border_x: tuple, border_y: tuple):
    original_img = image_array
    """See assignment sheet for usage description"""
    if not isinstance(image_array, np.ndarray) or image_array.ndim != 2:
        raise NotImplementedError("image_array must be a 2D numpy array")

    border_x_start, border_x_end = border_x
    border_y_start, border_y_end = border_y

    try:  # Check for conversion to int (would raise ValueError anyway but we will write a nice error message)
        border_x_start = int(border_x_start)
        border_x_end = int(border_x_end)
        border_y_start = int(border_y_start)
        border_y_end = int(border_y_end)
    except ValueError as e:
        raise ValueError(f"Could not convert entries in border_x and border_y ({border_x} and {border_y}) to int! "
                         f"Error: {e}")

    if border_x_start < 1 or border_x_end < 1:
        raise ValueError(f"Values of border_x must be greater than 0 but are {border_x_start, border_x_end}")

    if border_y_start < 1 or border_y_end < 1:
        raise ValueError(f"Values of border_y must be greater than 0 but are {border_y_start, border_y_end}")

    remaining_size_x = image_array.shape[0] - (border_x_start + border_x_end)
    remaining_size_y = image_array.shape[1] - (border_y_start + border_y_end)
    if remaining_size_x < 16 or remaining_size_y < 16:
        raise ValueError(f"the size of the remaining image after removing the border must be greater equal (16,16) "
                         f"but was ({remaining_size_x},{remaining_size_y})")

    # Create known_array
    known_array = np.zeros_like(image_array)
    known_array[border_x_start:-border_x_end, border_y_start:-border_y_end] = 1

    # Create target_array - don't forget to use .copy(), otherwise target_array and image_array might point to the
    # same array!
    target_array = image_array[known_array == 0].copy()

    # Use image_array as input_array
    image_array[known_array == 0] = 0

    return image_array, known_array, target_array, original_img


from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset, Subset
import torchvision
import random
import h5py


class RandomSeqDataset(Dataset):
    def __init__(self,path,key):
        """Here we define our __init__ method. In this case, we will take two
        arguments, the sequence length `sequence_length` and the number of
        features per sequence position `n_features`.
        """
        # super().__init__()  # Optional, since Dataset.__init__() is a no-op
        with h5py.File(path, 'r') as f:
            data = f[key][:]
        if torch.cuda.is_available():
            self.data = torch.FloatTensor(data).cuda()
        else:
            self.data = data


    def __getitem__(self, index):
        """ Here we have to create a random sample and add the signal in
        positive-class samples. Positive-class samples will have a label "1",
        negative-class samples will have a label "0".
        """
        # grabing a sample because of their index
        # reshaping it because every training pic is 90x90
        # and then getting the data i will feed the network
        input_array, original_array = self.data[index]
        return input_array, original_array

    def __len__(self):
        """ Optional: Here we can define the number of samples in our dataset

        __len__() should take no arguments and return the number of samples in
        our dataset
        """
        return len(self.data)


# We can start by building the logic of converting-and-stacking-if-possible:
def stack_or_not(something_to_stack: list):
    """This function will attempt to stack `something_to_stack` and convert it
    to a tensor. If not possible, `something_to_stack` will be returned as it
    was.
    """
    try:
        # Convert to tensors (TypeError if fails)
        tensor_list = [torch.tensor(s) for s in something_to_stack]
        # Try to stack tensors (RuntimeError if fails)
        stacked_tensors = torch.stack(tensor_list, dim=0)
        return stacked_tensors
    except (TypeError, RuntimeError):
        return something_to_stack


# And now, we use it in our collate_fn
def stack_if_possible_collate_fn(batch_as_list: list):
    """Function to be passed to torch.utils.data.DataLoader as collate_fn

    Will stack samples to mini-batch if possible, otherwise returns list
    """
    # Number of entries per sample-tuple (e.g. 3 for features, labels, IDs)
    n_entries_per_sample = len(batch_as_list[0])
    # Go through all entries in all samples and apply our stack_or_not()
    list_batch = [stack_or_not([sample[entry_i] for sample in batch_as_list])
                  for entry_i in range(n_entries_per_sample)]
    # Return the mini-batch
    return list_batch