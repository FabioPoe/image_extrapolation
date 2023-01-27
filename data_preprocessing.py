"""
Author: Fabio Pöschko
Matr.Nr.: K11905017
Exercise 5
"""

import os
from matplotlib.image import imread
import numpy as np
from tqdm import tqdm



#this is just the ex4 function from previously
def ex4(image_array: np.ndarray, border_x: tuple, border_y: tuple):
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

    return image_array, known_array, target_array



# the code below collects the files and for every file it transforms it to 90,90 format,
# then it chooses a random border and applies ex4 to the image
# then it collects the images in a train, test and validation array and saves them in a numpy file

import torch
import torchvision
import random

#splitting the dataset along the users, because users can have the same pictures up to 5 times i think
n_samples = 400
shuffled_indices = np.random.permutation(n_samples)
testset_inds = shuffled_indices[:int(n_samples / 10)]
validationset_inds = shuffled_indices[int(n_samples / 10):int(n_samples / 10) * 2]
trainingset_inds = shuffled_indices[int(n_samples / 10) * 2:]


# n is how often a picture is there with different border choices
# i choose 1 because anything bigger overwhelms my RAM
n = 1
train_arrays = []
with tqdm(total=400) as pbar:
    for folder in os.listdir(r"C:\Users\fabio\Google_Drive\AI_Studium\programing_python\ass2\ex5\data\dataset"):
        if int(folder) in trainingset_inds:
            filenames = os.listdir(
                os.path.join(r"C:\Users\fabio\Google_Drive\AI_Studium\programing_python\ass2\ex5\data\dataset", folder))
            for file in filenames:
                full_filename = os.path.join(
                    r"C:\Users\fabio\Google_Drive\AI_Studium\programing_python\ass2\ex5\data\dataset", folder, file)
                img = imread(full_filename) / 255
                image = torch.from_numpy(img)
                image = torch.unsqueeze(image, 0)
                transforms = torch.nn.Sequential(
                    torchvision.transforms.Resize(size=(90, 90))
                )
                transformed_img = transforms(image)
                np_trafo_img = np.array(transformed_img).reshape(90, 90)
                original_image = np_trafo_img.copy()

                valid_border_choices = [[5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9],
                                        [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9],
                                        [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], ]
                for c in range(0, n):
                    border = random.choice(valid_border_choices)
                    input_array, known_array, target_array = ex4(np_trafo_img, border, border)
                    # then save these arrays
                    train_arrays.append([input_array, original_image])
        pbar.update(1)
print("finished")

test_arrays = []
with tqdm(total=400) as pbar:
    for folder in os.listdir(r"C:\Users\fabio\Google_Drive\AI_Studium\programing_python\ass2\ex5\data\dataset"):
        if int(folder) in testset_inds:
            filenames = os.listdir(
                os.path.join(r"C:\Users\fabio\Google_Drive\AI_Studium\programing_python\ass2\ex5\data\dataset", folder))
            for file in filenames:
                full_filename = os.path.join(
                    r"C:\Users\fabio\Google_Drive\AI_Studium\programing_python\ass2\ex5\data\dataset", folder, file)
                img = imread(full_filename) / 255
                image = torch.from_numpy(img)
                image = torch.unsqueeze(image, 0)
                transforms = torch.nn.Sequential(
                    torchvision.transforms.Resize(size=(90, 90))
                )
                transformed_img = transforms(image)
                np_trafo_img = np.array(transformed_img).reshape(90, 90)
                original_image = np_trafo_img.copy()

                valid_border_choices = [[5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9],
                                        [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9],
                                        [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], ]

                for c in range(0, n):
                    border = random.choice(valid_border_choices)
                    input_array, known_array, target_array = ex4(np_trafo_img, border, border)
                    # then save these arrays
                    test_arrays.append([input_array, original_image])
        pbar.update(1)
print("finished")

validation_arrays = []
with tqdm(total=400) as pbar:
    for folder in os.listdir(r"C:\Users\fabio\Google_Drive\AI_Studium\programing_python\ass2\ex5\data\dataset"):
        if int(folder) in validationset_inds:
            filenames = os.listdir(
                os.path.join(r"C:\Users\fabio\Google_Drive\AI_Studium\programing_python\ass2\ex5\data\dataset", folder))
            for file in filenames:
                full_filename = os.path.join(
                    r"C:\Users\fabio\Google_Drive\AI_Studium\programing_python\ass2\ex5\data\dataset", folder, file)
                img = imread(full_filename) / 255
                image = torch.from_numpy(img)
                image = torch.unsqueeze(image, 0)
                transforms = torch.nn.Sequential(
                    torchvision.transforms.Resize(size=(90, 90))
                )
                transformed_img = transforms(image)
                np_trafo_img = np.array(transformed_img).reshape(90, 90)
                original_image = np_trafo_img.copy()

                valid_border_choices = [[5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [6, 5], [6, 6], [6, 7], [6, 8], [6, 9],
                                        [7, 5], [7, 6], [7, 7], [7, 8], [7, 9], [8, 5], [8, 6], [8, 7], [8, 8], [8, 9],
                                        [9, 5], [9, 6], [9, 7], [9, 8], [9, 9], ]
                for c in range(0, n):
                    border = random.choice(valid_border_choices)
                    input_array, known_array, target_array = ex4(np_trafo_img, border, border)
                    # then save these arrays
                    validation_arrays.append([input_array, original_image])
        pbar.update(1)


print("finished")

#saving the arrays
train_arrays = np.array(train_arrays)
test_arrays = np.array(test_arrays)
validation_arrays = np.array(validation_arrays)
np.savez(r"D:\studium_tests\small_original_dataset2_controll.npz", train=train_arrays, test=test_arrays,
         validation=validation_arrays)
print("everything completed")