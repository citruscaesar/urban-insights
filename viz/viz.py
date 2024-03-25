import torch
import numpy as np
import pandas as pd
import imageio.v3 as iio
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def calculate_statistics(self, df: pd.DataFrame):
    #Archaic Way
    sums = np.array([0, 0, 0], dtype = np.float64)
    sum_of_squares = np.array([0, 0, 0], dtype = np.float64)
    pixels_per_channel: float = len(df) * 256 * 256

    for file_path in tqdm(df.file_path):
        image = iio.imread(file_path).transpose(2, 0, 1)
        sums += image.sum(axis = (1, 2))
        sum_of_squares += np.power(image, 2).sum(axis = (1, 2))

    means = sums/pixels_per_channel
    std_devs = np.sqrt(np.abs(sum_of_squares / pixels_per_channel - (means ** 2)))
    return means, std_devs

def viz_batch(batch: tuple[torch.Tensor, torch.Tensor], le: LabelEncoder) -> None:
    images, targets = batch
    labels = le.inverse_transform(targets)
    assert images.shape[0] == targets.shape[0], "#images != #targets"

    subplot_dims:tuple[int, int]
    if images.shape[0] <= 8:
        subplot_dims = (1, images.shape[0])
    else:
        subplot_dims = (int(np.ceil(images.shape[0]/8)), 8)

    figsize = 20
    figsize_factor = subplot_dims[0] / subplot_dims[1]
    _, axes = plt.subplots(nrows = subplot_dims[0], 
                           ncols = subplot_dims[1], 
                           figsize = (figsize, figsize * figsize_factor))
    for idx, ax in enumerate(axes.ravel()):
        ax.imshow(images[idx].permute(1, 2, 0))
        ax.tick_params(axis = "both", which = "both", 
                       bottom = False, top = False, 
                       left = False, right = False,
                       labeltop = False, labelbottom = False, 
                       labelleft = False, labelright = False)
        ax.set_xlabel(f"{labels[idx]}({targets[idx]})")