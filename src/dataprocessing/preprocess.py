import torch
import numpy as np


def normalize(input_t):
    """
    Normalize every input channel to mean 0 and variance 1 for each input

    :param input_t: (torch.tensor or np.array) of shape (.., W, H)
    :return: input_normalized (torch.tensor)
    """
    input_as_array = np.asarray(input_t)

    means = np.nanmean(input_as_array, axis=(-2, -1), keepdims=True)
    stds = np.nanstd(input_as_array, axis=(-2, -1), keepdims=True)

    input_normalized = (input_as_array - means) / stds
    input_normalized = torch.from_numpy(input_normalized)

    return input_normalized


def target_to_crack_tip_position(target):
    """
    Extracts crack tip positions from segmentation ground truth targets

    :param target: (torch.tensor) of shape (W, H)
    :return: crack_tip_positions (torch.tensor) of size 2
    """
    pixel_position = torch.nonzero((target == 2), as_tuple=False)
    crack_tip_position = pixel_position / 1.
    return crack_tip_position
