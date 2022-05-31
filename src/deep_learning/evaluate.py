import numpy as np
import torch


def get_deviation(outputs, labels):
    """
    Calculates the sum of Euklidean distances of the predictions to the **labels** (if both exist).
    The prediction is calculated by taking the mean of all segmentated pixels.
    If there is no segmented pixel or no labeled pixel, the sample is skipped.

    :param outputs (torch.tensor) of shape (B x 1 x W x H) (after sigmoid/softmax)
    :param labels (torch.tensor) of the same shape
    :return: dist_euklid (float) sum of Euklidean distances between predictions and labels
    """
    if not outputs.size() == labels.size():
        raise AssertionError("Output and Labels need to have the same torch.Size!")

    batch_size = outputs.shape[0]
    is_crack_tip = torch.where(outputs >= 0.5, 1, 0)

    dist_euklid = []
    for i in range(batch_size):
        prediction_i = torch.nonzero(is_crack_tip[i], as_tuple=False)[:, -2:] / 1.
        label_i = torch.nonzero((labels[i] == 1), as_tuple=False)[:, -2:] / 1.
        # skip the unlabeled or unsegmented
        if len(label_i) == 0 or len(prediction_i) == 0:
            continue
        prediction_i = torch.mean(prediction_i, dim=0)
        label_i = torch.mean(label_i, dim=0)
        dist = torch.sqrt(torch.sum((prediction_i - label_i) ** 2)).item()
        dist_euklid.append(dist)

    return np.asarray(dist_euklid)


def get_reliability(outputs, labels):
    """
    Calculates the reliability of a segmentation model's output batch and the corresponding labels.

    :param outputs: (torch.tensor) output of the model (after Sigmoid/Softmax) (B x H x W)
    :param labels: (torch.tensor) corresponding labels (B x H x W) with 1's and 0's
    :return: score (int) reliability score (1.0 = 100 %)
    """
    if not outputs.size() == labels.size():
        raise AssertionError("Output and Labels need to have the same torch.Size!")

    batch_size = outputs.shape[0]
    is_crack_tip = torch.where(outputs >= 0.5, 1, 0)

    unpredicted = 0
    for i in range(batch_size):
        prediction_i = torch.nonzero(is_crack_tip[i], as_tuple=False)[:, -2:] / 1.
        label_i = torch.nonzero((labels[i] == 1), as_tuple=False)[:, -2:] / 1.
        # skip the unlabeled or unsegmented
        if len(label_i) > 0 and len(prediction_i) == 0:
            unpredicted += 1

    score = 1. - unpredicted / batch_size

    return score
