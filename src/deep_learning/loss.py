from torch import nn
from torch.nn.functional import mse_loss


class DiceLoss(nn.Module):
    """Dice loss for unbalanced binary segmentation problems"""
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target):
        """
        Applies the criterion to a batch of predictions and corresponding targets.

        :param prediction: (torch.tensor) of size (B x C x H x W)
        :param target: (torch.tensor) of size (B x C x H x W)
        :return: loss (torch.tensor) 1 - dice coefficient
        """
        if not prediction.size() == target.size() and len(prediction.size()) == 4:
            raise AssertionError("'prediction' and 'target' need to have length 4 and the same size")
        prediction = prediction[:, 0].contiguous().view(-1)
        target = target[:, 0].contiguous().view(-1)
        intersection = (prediction * target).sum()
        dsc = (2. * intersection + self.eps) / (prediction.sum() + target.sum() + self.eps)
        return 1. - dsc


class MSELoss(nn.Module):
    """Mean Squared Error loss with (global) weight factor"""
    def __init__(self, weight_factor=1, reduction='mean'):
        """
        :param weight_factor: (float) factor by which the loss is multiplied
        :param reduction: (str) 'mean' or 'sum'
        """
        super().__init__()
        self.weight_factor = weight_factor
        self.reduction = reduction

    def forward(self, prediction, target):
        """
        Applies the criterion to a batch of predictions and corresponding targets.

        :param prediction: (torch.tensor)
        :param target: (torch.tensor)
        :return: loss (torch.tensor)
        """
        return self.weight_factor * mse_loss(prediction, target, reduction=self.reduction)
