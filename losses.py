import torch
from torch import nn as nn


def compute_per_channel_dice(input, target, smooth=1.):
    # assumes that input is a normalized probability

    # input and target shapes must match
    if target.dim() == 4:
        target = target.unsqueeze(1)

    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)

    target = target.float()
    # Compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)

    denominator = (input + target).sum(-1)
    return (2. * intersect + smooth) / (denominator + smooth)


class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    """

    def __init__(self, smooth=1., sigmoid_normalization=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target, expand=True):
        # get probabilities from logits
        input = self.normalization(input)

        per_channel_dice = compute_per_channel_dice(input, target, smooth=self.smooth)
        # Average the Dice experiments_log across all channels/classes
        return torch.mean(1. - per_channel_dice)


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target, expand=True):
        return self.loss(input, target)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)

    return transposed.contiguous().view(C, -1)


def get_loss_criterion(name, weight=None):
    """
    Returns the loss function based on the loss_str.
    :param name: specifies the loss function to be used
    :param weight: a manual rescaling weight given to each class
    :param ignore_index: specifies a target value that is ignored and does not contribute to the input gradient
    :return: an instance of the loss function
    """

    if name == 'bce':
        return BinaryCrossEntropyLoss()  # include sigmoid
    elif name == 'ce':
        return nn.CrossEntropyLoss(weight=weight)
    elif name == 'dice':
        return DiceLoss()
