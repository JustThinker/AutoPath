import numpy as np
import torch
import torch.nn as nn

from losses import compute_per_channel_dice


class Medical3DSegMetrics:
    """
    Computes some metrics for 3D medical image segmentation.
    Metrics includes [dice, Relative absolute volume difference (RAVD), Average symmetric surface distance (ASSD), Maximum symmetric surface distance (MSSD)]
    """

    def __init__(self, smooth=1., connectivity=1, sigmoid_normalization=True):
        self.smooth = smooth
        self.connectivity = connectivity
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def __call__(self, input, target):
        # Average across channels in order to get the final experiments_log
        input = self.normalization(input)
        binary_prediction = self._binarize_predictions(input)

        if len(binary_prediction.shape) == 4:
            binary_prediction = binary_prediction.permute((1, 0, 2, 3))
            target = target.permute((1, 0, 2, 3))
        else:
            binary_prediction = torch.squeeze(binary_prediction, dim=1)

        dice_list = []

        for i in range(input.size(0)):
            t = target[i]
            p = binary_prediction[i]

            dice = torch.mean(compute_per_channel_dice(p, t, smooth=self.smooth)).item()
            dice_list.append(dice)

        dice_score = np.mean(dice_list)

        return dice_score

    def _binarize_predictions(self, input):
        """
        Puts 1 for the class/channel with the highest probability and 0 in other channels. Returns byte tensor of the
        same size as the input tensor.
        """
        if input.shape[1] == 1:
            return (input > 0.5).float()
        else:
            _, max_index = torch.max(input, dim=0, keepdim=True)
            return torch.zeros_like(input, dtype=torch.float).scatter_(0, max_index, 1)


def get_evaluation_metric():
    return Medical3DSegMetrics()
