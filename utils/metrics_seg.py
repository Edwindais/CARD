import torch
import torch.nn as nn


class DiceMetric(nn.Module):
    def __init__(self, n_classes):
        super(DiceMetric, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        # input tensor shape: (B,1,H,W,D)
        # returns one-hot: (B,C,H,W,D)
        tensor_list = [(input_tensor == i) for i in range(self.n_classes)]
        return torch.cat(tensor_list, dim=1).float()

    def forward(self, inputs, target, softmax=False):
        """
        Args:
            inputs: Tensor, shape (B,1,H,W,D) or (B,C,H,W,D)
            target: Tensor, same shape as inputs
            softmax: whether to apply softmax on inputs
        Returns:
            dice_scores: Tensor of shape (B, C)
        """
        # if inputs are logits over classes, softmax then choose class dims
        if softmax and inputs.dim() == 5:
            inputs = torch.softmax(inputs, dim=1)
        # ensure one-hot encoding
        if (inputs.dim() == 4 or inputs.dim() == 5) and inputs.size(1) == 1:
            inputs = self._one_hot_encoder(inputs)
        if (target.dim() == 4 or target.dim() == 5) and target.size(1) == 1:
            target = self._one_hot_encoder(target)
        assert inputs.size() == target.size(), "Input and target must have same shape"

        # flatten spatial dims
        dims = tuple(range(2, inputs.dim()))  # exclude batch and class dims
        # compute intersection and sums
        intersection = torch.sum(inputs * target, dim=dims)
        sums = torch.sum(inputs, dim=dims) + torch.sum(target, dim=dims)
        # avoid zero division
        eps = 1e-6
        dice = (2 * intersection + eps) / (sums + eps)
        return dice  # shape: (B, C)
