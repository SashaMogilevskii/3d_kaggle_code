import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import one_hot
from torch import Tensor
from typing import Union


# Basic CrossEntropy
class CategoricalCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CategoricalCrossEntropyLoss, self).__init__()

    def forward(self, predicted, target):
        predicted_probs = F.softmax(predicted, dim=1)
        loss = F.nll_loss(torch.log(predicted_probs + 1e-5), target)
        return loss


# Basic DiceLoss
class MyDiceLoss(nn.Module):

    def __init__(self):
        super(MyDiceLoss, self).__init__()

    def forward(self, predicted, target):
        smooth = 1e-5

        predicted = F.softmax(predicted, dim=1)
        target_one_hot = F.one_hot(target, num_classes=predicted.shape[1]).permute(0, 4, 1, 2, 3).float()

        intersection = torch.sum(predicted * target_one_hot, dim=(2, 3, 4))
        union = torch.sum(predicted, dim=(2, 3, 4)) + torch.sum(target_one_hot, dim=(2, 3, 4))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        loss = 1.0 - dice.mean()

        return loss


# https://github.com/mathiaszinnen/focal_loss_torch/blob/main/focal_loss/focal_loss.py

class FocalLoss(nn.Module):
    """Computes the focal loss between input and target
    as described here https://arxiv.org/abs/1708.02002v2

    Args:
        gamma (float):  The focal loss focusing parameter.
        weights (Union[None, Tensor]): Rescaling weight given to each class.
        If given, has to be a Tensor of size C. optional.
        reduction (str): Specifies the reduction to apply to the output.
        it should be one of the following 'none', 'mean', or 'sum'.
        default 'mean'.
        ignore_index (int): Specifies a target value that is ignored and
        does not contribute to the input gradient. optional.
        eps (float): smoothing to prevent log from returning inf.
    """

    def __init__(
            self,
            gamma,
            weights: Union[None, Tensor] = None,
            reduction: str = 'mean',
            ignore_index=-100,
            eps=1e-16
    ) -> None:
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError(
                'Reduction {} not implemented.'.format(reduction)
            )
        assert weights is None or isinstance(weights, Tensor), \
            'weights should be of type Tensor or None, but {} given'.format(
                type(weights))
        self.reduction = reduction
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = eps
        self.weights = weights

    def _get_weights(self, target: Tensor) -> Tensor:
        if self.weights is None:
            return torch.ones(target.shape[0])
        weights = target * self.weights
        return weights.sum(dim=-1)

    def _process_target(
            self, target: Tensor, num_classes: int, mask: Tensor
    ) -> Tensor:

        # convert all ignore_index elements to zero to avoid error in one_hot
        # note - the choice of value 0 is arbitrary, but it should not matter as these elements will be ignored in the loss calculation
        target = target * (target != self.ignore_index)
        target = target.view(-1)
        return one_hot(target, num_classes=num_classes)

    def _process_preds(self, x: Tensor) -> Tensor:
        if x.dim() == 1:
            x = torch.vstack([1 - x, x])
            x = x.permute(1, 0)
            return x
        return x.view(-1, x.shape[-1])

    def _calc_pt(
            self, target: Tensor, x: Tensor, mask: Tensor
    ) -> Tensor:
        p = target * x
        p = p.sum(dim=-1)
        p = p * ~mask
        return p

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        assert torch.all((x >= 0.0) & (x <= 1.0)), ValueError(
            'The predictions values should be between 0 and 1, \
                make sure to pass the values to sigmoid for binary \
                classification or softmax for multi-class classification'
        )
        mask = target == self.ignore_index
        mask = mask.view(-1)
        x = self._process_preds(x)
        num_classes = x.shape[-1]
        target = self._process_target(target, num_classes, mask)
        weights = self._get_weights(target).to(x.device)
        pt = self._calc_pt(target, x, mask)
        focal = 1 - pt
        nll = -torch.log(self.eps + pt)
        nll = nll.masked_fill(mask, 0)
        loss = weights * (focal ** self.gamma) * nll
        return self._reduce(loss, mask, weights)

    def _reduce(self, x: Tensor, mask: Tensor, weights: Tensor) -> Tensor:
        if self.reduction == 'mean':
            return x.sum() / (~mask * weights).sum()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x


if __name__ == '__main__':
    import torch
    import random
    import numpy as np

    # Set the seed for CPU
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set the seed for GPU if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Additional steps to ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Now any random operation using torch, random, or numpy should be reproducible

    y_pred = torch.randint(-10, 10, (7, 3), dtype=torch.float32)
    y_pred = F.softmax(y_pred)
    # y_true = torch.tensor([[0, 0, 1],
    #                        [0, 1, 0],
    #                        [1, 0, 0],
    #                        [0, 1, 0],
    #                        [0, 0, 1]])

    y_true = torch.tensor([[1],
                           [2],
                           [0],
                           [0],
                           [0],
                           [2],
                           [0]])

    loss = FocalLoss(gamma=3 , weights=torch.Tensor([0.1, 0.45, 0.45]))

    print(loss(y_pred, y_true))