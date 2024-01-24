import sys
sys.path.append('../../pytorch3dunet/unet3d/')
from sklearn.metrics import roc_auc_score
from metrics import MeanIoU

from loguru import logger

import torch
import torch.nn.functional as F


def print_metric_MeanIoU(y_pred, y_true):
    metric = MeanIoU(skip_channels=())
    y_pred = torch.Tensor(y_pred)
    y_true = F.one_hot(torch.Tensor(y_true).long(), num_classes=5).permute(0, 4, 1, 2, 3).float()
    logger.info(f"MeanIoU : {metric(y_pred, y_true)}")

    metric = MeanIoU(skip_channels=[0])
    logger.info(f"MeanIoU without 0 channel: {metric(y_pred, y_true):.3f}")
    metric = MeanIoU(skip_channels=((0, 5)))
    logger.info(f"MeanIoU without 0, 5 channel: {metric(y_pred, y_true):.3f}")


def print_classification_metrics(y_pred, y_true, label):
    """

    Args:
        y_pred:
        y_true:
        label:

    Returns:
        predicted_probs = torch.tensor([[0.0, 0.7, 0.3],
                                    [0.5, 0.4, 0.1],
                                    [0.0, 0.5, 0.4],
                                    [0.1, 0.8, 0.1]])

        true_labels = torch.tensor([[0, 1, 0],
                                    [1, 0, 0],
                                    [0, 1, 0],
                                    [0, 1,  0]])

    print_classification_metrics(predicted_probs, true_labels)
    Examples:

    """
    num_classes = y_pred.shape[1]
    mean_errors_per_class = []
    auc_scores_per_class = []
    log_loss_per_class = []
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)

    simple_cross_entropy = F.cross_entropy(y_pred, y_true.float())
    for class_idx in range(num_classes):
        class_errors = torch.abs(y_pred[:, class_idx] - y_true[:, class_idx])
        mean_error = class_errors.mean()
        mean_errors_per_class.append(round(mean_error.item(),3))

        auc = roc_auc_score(y_true[:, class_idx], y_pred[:, class_idx])
        auc_scores_per_class.append(round(auc, 3))

        log_loss = F.binary_cross_entropy(y_pred[:, class_idx], y_true[:, class_idx].float())
        log_loss_per_class.append(round(log_loss.item(), 3))

    logger.info(f"Mean  per Class {label}:{ mean_errors_per_class}")
    logger.info(f"MEAN error: {round(sum(mean_errors_per_class) / len(mean_errors_per_class), 3)}")

    logger.info(f"AUC per Class {label}: {auc_scores_per_class}")
    logger.info(f"MEAN AUC per Class {label}: {round(sum(auc_scores_per_class) / len(auc_scores_per_class), 3)}")

    logger.info(f"Log Loss per Class {label}: {log_loss_per_class}")
    logger.info(f"MEAN Log Loss  per Class {label}: {round(sum(log_loss_per_class) / len(log_loss_per_class), 3)}")
    logger.info(f"Cross entropy: {label} {round(simple_cross_entropy.item() , 3)}")

if __name__ == '__main__':

    predicted_probs = torch.tensor([[0.0, 0.7, 0.3],
                                    [0.8, 0.2, 0],
                                    [0.0, 0.5, 0.4],
                                    [0.0, 0.8, 0.1]])


    true_labels = torch.tensor([[0, 1, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 1,  1]])

    log_loss = F.cross_entropy(predicted_probs, true_labels.float())
    print(log_loss)

    print_classification_metrics(predicted_probs, true_labels, label='kidney')