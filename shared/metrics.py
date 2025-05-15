"""
shared/metrics.py

Contains utility functions for evaluating segmentation performance.
"""

import torch


def compute_iou(preds, targets, threshold=0.5, eps=1e-6):
    """
    Computes Intersection over Union (IoU) between predicted and target masks.

    Args:
        preds (Tensor): Predicted masks of shape (B, 1, H, W)
        targets (Tensor): Ground truth masks of shape (B, 1, H, W)
        threshold (float): Threshold for binarizing predictions
        eps (float): Small value to avoid division by zero

    Returns:
        float: Mean IoU over the batch
    """
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = ((preds + targets) >= 1).float().sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def compute_dice(preds, targets, threshold=0.5, eps=1e-6):
    """
    Computes Dice coefficient between predicted and target masks.

    Args:
        preds (Tensor): Predicted masks of shape (B, 1, H, W)
        targets (Tensor): Ground truth masks of shape (B, 1, H, W)
        threshold (float): Threshold for binarizing predictions
        eps (float): Small value to avoid division by zero

    Returns:
        float: Mean Dice coefficient over the batch
    """
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    sum_preds_targets = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (sum_preds_targets + eps)
    return dice.mean().item()
