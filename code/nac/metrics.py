from __future__ import annotations

import torch


def threshold_mask(probability: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (probability >= threshold).float()


def dice_score(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prediction = prediction.flatten(start_dim=1)
    target = target.flatten(start_dim=1)
    intersection = (prediction * target).sum(dim=1)
    denominator = prediction.sum(dim=1) + target.sum(dim=1)
    return ((2 * intersection + eps) / (denominator + eps)).mean()


def iou_score(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prediction = prediction.flatten(start_dim=1)
    target = target.flatten(start_dim=1)
    intersection = (prediction * target).sum(dim=1)
    union = prediction.sum(dim=1) + target.sum(dim=1) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def segmentation_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    prediction = threshold_mask(probability, threshold=threshold)
    return {
        "dice": dice_score(prediction, target),
        "iou": iou_score(prediction, target),
    }

