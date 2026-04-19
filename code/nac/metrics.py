from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class ThresholdSweep:
    thresholds: torch.Tensor
    true_positive: torch.Tensor
    predicted_positive: torch.Tensor
    target_positive: torch.Tensor
    pixel_count: torch.Tensor


def make_threshold_sweep(
    device: torch.device,
    steps: int = 19,
    low: float = 0.05,
    high: float = 0.95,
) -> torch.Tensor:
    return torch.linspace(low, high, steps=steps, device=device)


def accumulate_threshold_sweep(
    probability: torch.Tensor,
    target: torch.Tensor,
    thresholds: torch.Tensor,
) -> ThresholdSweep:
    threshold_grid = thresholds.view(1, -1, 1, 1, 1).to(device=probability.device, dtype=probability.dtype)
    prediction = (probability.unsqueeze(1) >= threshold_grid).to(dtype=probability.dtype)
    target_grid = target.unsqueeze(1)

    true_positive = (prediction * target_grid).sum(dim=(0, 2, 3, 4))
    predicted_positive = prediction.sum(dim=(0, 2, 3, 4))
    target_positive = target.sum().expand_as(true_positive)
    pixel_count = probability.new_tensor(
        probability.shape[0] * probability.shape[2] * probability.shape[3]
    ).expand_as(true_positive)

    return ThresholdSweep(
        thresholds=thresholds,
        true_positive=true_positive,
        predicted_positive=predicted_positive,
        target_positive=target_positive,
        pixel_count=pixel_count,
    )


def combine_threshold_sweeps(items: list[ThresholdSweep]) -> ThresholdSweep:
    if not items:
        raise ValueError("Cannot combine an empty threshold sweep list.")

    thresholds = items[0].thresholds
    true_positive = torch.stack([item.true_positive for item in items]).sum(dim=0)
    predicted_positive = torch.stack([item.predicted_positive for item in items]).sum(dim=0)
    target_positive = torch.stack([item.target_positive for item in items]).sum(dim=0)
    pixel_count = torch.stack([item.pixel_count for item in items]).sum(dim=0)
    return ThresholdSweep(
        thresholds=thresholds,
        true_positive=true_positive,
        predicted_positive=predicted_positive,
        target_positive=target_positive,
        pixel_count=pixel_count,
    )


def metrics_from_threshold_sweep(sweep: ThresholdSweep, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    true_positive = sweep.true_positive
    predicted_positive = sweep.predicted_positive
    target_positive = sweep.target_positive
    pixel_count = sweep.pixel_count.clamp_min(eps)

    dice = (2 * true_positive + eps) / (predicted_positive + target_positive + eps)
    iou = (true_positive + eps) / (predicted_positive + target_positive - true_positive + eps)
    pred_area = predicted_positive / pixel_count
    target_area = target_positive / pixel_count

    best_index = torch.argmax(dice)
    return {
        "dice": dice[best_index],
        "iou": iou[best_index],
        "pred_area": pred_area[best_index],
        "target_area": target_area[best_index],
        "threshold": sweep.thresholds[best_index],
    }


def segmentation_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    prediction = threshold_mask(probability, threshold=threshold)
    return {
        "dice": dice_score(prediction, target),
        "iou": iou_score(prediction, target),
        "pred_area": prediction.mean(),
        "target_area": target.mean(),
        "soft_area": probability.mean(),
    }
