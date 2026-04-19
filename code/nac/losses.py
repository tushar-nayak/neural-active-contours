from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    edge: float = 4.0
    length: float = 0.10
    curvature: float = 0.03
    region: float = 1.0
    compactness: float = 0.05
    area: float = 10.0
    binary: float = 0.20
    energy_prior: float = 0.05
    area_target: float = 0.10
    area_min: float = 0.015
    area_max: float = 0.35


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    red = image[:, 0:1]
    green = image[:, 1:2]
    blue = image[:, 2:3]
    return 0.299 * red + 0.587 * green + 0.114 * blue


def finite_difference_energy(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dx = F.pad(x[..., :, 1:] - x[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(x[..., 1:, :] - x[..., :-1, :], (0, 0, 0, 1))
    energy = torch.sqrt(dx.pow(2) + dy.pow(2) + eps)
    return (energy - eps**0.5).clamp_min(0.0)


def normalize_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = x.flatten(start_dim=1)
    min_value = flat.min(dim=1).values.view(-1, 1, 1, 1)
    max_value = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (x - min_value) / (max_value - min_value + eps)


def image_gradient_energy(image: torch.Tensor) -> torch.Tensor:
    gray = rgb_to_gray(image)
    return normalize_per_sample(finite_difference_energy(gray))


def feature_gradient_energy(features: torch.Tensor) -> torch.Tensor:
    energy = finite_difference_energy(features).mean(dim=1, keepdim=True)
    return normalize_per_sample(energy)


def boundary_alignment_loss(boundary: torch.Tensor, external_energy: torch.Tensor) -> torch.Tensor:
    boundary = normalize_per_sample(boundary)
    return F.mse_loss(boundary, external_energy)


def compactness_loss(probability: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    perimeter = finite_difference_energy(probability).mean(dim=(1, 2, 3))
    area = probability.mean(dim=(1, 2, 3)).clamp_min(eps)
    return (perimeter.pow(2) / area).mean()


def curvature_loss(probability: torch.Tensor) -> torch.Tensor:
    ddx = probability[..., :, 2:] - 2 * probability[..., :, 1:-1] + probability[..., :, :-2]
    ddy = probability[..., 2:, :] - 2 * probability[..., 1:-1, :] + probability[..., :-2, :]
    return ddx.abs().mean() + ddy.abs().mean()


def chan_vese_region_loss(image: torch.Tensor, probability: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    gray = rgb_to_gray(image)
    inside_mass = probability.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    outside = 1.0 - probability
    outside_mass = outside.sum(dim=(2, 3), keepdim=True).clamp_min(eps)

    inside_mean = (gray * probability).sum(dim=(2, 3), keepdim=True) / inside_mass
    outside_mean = (gray * outside).sum(dim=(2, 3), keepdim=True) / outside_mass
    inside_term = probability * (gray - inside_mean).pow(2)
    outside_term = outside * (gray - outside_mean).pow(2)
    return (inside_term + outside_term).mean()


def area_prior_loss(
    probability: torch.Tensor,
    target_fraction: float = 0.10,
    min_fraction: float = 0.015,
    max_fraction: float = 0.35,
) -> torch.Tensor:
    # Heuristic Kvasir size prior: it nudges the model toward plausible polyp area
    # without pretending we know the exact foreground fraction.
    area = probability.mean(dim=(1, 2, 3))
    target_penalty = (area - target_fraction).pow(2)
    small_penalty = F.relu(min_fraction - area).pow(2)
    large_penalty = F.relu(area - max_fraction).pow(2)
    return (target_penalty + small_penalty + large_penalty).mean()


def active_contour_loss(
    logits: torch.Tensor,
    image: torch.Tensor,
    features: torch.Tensor | None = None,
    external_mode: str = "image",
    weights: LossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or LossWeights()
    probability = torch.sigmoid(logits)
    boundary = finite_difference_energy(probability)

    image_energy = image_gradient_energy(image)
    if external_mode == "image":
        external_energy = image_energy
        energy_prior = logits.new_tensor(0.0)
    elif external_mode == "features":
        if features is None:
            raise ValueError("external_mode='features' requires model features.")
        external_energy = feature_gradient_energy(features)
        energy_prior = F.mse_loss(external_energy, image_energy.detach())
    else:
        raise ValueError(f"Unknown external mode: {external_mode}")

    edge = boundary_alignment_loss(boundary, external_energy)
    length = boundary.mean()
    curve = curvature_loss(probability)
    region = chan_vese_region_loss(image, probability)
    compactness = compactness_loss(probability)
    area = area_prior_loss(
        probability,
        target_fraction=weights.area_target,
        min_fraction=weights.area_min,
        max_fraction=weights.area_max,
    )
    binary = (probability * (1.0 - probability)).mean()

    total = (
        weights.edge * edge
        + weights.length * length
        + weights.curvature * curve
        + weights.region * region
        + weights.compactness * compactness
        + weights.area * area
        + weights.binary * binary
        + weights.energy_prior * energy_prior
    )

    parts = {
        "total": total.detach(),
        "edge": edge.detach(),
        "length": length.detach(),
        "curvature": curve.detach(),
        "region": region.detach(),
        "compactness": compactness.detach(),
        "area": area.detach(),
        "binary": binary.detach(),
        "energy_prior": energy_prior.detach(),
    }
    return total, parts
