from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _tensor_to_image(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).astype(np.uint8))


def _mask_to_image(mask: torch.Tensor) -> Image.Image:
    array = mask.detach().cpu().squeeze().clamp(0, 1).numpy()
    return Image.fromarray((array * 255).astype(np.uint8)).convert("RGB")


def _overlay(image: Image.Image, probability: torch.Tensor) -> Image.Image:
    base = image.convert("RGBA")
    mask = probability.detach().cpu().squeeze().clamp(0, 1).numpy()
    red = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    red[..., 0] = 255
    red[..., 3] = (mask * 150).astype(np.uint8)
    overlay = Image.fromarray(red, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def save_prediction_panel(
    image: torch.Tensor,
    probability: torch.Tensor,
    mask: torch.Tensor | None,
    output_path: Path,
) -> None:
    image_pil = _tensor_to_image(image)
    pred_pil = _mask_to_image(probability)
    overlay_pil = _overlay(image_pil, probability)

    panels = [image_pil, pred_pil, overlay_pil]
    if mask is not None:
        panels.insert(1, _mask_to_image(mask))

    width, height = image_pil.size
    canvas = Image.new("RGB", (width * len(panels), height), color=(255, 255, 255))
    for index, panel in enumerate(panels):
        canvas.paste(panel.resize((width, height)), (index * width, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)

