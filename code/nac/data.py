from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class DataSplit:
    train: list[int]
    val: list[int]
    test: list[int]


def find_kvasir_pairs(root: Path) -> list[tuple[Path, Path]]:
    image_dir = root / "images"
    mask_dir = root / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Expected Kvasir-SEG folders at {image_dir} and {mask_dir}."
        )

    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")
        pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(f"No image/mask pairs found in {root}")
    return pairs


def make_split(
    item_count: int,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 7,
) -> DataSplit:
    if item_count < 3:
        raise ValueError("At least three samples are required for train/val/test splits.")
    if val_fraction <= 0 or test_fraction <= 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Split fractions must leave a non-empty training set.")

    indices = list(range(item_count))
    random.Random(seed).shuffle(indices)
    test_count = max(1, round(item_count * test_fraction))
    val_count = max(1, round(item_count * val_fraction))
    train_count = item_count - val_count - test_count
    if train_count <= 0:
        raise ValueError("Split fractions produced an empty training set.")

    train = indices[:train_count]
    val = indices[train_count : train_count + val_count]
    test = indices[train_count + val_count :]
    return DataSplit(train=train, val=val, test=test)


def _resize_image(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), resample=Image.Resampling.BILINEAR)


def _resize_mask(mask: Image.Image, size: int) -> Image.Image:
    return mask.resize((size, size), resample=Image.Resampling.NEAREST)


def _image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    array = np.asarray(mask.convert("L"), dtype=np.float32)
    array = (array > 127).astype(np.float32)
    return torch.from_numpy(array).unsqueeze(0)


class KvasirSegDataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        indices: Sequence[int] | None = None,
        image_size: int = 256,
    ) -> None:
        self.root = Path(root)
        self.pairs = find_kvasir_pairs(self.root)
        self.indices = list(indices) if indices is not None else list(range(len(self.pairs)))
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor | str]:
        pair_index = self.indices[item]
        image_path, mask_path = self.pairs[pair_index]
        image = _resize_image(Image.open(image_path), self.image_size)
        mask = _resize_mask(Image.open(mask_path), self.image_size)

        return {
            "image": _image_to_tensor(image),
            "mask": _mask_to_tensor(mask),
            "name": image_path.stem,
        }


def build_datasets(
    root: Path | str,
    image_size: int = 256,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 7,
) -> tuple[KvasirSegDataset, KvasirSegDataset, KvasirSegDataset]:
    pairs = find_kvasir_pairs(Path(root))
    split = make_split(
        len(pairs),
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    return (
        KvasirSegDataset(root, split.train, image_size=image_size),
        KvasirSegDataset(root, split.val, image_size=image_size),
        KvasirSegDataset(root, split.test, image_size=image_size),
    )

