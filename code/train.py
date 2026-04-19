from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from nac.data import build_datasets
from nac.losses import LossWeights, active_contour_loss
from nac.metrics import (
    accumulate_threshold_sweep,
    combine_threshold_sweeps,
    make_threshold_sweep,
    metrics_from_threshold_sweep,
    segmentation_metrics,
)
from nac.model import TinyNACNet
from nac.utils import default_device, repo_root, save_checkpoint, save_json, set_seed


def parse_threshold(value: str) -> float | None:
    if value.lower() == "auto":
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Train unsupervised neural active contours.")
    parser.add_argument("--data-root", type=Path, default=root / "dataset" / "Kvasir-SEG")
    parser.add_argument("--output-dir", type=Path, default=root / "checkpoints")
    parser.add_argument("--external-mode", choices=["image", "features"], default="image")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threshold", default="auto")
    parser.add_argument("--calibration-steps", type=int, default=19)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def maybe_limit(dataset, limit: int):
    if limit and limit > 0:
        return Subset(dataset, range(min(limit, len(dataset))))
    return dataset


def train_one_epoch(
    model: TinyNACNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    external_mode: str,
    weights: LossWeights,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    batches = 0

    for batch in tqdm(loader, desc="train", leave=False):
        image = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(image)
        loss, parts = active_contour_loss(
            output["mask_logits"],
            image,
            features=output["features"],
            external_mode=external_mode,
            weights=weights,
        )
        loss.backward()
        optimizer.step()

        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        batches += 1

    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def validate(
    model: TinyNACNet,
    loader: DataLoader,
    device: torch.device,
    threshold: float | None,
    calibration_steps: int,
) -> dict[str, float]:
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    total_pred_area = 0.0
    total_target_area = 0.0
    total_soft_area = 0.0
    batches = 0
    sweep_items = []
    threshold_grid = make_threshold_sweep(device, steps=calibration_steps) if threshold is None else None

    for batch in tqdm(loader, desc="val", leave=False):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        probability = torch.sigmoid(model(image)["mask_logits"])
        if threshold is None:
            total_soft_area += float(probability.mean())
            sweep_items.append(accumulate_threshold_sweep(probability, mask, threshold_grid))
        else:
            metrics = segmentation_metrics(probability, mask, threshold=threshold)
            total_dice += float(metrics["dice"])
            total_iou += float(metrics["iou"])
            total_pred_area += float(metrics["pred_area"])
            total_target_area += float(metrics["target_area"])
            total_soft_area += float(metrics["soft_area"])
        batches += 1

    if threshold is None:
        metrics = metrics_from_threshold_sweep(combine_threshold_sweeps(sweep_items))
        return {
            "dice": float(metrics["dice"]),
            "iou": float(metrics["iou"]),
            "pred_area": float(metrics["pred_area"]),
            "target_area": float(metrics["target_area"]),
            "soft_area": total_soft_area / max(1, batches),
            "threshold": float(metrics["threshold"]),
        }

    return {
        "dice": total_dice / max(1, batches),
        "iou": total_iou / max(1, batches),
        "pred_area": total_pred_area / max(1, batches),
        "target_area": total_target_area / max(1, batches),
        "soft_area": total_soft_area / max(1, batches),
        "threshold": float(threshold),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else default_device()
    threshold = parse_threshold(args.threshold)

    train_set, val_set, _ = build_datasets(
        args.data_root,
        image_size=args.image_size,
        seed=args.seed,
    )
    train_set = maybe_limit(train_set, args.limit_train)
    val_set = maybe_limit(val_set, args.limit_val)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TinyNACNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    weights = LossWeights()
    config = vars(args).copy()
    config["data_root"] = str(config["data_root"])
    config["output_dir"] = str(config["output_dir"])
    config["device"] = str(device)
    config["threshold"] = args.threshold
    save_json(config, args.output_dir / "config.json")

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.external_mode,
            weights,
        )
        val_metrics = validate(
            model,
            val_loader,
            device,
            threshold=threshold,
            calibration_steps=args.calibration_steps,
        )
        metrics = {f"train_{k}": v for k, v in train_metrics.items()}
        metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        config["decision_threshold"] = val_metrics["threshold"]
        save_json(config, args.output_dir / "config.json")

        print(
            f"epoch {epoch:03d} "
            f"loss={train_metrics.get('total', 0.0):.4f} "
            f"edge={train_metrics.get('edge', 0.0):.4f} "
            f"len={train_metrics.get('length', 0.0):.4f} "
            f"curv={train_metrics.get('curvature', 0.0):.4f} "
            f"region={train_metrics.get('region', 0.0):.4f} "
            f"compact={train_metrics.get('compactness', 0.0):.4f} "
            f"area={train_metrics.get('area', 0.0):.4f} "
            f"binary={train_metrics.get('binary', 0.0):.4f} "
            f"prior={train_metrics.get('energy_prior', 0.0):.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_thr={val_metrics['threshold']:.2f} "
            f"pred_area={val_metrics['pred_area']:.3f} "
            f"soft_area={val_metrics['soft_area']:.3f} "
            f"mask_area={val_metrics['target_area']:.3f}"
        )

        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, config, metrics)
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, config, metrics)

    print(f"Best validation Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
