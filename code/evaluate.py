from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nac.data import build_datasets
from nac.metrics import segmentation_metrics
from nac.model import TinyNACNet
from nac.utils import default_device, load_checkpoint, repo_root
from nac.visualize import save_prediction_panel


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Evaluate a neural active contour checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=root / "checkpoints" / "best.pt")
    parser.add_argument("--data-root", type=Path, default=root / "dataset" / "Kvasir-SEG")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--save-samples", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs" / "eval")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def select_split(args: argparse.Namespace):
    config = getattr(args, "checkpoint_config", {})
    image_size = int(config.get("image_size", 256))
    seed = int(config.get("seed", 7))
    train_set, val_set, test_set = build_datasets(args.data_root, image_size=image_size, seed=seed)
    return {"train": train_set, "val": val_set, "test": test_set}[args.split]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else default_device()
    checkpoint = load_checkpoint(args.checkpoint, device)
    args.checkpoint_config = checkpoint.get("config", {})

    model = TinyNACNet(base_channels=int(args.checkpoint_config.get("base_channels", 24))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = select_split(args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    dice_total = 0.0
    iou_total = 0.0
    batches = 0
    saved = 0

    for batch in tqdm(loader, desc=f"evaluate:{args.split}"):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        probability = torch.sigmoid(model(image)["mask_logits"])
        metrics = segmentation_metrics(probability, mask, threshold=args.threshold)
        dice_total += float(metrics["dice"])
        iou_total += float(metrics["iou"])
        batches += 1

        for item in range(image.shape[0]):
            if saved >= args.save_samples:
                break
            name = batch["name"][item]
            save_prediction_panel(
                image[item],
                probability[item],
                mask[item],
                args.output_dir / f"{name}.png",
            )
            saved += 1

    print(f"split={args.split}")
    print(f"dice={dice_total / max(1, batches):.4f}")
    print(f"iou={iou_total / max(1, batches):.4f}")
    if args.save_samples:
        print(f"saved_samples={min(saved, args.save_samples)} to {args.output_dir}")


if __name__ == "__main__":
    main()

