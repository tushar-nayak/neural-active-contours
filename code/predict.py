from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from nac.data import build_datasets
from nac.model import TinyNACNet
from nac.utils import default_device, load_checkpoint, repo_root
from nac.visualize import save_prediction_panel


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(description="Export neural active contour predictions.")
    parser.add_argument("--checkpoint", type=Path, default=root / "checkpoints" / "best.pt")
    parser.add_argument("--data-root", type=Path, default=root / "dataset" / "Kvasir-SEG")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=root / "predictions")
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else default_device()
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint.get("config", {})

    model = TinyNACNet(base_channels=int(config.get("base_channels", 24))).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_size = int(config.get("image_size", 256))
    seed = int(config.get("seed", 7))
    datasets = dict(zip(["train", "val", "test"], build_datasets(args.data_root, image_size=image_size, seed=seed)))
    dataset = datasets[args.split]
    if args.max_images > 0:
        dataset = Subset(dataset, range(min(args.max_images, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    saved = 0
    for batch in tqdm(loader, desc=f"predict:{args.split}"):
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        probability = torch.sigmoid(model(image)["mask_logits"])
        for item in range(image.shape[0]):
            name = batch["name"][item]
            save_prediction_panel(
                image[item],
                probability[item],
                mask[item],
                args.output_dir / args.split / f"{name}.png",
            )
            saved += 1

    print(f"saved={saved} to {args.output_dir / args.split}")


if __name__ == "__main__":
    main()

