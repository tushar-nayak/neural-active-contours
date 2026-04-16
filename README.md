# Unsupervised Neural Active Contours

A deep learning approach to Lectures 11 and 12.

This project explores unsupervised image segmentation through a neural version of
classical active contours, also known as snakes. Instead of training a standard
supervised segmentation network such as a U-Net with ground-truth masks, the goal
is to build a lightweight CNN that outputs a 2D contour map directly from an
input image.

The network is trained without segmentation labels. Its loss function is a
custom PyTorch implementation of the classical Active Contour energy equation,
where the model directly minimizes internal contour energy while encouraging the
predicted contour to align with image evidence.

## Core Idea

Classical active contour models optimize a curve by balancing two forces:

- Internal energy, which controls contour smoothness, stiffness, and continuity.
- External energy, which attracts the contour toward meaningful image boundaries.

This project replaces manual contour evolution with a lightweight CNN whose
output is optimized using the same energy principles. The CNN learns to produce
contour maps that satisfy active-contour constraints, without requiring
ground-truth segmentation masks during training.

## Initial Method

The first version will use image gradients as the external energy term. This
keeps the method close to the classical snake formulation:

- The CNN predicts a 2D contour map.
- Internal contour terms penalize noisy, fragmented, or overly flexible curves.
- External image-gradient terms reward alignment with strong lesion or cell
  boundaries.
- Training is fully unsupervised and implemented in PyTorch.

## Extension: Learned External Energy

A planned extension is to go beyond raw image gradients for the external energy
term. Instead of relying only on edge magnitude, the active-contour CNN can be
combined with a lightweight learned image-feature extractor.

This feature extractor would produce boundary-aware image features that serve as
a richer external energy field. The contour network could then align with
learned semantic or texture cues, not just local gradient strength. This may help
with noisy images, weak boundaries, artifacts, and cases where the most useful
segmentation signal is not captured by simple image gradients.

## Candidate Datasets

The project is currently set up for Kvasir-SEG, a compact 2D medical polyp
segmentation dataset. The dataset is stored locally under `dataset/Kvasir-SEG/`
and is ignored by Git.

Other candidate 2D datasets for later experiments:

- ISIC, the International Skin Imaging Collaboration lesion dataset.
- BBBC038, the Broad Bioimage Benchmark microscopy dataset.

These datasets contain objects with clear but sometimes noisy boundaries, making
them appropriate for evaluating energy-minimizing contour models without
requiring 3D volumetric rendering.

## Goals

- Implement a lightweight CNN that predicts 2D contour maps.
- Implement an unsupervised PyTorch loss based on active contour energy.
- Compare raw image-gradient external energy with learned feature-based external
  energy.
- Evaluate whether neural active contours can recover useful segmentation
  boundaries without ground-truth masks.

## Project Layout

```text
code/
  train.py              # unsupervised training loop
  evaluate.py           # Dice/IoU evaluation with held-out masks
  predict.py            # export prediction panels
  download_kvasir.py    # dataset downloader
  nac/                  # model, losses, metrics, data loading
dataset/
  Kvasir-SEG/           # local dataset, ignored by Git
checkpoints/            # local training checkpoints, ignored by Git
outputs/                # local evaluation outputs, ignored by Git
predictions/            # local prediction panels, ignored by Git
```

## Quick Start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

Download Kvasir-SEG if it is not already present:

```bash
python code/download_kvasir.py --insecure-ssl
```

Run a small smoke test:

```bash
python code/train.py --epochs 1 --batch-size 2 --limit-train 8 --limit-val 4
```

Train with raw image-gradient external energy:

```bash
python code/train.py --external-mode image --epochs 50 --batch-size 4
```

Train with learned feature-gradient external energy:

```bash
python code/train.py --external-mode features --epochs 50 --batch-size 4
```

Evaluate the best checkpoint:

```bash
python code/evaluate.py --checkpoint checkpoints/best.pt --split test
```
