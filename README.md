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

Train both variants and compare them:

```bash
./scripts/train_and_compare.sh
```

Optional overrides:

```bash
EPOCHS=20 BATCH_SIZE=2 IMAGE_SIZE=192 ./scripts/train_and_compare.sh
```

## Experiment Log

### Dataset Setup

Kvasir-SEG was selected over ISIC and BBBC038 for the first implementation
because it is small, medical, 2D, and has clear polyp masks for evaluation. The
dataset is downloaded from the official Simula host and extracted locally to:

```text
dataset/Kvasir-SEG/
  images/
  masks/
```

The dataset contains 1000 images and 1000 masks. The whole `dataset/` folder is
ignored by Git, so the data stays local and is not pushed to GitHub.

### Baseline Smoke Test

The first smoke test confirmed that the code path worked:

- Kvasir images and masks loaded correctly.
- Training ran for both `image` and `features` external-energy modes.
- Evaluation produced Dice/IoU metrics and qualitative prediction panels.

However, the one-epoch smoke test did not produce meaningful masks. This was
expected because it used only a few samples and was intended to catch runtime
errors, not train a usable model.

### First Full-Run Failure Mode

The first longer image-gradient run showed a clear degenerate solution:

```text
epoch 001 loss=1.3463 val_dice=0.2732 val_iou=0.1736 pred_area=0.909 mask_area=0.159
epoch 002 loss=1.2392 val_dice=0.2547 val_iou=0.1595 pred_area=0.993 mask_area=0.159
epoch 003 loss=1.2207 val_dice=0.2547 val_iou=0.1595 pred_area=0.994 mask_area=0.159
```

The important signal is `pred_area`, not just Dice or IoU. The model was
predicting almost the entire image as polyp, while the real masks covered about
16% of the image. The Dice and IoU values were therefore not evidence of useful
segmentation; they were the overlap produced by an almost all-foreground mask.

### Diagnostics Added

To make this failure mode visible, training and evaluation now log:

- `pred_area`: fraction of pixels predicted foreground after thresholding.
- `soft_area`: average predicted probability before thresholding.
- `mask_area`: fraction of pixels marked foreground in the evaluation mask.

Interpretation:

- `pred_area` near `1.000` means the model is predicting almost everything.
- `pred_area` near `0.000` means the model is predicting almost nothing.
- `soft_area` near the expected mask area but poor `pred_area` means the model
  may be poorly calibrated around the fixed threshold.

### Loss Correction

The first loss version allowed a trivial all-foreground solution. Two issues were
identified:

- The finite-difference boundary energy added a small artificial value even for
  constant masks.
- The area prior was too weak and allowed predictions covering almost the whole
  image.

The loss was updated to:

- make constant masks have zero boundary energy;
- use an edge-attraction reward based on image or feature energy;
- strengthen the area prior around a plausible Kvasir polyp size;
- print `soft_area` during training so threshold calibration can be monitored.

The next run should be started fresh after pulling the latest branch:

```bash
git pull
./scripts/train_and_compare.sh
```

For a healthy run, `pred_area` should not stay near `1.000`, and `soft_area`
should generally stay closer to the Kvasir mask area range rather than collapsing
to all foreground or all background.
