# Self-Supervised Pretrain

## Purpose

This branch treats representation learning as the first problem and segmentation
as the second problem.

## Why it exists

- The contour model may be failing partly because the encoder is weak.
- Self-supervised pretraining can learn better low-level and mid-level image
  features without using masks.
- A stronger encoder can make later supervised fine-tuning easier.

## What success looks like

- Better transfer than training from scratch.
- Faster convergence during supervised fine-tuning.
- Improved boundary localization once a segmentation head is attached.

## Immediate next step

Pretrain the encoder on the available image data, then fine-tune a segmentation
head:

- contrastive learning,
- masked image modeling,
- or a simple augmentation-heavy self-supervised setup.

After pretraining, compare fine-tuning performance to the plain U-Net baseline.
