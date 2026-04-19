# U-Net Baseline

## Purpose

This branch is the supervised reference model. Its job is to answer a simple
question: how well does a standard segmentation network perform on Kvasir-SEG
when trained directly on masks?

## Why it exists

- It gives us a real performance ceiling for Dice and IoU.
- It tells us whether the contour-based approach is competitive at all.
- It provides a stable baseline before adding any extra regularization or weak
  supervision tricks.

## What success looks like

- Stable training with standard supervised losses.
- Better Dice/IoU than the unsupervised contour model.
- Clean qualitative masks without threshold calibration hacks.

## Immediate next step

Implement the simplest strong baseline first:

- U-Net encoder-decoder.
- Dice loss plus BCE or focal BCE.
- Standard train/val/test reporting on Kvasir-SEG.
