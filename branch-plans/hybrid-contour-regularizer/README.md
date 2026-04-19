# Hybrid Contour Regularizer

## Purpose

This branch keeps the supervised U-Net baseline, then adds contour energy as an
auxiliary regularizer instead of the main training signal.

## Why it exists

- The current unsupervised contour-only objective is underconstrained.
- A supervised mask loss gives the model the right target.
- A contour prior can still help shape smoother, more boundary-aware masks.

## What success looks like

- Better or equal Dice compared with the U-Net baseline.
- Visibly cleaner boundaries and less ragged foreground.
- The contour term helps without destabilizing training.

## Immediate next step

Start from the U-Net baseline and add one contour term at a time:

- boundary alignment,
- compactness or perimeter penalty,
- optional curvature regularization.

Keep the supervised mask loss dominant.
