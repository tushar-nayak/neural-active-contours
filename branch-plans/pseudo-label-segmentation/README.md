# Pseudo-Label Segmentation

## Purpose

This branch explores weak supervision by generating labels automatically and
training a conventional segmentation model on those labels.

## Why it exists

- Purely unsupervised contour learning has been unstable.
- Pseudo-labels can turn the problem into something much closer to standard
  supervised segmentation.
- This path can still reduce manual annotation dependence if the pseudo-labels
  are good enough.

## What success looks like

- Pseudo-labels that are cleaner than the raw contour outputs.
- A segmentation model that beats the unsupervised contour baseline.
- A repeatable label-generation pipeline.

## Immediate next step

Pick one pseudo-label source and make it reproducible:

- SAM/SAM2 masks,
- classical image processing,
- or a pretrained medical segmentation model.

Then train a standard segmentation network on those labels and measure how
close it gets to the ground-truth masks.
