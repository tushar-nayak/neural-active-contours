# Unsupervised Neural Active Contours

## Report

This repository implements an unsupervised neural active-contour pipeline for
2D segmentation. The core idea is to replace a hand-evolved snake with a small
CNN that predicts a foreground probability map, then train that map with an
active-contour-style objective instead of supervised mask labels.

The project was evaluated on Kvasir-SEG, using two external-energy variants:

- `image`: raw image-gradient energy
- `features`: learned feature-gradient energy

The current codebase includes calibrated threshold selection, a compactness
penalty, and a heuristic area prior. Those changes were added after the first
round of runs showed that a fixed `0.5` threshold and a weaker contour loss were
producing degenerate masks.

## What Was Built

The system is organized around four pieces:

- `code/nac/model.py`: a lightweight encoder-decoder that outputs mask logits
  and intermediate features.
- `code/nac/losses.py`: an active-contour-inspired loss with edge alignment,
  smoothness, curvature, compactness, binary regularization, and a heuristic
  area prior.
- `code/train.py`: training with validation-time threshold calibration.
- `code/evaluate.py`: checkpoint evaluation and qualitative panel export.

The compare script, `scripts/train_and_compare.sh`, runs the image-gradient
baseline by default and can include the learned-feature branch when
`COMPARE_FEATURES=1` is set.

## Experimental Setup

The main comparison runs used the same small smoke-scale configuration:

- dataset: Kvasir-SEG
- image size: `128`
- base channels: `8`
- train limit: `128`
- validation limit: `32`
- epochs: `4`

This is not a full benchmark. It is a controlled diagnostic setup meant to show
which objective behaves better and how the model fails when it does fail.

## Results

### Image-Only Baseline

Run directory:

- [runs/compare-20260419-175644/summary.txt](/home/sofa/host_dir/nad/neural-active-contours/runs/compare-20260419-175644/summary.txt)

Test results:

- Dice: `0.3573`
- IoU: `0.2422`
- threshold: `0.20`
- predicted area: `0.4231`
- ground-truth mask area: `0.1299`

Interpretation:

- The model is no longer collapsing to an empty mask.
- It still predicts too much foreground, by roughly a factor of three relative
  to the ground-truth area.
- The calibrated threshold matters. A fixed `0.5` cutoff would miss this model
  entirely, because the learned probabilities are lower than that.

### Feature-Enabled Comparison

Run directory:

- [runs/compare-20260419-175837/summary.txt](/home/sofa/host_dir/nad/neural-active-contours/runs/compare-20260419-175837/summary.txt)

Test results:

- Image-gradient branch:
  - Dice: `0.3452`
  - IoU: `0.2309`
  - threshold: `0.20`
  - predicted area: `0.4736`
  - ground-truth mask area: `0.1299`
- Learned-features branch:
  - Dice: `0.3548`
  - IoU: `0.2399`
  - threshold: `0.10`
  - predicted area: `0.3948`
  - ground-truth mask area: `0.1299`

Interpretation:

- On this run, the learned-feature branch slightly outperformed the raw
  image-gradient branch.
- The gap is small, so the learned features are not a clear win yet.
- Both branches still overpredict foreground substantially, but the learned
  feature branch is less bloated than the image-gradient baseline in this run.
- The very low learned threshold (`0.10`) shows that the model outputs are still
  not naturally calibrated around a crisp foreground/background split.

## Historical Failure Modes

The initial implementation had two recurring problems:

- masks collapsed toward all foreground in some runs;
- masks collapsed toward all background after thresholding in other runs.

Those failures came from a mismatch between the learned probability scale and
the hard `0.5` threshold, plus an objective that was too weakly constrained to
prevent broad, low-information solutions.

The later loss revisions fixed the obvious degeneracies, but they did not turn
the system into a clean polyp segmenter. They moved it from "obviously broken"
to "plausible but still underconstrained."

## Current Diagnosis

What is working:

- data loading and train/val/test splitting;
- end-to-end training for both external-energy modes;
- checkpointing and evaluation;
- calibrated threshold selection;
- qualitative output export;
- a loss that no longer collapses immediately.

What is still not working well:

- the predicted masks are still too large;
- Dice is modest rather than strong;
- the learned-feature branch is only marginally better, not decisively better;
- the objective still prefers broad foreground regions over tight contours.

Why this happens:

- the loss is unsupervised, so it must infer the segmentation boundary from
  image structure alone;
- the area prior is only a heuristic, not a true estimate of object size;
- the output probabilities are not naturally centered on a standard threshold;
- the model can satisfy the contour objective with a coarse foreground blob
  instead of a precise boundary.

## Conclusion

The repository now contains a working experimental pipeline and a clear record
of the failure modes and partial fixes.

The image-gradient baseline is stable enough to use as a reference point. The
learned-feature branch is promising, but only slightly better in the latest
comparison and not yet strong enough to justify extra complexity on its own.

The main open problem is objective quality: the current loss is good enough to
avoid trivial collapse, but not yet strong enough to produce consistently tight
segmentation contours.

## Branch Plans

The next-stage experiment paths are documented here:

- [Branch plans index](./branch-plans/README.md)
- [U-Net baseline](./branch-plans/unet-baseline/README.md)
- [Hybrid contour regularizer](./branch-plans/hybrid-contour-regularizer/README.md)
- [Pseudo-label segmentation](./branch-plans/pseudo-label-segmentation/README.md)
- [Self-supervised pretrain](./branch-plans/self-supervised-pretrain/README.md)

## Quick Start

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

Download Kvasir-SEG if needed:

```bash
python code/download_kvasir.py --insecure-ssl
```

Run the image-only comparison:

```bash
./scripts/train_and_compare.sh
```

Include the learned-feature branch too:

```bash
COMPARE_FEATURES=1 ./scripts/train_and_compare.sh
```

Useful overrides:

```bash
EPOCHS=20 BATCH_SIZE=2 IMAGE_SIZE=192 ./scripts/train_and_compare.sh
```
