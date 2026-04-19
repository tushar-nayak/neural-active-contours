# Neural Active Contours Code

This folder contains a complete PyTorch pipeline for unsupervised neural active
contours on Kvasir-SEG.

The masks are used only for validation and testing metrics. Training uses the
input images plus an active-contour energy loss.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

The dataset is expected at:

```text
dataset/Kvasir-SEG/
  images/
  masks/
```

If needed, download it with:

```bash
python code/download_kvasir.py
```

## Train

Raw image-gradient external energy:

```bash
python code/train.py --external-mode image --epochs 50 --batch-size 4
```

Learned feature-gradient external energy:

```bash
python code/train.py --external-mode features --epochs 50 --batch-size 4
```

Training now calibrates the validation decision threshold by default. Use
`--threshold 0.5` for a fixed cutoff or `--threshold auto` during evaluation
to sweep thresholds on the chosen split.

The default comparison script now runs the image-gradient baseline only; set
`COMPARE_FEATURES=1` if you want the learned-feature branch included too.

For a quick smoke run:

```bash
python code/train.py --epochs 1 --batch-size 2 --limit-train 8 --limit-val 4
```

## Evaluate

```bash
python code/evaluate.py --checkpoint checkpoints/best.pt --split test
```

## Export Predictions

```bash
python code/predict.py --checkpoint checkpoints/best.pt --split test --max-images 20
```

Outputs are written to ignored local folders such as `checkpoints/` and
`predictions/`.
