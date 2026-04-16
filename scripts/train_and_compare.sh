#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
BASE_CHANNELS="${BASE_CHANNELS:-24}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SPLIT="${SPLIT:-test}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/dataset/Kvasir-SEG}"

RUN_DIR="$ROOT_DIR/runs/compare-$(date +%Y%m%d-%H%M%S)"
IMAGE_CKPT="$RUN_DIR/image-gradient/checkpoints"
FEATURE_CKPT="$RUN_DIR/learned-features/checkpoints"
IMAGE_EVAL="$RUN_DIR/image-gradient/eval"
FEATURE_EVAL="$RUN_DIR/learned-features/eval"
SUMMARY="$RUN_DIR/summary.txt"

mkdir -p "$RUN_DIR"

echo "Run directory: $RUN_DIR"
echo "Dataset: $DATA_ROOT"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Image size: $IMAGE_SIZE"
echo "Evaluation split: $SPLIT"
echo

if [[ ! -d "$DATA_ROOT/images" || ! -d "$DATA_ROOT/masks" ]]; then
  echo "Kvasir-SEG was not found at $DATA_ROOT"
  echo "Run: $PYTHON_BIN code/download_kvasir.py --insecure-ssl"
  exit 1
fi

echo "== Training image-gradient external energy =="
"$PYTHON_BIN" code/train.py \
  --data-root "$DATA_ROOT" \
  --external-mode image \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --base-channels "$BASE_CHANNELS" \
  --num-workers "$NUM_WORKERS" \
  --output-dir "$IMAGE_CKPT" \
  2>&1 | tee "$RUN_DIR/image-gradient-train.log"

echo
echo "== Training learned-feature external energy =="
"$PYTHON_BIN" code/train.py \
  --data-root "$DATA_ROOT" \
  --external-mode features \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --image-size "$IMAGE_SIZE" \
  --base-channels "$BASE_CHANNELS" \
  --num-workers "$NUM_WORKERS" \
  --output-dir "$FEATURE_CKPT" \
  2>&1 | tee "$RUN_DIR/learned-features-train.log"

echo
echo "== Evaluating image-gradient model =="
"$PYTHON_BIN" code/evaluate.py \
  --checkpoint "$IMAGE_CKPT/best.pt" \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --output-dir "$IMAGE_EVAL" \
  2>&1 | tee "$RUN_DIR/image-gradient-eval.log"

echo
echo "== Evaluating learned-feature model =="
"$PYTHON_BIN" code/evaluate.py \
  --checkpoint "$FEATURE_CKPT/best.pt" \
  --data-root "$DATA_ROOT" \
  --split "$SPLIT" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --output-dir "$FEATURE_EVAL" \
  2>&1 | tee "$RUN_DIR/learned-features-eval.log"

image_dice="$(awk -F= '/^dice=/{print $2}' "$RUN_DIR/image-gradient-eval.log" | tail -1)"
image_iou="$(awk -F= '/^iou=/{print $2}' "$RUN_DIR/image-gradient-eval.log" | tail -1)"
feature_dice="$(awk -F= '/^dice=/{print $2}' "$RUN_DIR/learned-features-eval.log" | tail -1)"
feature_iou="$(awk -F= '/^iou=/{print $2}' "$RUN_DIR/learned-features-eval.log" | tail -1)"

{
  echo "Comparison summary"
  echo "=================="
  echo "Run directory: $RUN_DIR"
  echo "Split: $SPLIT"
  echo
  printf "%-26s %-10s %-10s\n" "model" "dice" "iou"
  printf "%-26s %-10s %-10s\n" "image-gradient" "$image_dice" "$image_iou"
  printf "%-26s %-10s %-10s\n" "learned-features" "$feature_dice" "$feature_iou"
  echo
  echo "Qualitative samples:"
  echo "  $IMAGE_EVAL"
  echo "  $FEATURE_EVAL"
} | tee "$SUMMARY"

echo
echo "Done. Summary written to $SUMMARY"

