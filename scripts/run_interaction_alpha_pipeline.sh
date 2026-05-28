#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_CMD="${PYTHON_CMD:-python}"
LOGS_DIR="${LOGS_DIR:-outputs/logs}"
DATA_DIR="${DATA_DIR:-outputs/interaction_grip}"
FUTURE_DIR="${FUTURE_DIR:-outputs/interaction_grip_future}"
TEACHER_DIR="${TEACHER_DIR:-outputs/interaction_grip_teacher}"
ALPHA_DIR="${ALPHA_DIR:-outputs/interaction_grip_alpha}"
DEVICE="${DEVICE:-cuda}"
HISTORY="${HISTORY:-6}"
FUTURE="${FUTURE:-15}"
FUTURE_EPOCHS="${FUTURE_EPOCHS:-5}"
ALPHA_EPOCHS="${ALPHA_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1024}"

"$PYTHON_CMD" scripts/prepare_interaction_grip_dataset.py \
  --logs-dir "$LOGS_DIR" \
  --out-dir "$DATA_DIR" \
  --history "$HISTORY" \
  --future "$FUTURE"

"$PYTHON_CMD" -m interaction_modeling.train_future \
  --data-root "$DATA_DIR" \
  --output-dir "$FUTURE_DIR" \
  --device "$DEVICE" \
  --epochs "$FUTURE_EPOCHS" \
  --batch-size "$BATCH_SIZE"

"$PYTHON_CMD" -m interaction_modeling.distill_alpha_teacher \
  --data-root "$DATA_DIR" \
  --checkpoint "$FUTURE_DIR/best.pt" \
  --out-root "$TEACHER_DIR" \
  --device "$DEVICE"

"$PYTHON_CMD" -m interaction_modeling.train_alpha \
  --data-root "$TEACHER_DIR" \
  --output-dir "$ALPHA_DIR" \
  --device "$DEVICE" \
  --epochs "$ALPHA_EPOCHS" \
  --batch-size "$BATCH_SIZE"

