#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SWEEP_MANIFEST="${SWEEP_MANIFEST:-outputs/logs/track_b_risk_sweep_manifest.csv}"
SWEEP_ANALYSIS_DIR="${SWEEP_ANALYSIS_DIR:-outputs/analysis/track_b_risk_sweep}"
TRACK_B_MODEL_PATH="${TRACK_B_MODEL_PATH:-outputs/uncertainty/checkpoints/track_b_lstm/best.pt}"
FOLLOWUP_MANIFEST="${FOLLOWUP_MANIFEST:-outputs/logs/track_b_best_seed31_50_manifest.csv}"
FOLLOWUP_SEEDS="${FOLLOWUP_SEEDS:-31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50}"
FOLLOWUP_STEPS="${FOLLOWUP_STEPS:-300}"
WAIT_FOR_SWEEP="${WAIT_FOR_SWEEP:-1}"
POLL_SEC="${POLL_SEC:-120}"

if [[ "$WAIT_FOR_SWEEP" == "1" ]]; then
  python scripts/analyze_track_b_risk_sweep.py \
    --manifest "$SWEEP_MANIFEST" \
    --output-dir "$SWEEP_ANALYSIS_DIR" \
    --wait \
    --poll-sec "$POLL_SEC"
else
  python scripts/analyze_track_b_risk_sweep.py \
    --manifest "$SWEEP_MANIFEST" \
    --output-dir "$SWEEP_ANALYSIS_DIR"
fi

source "${SWEEP_ANALYSIS_DIR}/best.env"

FOLLOWUP_TAG="trackb_best_${BEST_SWEEP_TAG}_seed31_50"

echo "[INFO] Continuing with best config from sweep"
echo "[INFO] risk_gain=${BEST_RISK_GAIN}"
echo "[INFO] min_distance_scale=${BEST_MIN_DISTANCE_SCALE}"
echo "[INFO] selection_mode=${BEST_SELECTION_MODE}"
echo "[INFO] run_tag=${FOLLOWUP_TAG}"
echo "[INFO] followup_manifest=${FOLLOWUP_MANIFEST}"
echo "[INFO] scenarios=<run_dataset_generation.sh defaults>"
echo "[INFO] seeds=${FOLLOWUP_SEEDS}"

PLANNERS="B2 B3" \
SIGMA_SOURCE=track_b \
TRACK_B_MODEL_PATH="$TRACK_B_MODEL_PATH" \
RISK_GAIN="$BEST_RISK_GAIN" \
MIN_DISTANCE_SCALE="$BEST_MIN_DISTANCE_SCALE" \
RUN_TAG="$FOLLOWUP_TAG" \
RUN_MANIFEST="$FOLLOWUP_MANIFEST" \
SEEDS="$FOLLOWUP_SEEDS" \
STEPS="$FOLLOWUP_STEPS" \
./scripts/run_dataset_generation.sh
