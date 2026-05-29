#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PLANNER="${PLANNER:-B1}"
PLANNER="${PLANNER^^}"
PLANNERS="${PLANNERS:-$PLANNER}"
SIGMA_SOURCE="${SIGMA_SOURCE:-estimated}"
SIGMA_SOURCES="${SIGMA_SOURCES:-$SIGMA_SOURCE}"
SIGMA_MODEL_PATH="${SIGMA_MODEL_PATH:-}"
TRACK_B_MODEL_PATH="${TRACK_B_MODEL_PATH:-$SIGMA_MODEL_PATH}"
SIGMA_HISTORY="${SIGMA_HISTORY:-10}"
SIGMA_DEVICE="${SIGMA_DEVICE:-}"
RUN_TAG="${RUN_TAG:-}"
RISK_GAIN="${RISK_GAIN:-0.9}"
MIN_DISTANCE_SCALE="${MIN_DISTANCE_SCALE:-0.1}"
ALPHA_MODEL_PATH="${ALPHA_MODEL_PATH:-}"
ALPHA_HISTORY="${ALPHA_HISTORY:-6}"
SEEDS="${SEEDS:-41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56}"
STEPS="${STEPS:-300}"
NOISE_SCALES="${NOISE_SCALES:-0.00 0.25 0.50 0.75 1.00 1.25 1.50 1.75 2.00}"
if [[ -z "${PYTHON_CMD:-}" ]]; then
  if [[ -x "${HOME}/miniconda3/envs/iwre-planner/bin/python" ]]; then
    PYTHON_CMD="${HOME}/miniconda3/envs/iwre-planner/bin/python"
  else
    PYTHON_CMD="python"
  fi
fi
CARLA_HOST="${CARLA_HOST:-localhost}"
CARLA_PORT="${CARLA_PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
RESTART_CARLA="${RESTART_CARLA:-1}"
CARLA_WAIT_TIMEOUT="${CARLA_WAIT_TIMEOUT:-120}"
CARLA_COOLDOWN_SEC="${CARLA_COOLDOWN_SEC:-5}"
CARLA_LOG_DIR="${CARLA_LOG_DIR:-outputs/carla_logs}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
RUN_MANIFEST="${RUN_MANIFEST:-outputs/logs/dataset_generation_manifest.csv}"

if [[ "${SIGMA_SOURCE,,}" == "all" || "${SIGMA_SOURCES,,}" == "all" ]]; then
  SIGMA_SOURCES="v1 v2 v3 ai_v1 ai_v2 ai_v3 track_a track_b"
fi

DEFAULT_SCENARIOS=(
  "configs/scenarios/lf_basic.yaml"
  "configs/scenarios/front_sudden_brake.yaml"
  "configs/scenarios/front_sudden_accel.yaml"
  "configs/scenarios/cut_in_left.yaml"
  "configs/scenarios/cut_in_right.yaml"
  "configs/scenarios/ego_lane_change_conditional_left.yaml"
  "configs/scenarios/ego_lane_change_conditional_right.yaml"
  "configs/scenarios/dense_mixed.yaml"
)

if [[ -n "${SCENARIOS:-}" ]]; then
  read -r -a SCENARIO_LIST <<< "$SCENARIOS"
else
  SCENARIO_LIST=("${DEFAULT_SCENARIOS[@]}")
fi

echo "[INFO] Dataset generation"
echo "[INFO] planners=${PLANNERS}"
echo "[INFO] sigma_sources=${SIGMA_SOURCES}"
echo "[INFO] sigma_model_path=${SIGMA_MODEL_PATH:-<auto/none>}"
echo "[INFO] track_b_model_path=${TRACK_B_MODEL_PATH:-<none>}"
echo "[INFO] sigma_history=${SIGMA_HISTORY}"
echo "[INFO] sigma_device=${SIGMA_DEVICE:-<auto>}"
echo "[INFO] run_tag=${RUN_TAG:-<none>}"
echo "[INFO] risk_gain=${RISK_GAIN}"
echo "[INFO] min_distance_scale=${MIN_DISTANCE_SCALE}"
echo "[INFO] alpha_model_path=${ALPHA_MODEL_PATH:-<heuristic>}"
echo "[INFO] python=${PYTHON_CMD}"
echo "[INFO] seeds=${SEEDS}"
echo "[INFO] steps=${STEPS}"
echo "[INFO] noise_scales=${NOISE_SCALES}"
echo "[INFO] scenarios=${SCENARIO_LIST[*]}"
echo "[INFO] restart_carla=${RESTART_CARLA}"
echo "[INFO] continue_on_error=${CONTINUE_ON_ERROR}"
echo "[INFO] run_manifest=${RUN_MANIFEST}"

mkdir -p "$CARLA_LOG_DIR"
mkdir -p "$(dirname "$RUN_MANIFEST")"

if [[ ! -f "$RUN_MANIFEST" ]]; then
  printf '%s\n' \
    "run_id,planner,sigma_source,sigma_model_path,alpha_model_path,scenario,scenario_id,seed,steps,noise_scale,risk_gain,min_distance_scale,status,ego_log,nbr_log,carla_log" \
    > "$RUN_MANIFEST"
fi

CARLA_PID=""

wait_for_carla() {
  local deadline=$((SECONDS + CARLA_WAIT_TIMEOUT))
  echo "[INFO] Waiting for CARLA on ${CARLA_HOST}:${CARLA_PORT}"

  while (( SECONDS < deadline )); do
    if "$PYTHON_CMD" - "$CARLA_HOST" "$CARLA_PORT" <<'PY' >/dev/null 2>&1
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1.0)
try:
    sock.connect((host, port))
finally:
    sock.close()
PY
    then
      echo "[INFO] CARLA is ready"
      return 0
    fi
    sleep 2
  done

  echo "[ERROR] Timed out waiting for CARLA after ${CARLA_WAIT_TIMEOUT}s" >&2
  return 1
}

start_carla() {
  local run_id="$1"
  local carla_log="${CARLA_LOG_DIR}/${run_id}_carla.log"

  echo "[INFO] Starting CARLA for ${run_id}"
  setsid ./scripts/run_carla.sh > "$carla_log" 2>&1 &
  CARLA_PID="$!"
  echo "[INFO] CARLA pid=${CARLA_PID}, log=${carla_log}"
  wait_for_carla
}

stop_carla() {
  if [[ -n "${CARLA_PID}" ]]; then
    echo -e "[INFO] Stopping CARLA pid=${CARLA_PID}\n"
    kill -TERM "-${CARLA_PID}" >/dev/null 2>&1 || true
    sleep "$CARLA_COOLDOWN_SEC"
    kill -KILL "-${CARLA_PID}" >/dev/null 2>&1 || true
    wait "$CARLA_PID" >/dev/null 2>&1 || true
    CARLA_PID=""
  fi
}

run_simulation() {
  local scenario="$1"
  local planner="$2"
  local seed="$3"
  local run_id="$4"
  local noise_scale="$5"
  local sigma_source="$6"
  local sigma_model_path="$7"

  if [[ "$RESTART_CARLA" == "1" ]]; then
    start_carla "$run_id"
  fi

  local cmd=(
    "$PYTHON_CMD" src/main.py
    --host "$CARLA_HOST"
    --port "$CARLA_PORT"
    --tm-port "$TM_PORT"
    --scenario "$scenario"
    --planner "$planner"
    --run-id "$run_id"
    --sigma-source "$sigma_source"
    --sigma-history "$SIGMA_HISTORY"
    --risk-gain "$RISK_GAIN"
    --min-distance-scale "$MIN_DISTANCE_SCALE"
    --seed "$seed"
    --steps "$STEPS"
    --noise-scale "$noise_scale"
    --alpha-history "$ALPHA_HISTORY"
  )
  if [[ -n "$ALPHA_MODEL_PATH" ]]; then
    cmd+=(--alpha-model-path "$ALPHA_MODEL_PATH")
  fi
  if [[ -n "$SIGMA_DEVICE" ]]; then
    cmd+=(--sigma-device "$SIGMA_DEVICE")
  fi
  if [[ -n "$sigma_model_path" ]]; then
    cmd+=(--sigma-model-path "$sigma_model_path")
  fi

  set +e
  "${cmd[@]}"
  local status="$?"
  set -e

  if [[ "$RESTART_CARLA" == "1" ]]; then
    stop_carla
  fi

  return "$status"
}

sanitize_tag() {
  local value="$1"
  value="${value//[^A-Za-z0-9_.-]/_}"
  value="${value##_}"
  value="${value%%_}"
  printf '%s' "$value"
}

make_run_id() {
  local scenario_id="$1"
  local planner="$2"
  local sigma_source="$3"
  local seed="$4"
  local tag="$5"

  if [[ "$planner" == "B1" ]]; then
    if [[ -n "$tag" ]]; then
      printf 's%s_%s_%s_seed%s' "$scenario_id" "$planner" "$tag" "$seed"
    else
      printf 's%s_%s_seed%s' "$scenario_id" "$planner" "$seed"
    fi
    return
  fi

  if [[ -n "$tag" ]]; then
    printf 's%s_%s_sig%s_%s_seed%s' "$scenario_id" "$planner" "$sigma_source" "$tag" "$seed"
  else
    printf 's%s_%s_sig%s_seed%s' "$scenario_id" "$planner" "$sigma_source" "$seed"
  fi
}

append_manifest() {
  local run_id="$1"
  local planner="$2"
  local sigma_source="$3"
  local sigma_model_path="$4"
  local scenario="$5"
  local scenario_id="$6"
  local seed="$7"
  local noise_scale="$8"
  local status="$9"
  local carla_log="${CARLA_LOG_DIR}/${run_id}_carla.log"
  local ego_log="outputs/logs/${run_id}_ego.csv"
  local nbr_log="outputs/logs/${run_id}_nbr.csv"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$run_id" \
    "$planner" \
    "$sigma_source" \
    "$sigma_model_path" \
    "$ALPHA_MODEL_PATH" \
    "$scenario" \
    "$scenario_id" \
    "$seed" \
    "$STEPS" \
    "$noise_scale" \
    "$RISK_GAIN" \
    "$MIN_DISTANCE_SCALE" \
    "$status" \
    "$ego_log" \
    "$nbr_log" \
    "$carla_log" \
    >> "$RUN_MANIFEST"
}

trap stop_carla EXIT

if [[ "$RESTART_CARLA" != "1" ]]; then
  echo "[INFO] RESTART_CARLA=0, so make sure CARLA is already running in another terminal."
  wait_for_carla
fi

read -r -a PLANNER_LIST <<< "$PLANNERS"
read -r -a SIGMA_SOURCE_LIST <<< "$SIGMA_SOURCES"

for scenario in "${SCENARIO_LIST[@]}"; do
  if [[ ! -f "$scenario" ]]; then
    echo "[ERROR] Scenario file not found: $scenario" >&2
    exit 1
  fi

  scenario_id="$(
    "$PYTHON_CMD" -c \
      "import sys; sys.path.insert(0, 'src'); from scenario.scenario_loader import load_scenario; print(load_scenario('$scenario').scenario_id)"
  )"

  read -r -a NOISE_SCALE_LIST <<< "$NOISE_SCALES"
  noise_scale_count="${#NOISE_SCALE_LIST[@]}"

  for planner_item in "${PLANNER_LIST[@]}"; do
    planner_item="${planner_item^^}"
    if [[ "$planner_item" == "B1" ]]; then
      planner_sigma_sources=("none")
    else
      planner_sigma_sources=("${SIGMA_SOURCE_LIST[@]}")
    fi

    for sigma_source_item in "${planner_sigma_sources[@]}"; do
      sigma_model_path=""
      if [[ "$sigma_source_item" == "none" ]]; then
        sigma_source_item="estimated"
      fi
      if [[ "$sigma_source_item" == "model" ]]; then
        sigma_model_path="$SIGMA_MODEL_PATH"
      fi
      if [[ "$sigma_source_item" == "track_b" ]]; then
        sigma_model_path="$TRACK_B_MODEL_PATH"
      fi
      if [[ ( "$sigma_source_item" == "model" || "$sigma_source_item" == "track_b" ) && -z "$sigma_model_path" ]]; then
        echo "[ERROR] SIGMA_SOURCE=${sigma_source_item} requires SIGMA_MODEL_PATH." >&2
        exit 1
      fi
      tag="$(sanitize_tag "$RUN_TAG")"

      for seed in $SEEDS; do
        run_id="$(make_run_id "$scenario_id" "$planner_item" "$sigma_source_item" "$seed" "$tag")"
        noise_index=$(( (seed + noise_scale_count - 1) % noise_scale_count ))
        noise_scale="${NOISE_SCALE_LIST[$noise_index]}"
        echo "[INFO] Running ${run_id} noise_scale=${noise_scale}"

        status="ok"
        if ! run_simulation "$scenario" "$planner_item" "$seed" "$run_id" "$noise_scale" "$sigma_source_item" "$sigma_model_path"; then
          echo "[ERROR] Simulation failed: ${run_id}" >&2
          status="failed"
          append_manifest "$run_id" "$planner_item" "$sigma_source_item" "$sigma_model_path" "$scenario" "$scenario_id" "$seed" "$noise_scale" "$status"
          if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
            exit 1
          fi
        else
          append_manifest "$run_id" "$planner_item" "$sigma_source_item" "$sigma_model_path" "$scenario" "$scenario_id" "$seed" "$noise_scale" "$status"
        fi
      done
    done
  done
done

echo "[INFO] Dataset generation finished. Logs are in outputs/logs/"
