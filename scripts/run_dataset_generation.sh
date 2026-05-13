#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PLANNER="${PLANNER:-B1}"
SEEDS="${SEEDS:-1 2 3 4 5}"
PYTHON_CMD="${PYTHON_CMD:-python}"
CARLA_HOST="${CARLA_HOST:-localhost}"
CARLA_PORT="${CARLA_PORT:-2000}"
TM_PORT="${TM_PORT:-8000}"
RESTART_CARLA="${RESTART_CARLA:-1}"
CARLA_WAIT_TIMEOUT="${CARLA_WAIT_TIMEOUT:-120}"
CARLA_COOLDOWN_SEC="${CARLA_COOLDOWN_SEC:-5}"
CARLA_LOG_DIR="${CARLA_LOG_DIR:-outputs/carla_logs}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

DEFAULT_SCENARIOS=(
  "configs/scenarios/lf_basic.yaml"
  "configs/scenarios/front_sudden_brake.yaml"
  "configs/scenarios/front_sudden_accel.yaml"
  "configs/scenarios/cut_in_left.yaml"
  "configs/scenarios/cut_in_right.yaml"
  "configs/scenarios/ego_lane_change_conditional_left.yaml"
  "configs/scenarios/ego_lane_change_conditional_right.yaml"
  "configs/scenarios/dense_mixed.yaml"
  "configs/scenarios/random_batch.yaml"
)

if [[ -n "${SCENARIOS:-}" ]]; then
  read -r -a SCENARIO_LIST <<< "$SCENARIOS"
else
  SCENARIO_LIST=("${DEFAULT_SCENARIOS[@]}")
fi

echo "[INFO] Dataset generation"
echo "[INFO] planner=${PLANNER}"
echo "[INFO] seeds=${SEEDS}"
echo "[INFO] scenarios=${SCENARIO_LIST[*]}"
echo "[INFO] restart_carla=${RESTART_CARLA}"
echo "[INFO] continue_on_error=${CONTINUE_ON_ERROR}"

mkdir -p "$CARLA_LOG_DIR"

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

  if [[ "$RESTART_CARLA" == "1" ]]; then
    start_carla "$run_id"
  fi

  set +e
  "$PYTHON_CMD" src/main.py \
    --host "$CARLA_HOST" \
    --port "$CARLA_PORT" \
    --tm-port "$TM_PORT" \
    --scenario "$scenario" \
    --planner "$planner" \
    --run-id "$run_id" \
    --seed "$seed"
  local status="$?"
  set -e

  if [[ "$RESTART_CARLA" == "1" ]]; then
    stop_carla
  fi

  return "$status"
}

trap stop_carla EXIT

if [[ "$RESTART_CARLA" != "1" ]]; then
  echo "[INFO] RESTART_CARLA=0, so make sure CARLA is already running in another terminal."
  wait_for_carla
fi

for scenario in "${SCENARIO_LIST[@]}"; do
  if [[ ! -f "$scenario" ]]; then
    echo "[ERROR] Scenario file not found: $scenario" >&2
    exit 1
  fi

  scenario_id="$(
    "$PYTHON_CMD" -c \
      "import sys; sys.path.insert(0, 'src'); from scenario.scenario_loader import load_scenario; print(load_scenario('$scenario').scenario_id)"
  )"

  for seed in $SEEDS; do
    run_id="s${scenario_id}_${PLANNER}_seed${seed}"
    echo "[INFO] Running ${run_id}"

    if ! run_simulation "$scenario" "$PLANNER" "$seed" "$run_id"; then
      echo "[ERROR] Simulation failed: ${run_id}" >&2
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        exit 1
      fi
    fi
  done
done

echo "[INFO] Dataset generation finished. Logs are in outputs/logs/"
