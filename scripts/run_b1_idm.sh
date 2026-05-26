#!/bin/bash

if [[ -z "${PYTHON_CMD:-}" ]]; then
  if [[ -x "${HOME}/miniconda3/envs/iwre-planner/bin/python" ]]; then
    PYTHON_CMD="${HOME}/miniconda3/envs/iwre-planner/bin/python"
  else
    PYTHON_CMD="python"
  fi
fi

"${PYTHON_CMD}" src/main.py \
  --map Town04 \
  --num-vehicles 10 \
  --steps 500 \
  --fps 20 \
  --log-path outputs/logs/b1_idm_log.csv
