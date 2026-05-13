#!/bin/bash

python src/main.py \
  --map Town04 \
  --num-vehicles 10 \
  --steps 500 \
  --fps 20 \
  --log-path outputs/logs/b1_idm_log.csv
