# Uncertainty Module

This package implements sigma estimators for B2/B3 planner experiments.

## Sigma Sources

- `estimated`: existing noise-std heuristic.
- `v1`, `v2`, `v3`: existing handcrafted runtime variants.
- `ai_v1`, `ai_v2`, `ai_v3`: existing noise residual models converted to sigma.
- `track_a`: rule-based physics + Kalman fusion from `uncertainty_module_prompt.md`.
- `track_b`: LSTM model trained to predict `sigma_label` from 10-step, 10-feature history.

## Track A

Track A has no PyTorch dependency.

```bash
python examples/demo_track_a.py
SIGMA_SOURCE=track_a PLANNERS="B2 B3" ./scripts/run_dataset_generation.sh
```

## Track B Training

Track B uses B1 neighbor logs as training data. The default pattern expects files like:

```text
outputs/logs/*_B1*_nbr.csv
```

Train locally or on Colab:

```bash
pip install torch numpy
python -m uncertainty.train \
  --log-root outputs/logs \
  --pattern "*_B1*_nbr.csv" \
  --output-dir outputs/uncertainty/checkpoints/track_b_lstm \
  --epochs 20 \
  --batch-size 32
```

Use the trained checkpoint in planner runs:

```bash
SIGMA_SOURCE=track_b \
SIGMA_MODEL_PATH=outputs/uncertainty/checkpoints/track_b_lstm/best.pt \
PLANNERS="B2 B3" \
./scripts/run_dataset_generation.sh
```

Smoke demos:

```bash
python examples/demo_track_a.py
python examples/demo_lstm_forward.py
```
