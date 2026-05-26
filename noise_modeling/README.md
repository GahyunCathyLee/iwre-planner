# Noise Modeling Training

Install dependencies:

```bash
pip install -r noise_modeling/requirements.txt
```

Smoke test on a tiny subset:

```bash
python -m noise_modeling.train --config noise_modeling/configs/ai_v2_hetero_mlp.yaml --limit-shards 2
python -m noise_modeling.evaluate --checkpoint outputs/noise_modeling/checkpoints/ai_v2_hetero_mlp/best.pt --split test --limit-shards 2
```

Full training:

```bash
python -m noise_modeling.train --config noise_modeling/configs/ai_v1_mlp.yaml
python -m noise_modeling.train --config noise_modeling/configs/ai_v2_hetero_mlp.yaml
python -m noise_modeling.train --config noise_modeling/configs/ai_v3_history_gru.yaml
```

Evaluate:

```bash
python -m noise_modeling.evaluate --checkpoint outputs/noise_modeling/checkpoints/ai_v1_mlp/best.pt --split test
python -m noise_modeling.evaluate --checkpoint outputs/noise_modeling/checkpoints/ai_v2_hetero_mlp/best.pt --split test
python -m noise_modeling.evaluate --checkpoint outputs/noise_modeling/checkpoints/ai_v3_history_gru/best.pt --split test
```

