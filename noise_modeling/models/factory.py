from __future__ import annotations

from typing import Dict

from noise_modeling.models.history import EgoCentricHistoryNoiseModelV3
from noise_modeling.models.mlp import EgoCentricHeteroscedasticMLPNoiseModelV2, EgoCentricMLPNoiseRegressorV1


def build_model(config: Dict):
    model_cfg = config["model"]
    model_type = model_cfg["type"]
    if model_type == "mlp_v1":
        return EgoCentricMLPNoiseRegressorV1(
            input_dim=int(model_cfg["input_dim"]),
            hidden_dims=model_cfg.get("hidden_dims", [128, 128]),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if model_type == "hetero_mlp_v2":
        return EgoCentricHeteroscedasticMLPNoiseModelV2(
            input_dim=int(model_cfg["input_dim"]),
            hidden_dims=model_cfg.get("hidden_dims", [128, 128, 64]),
            dropout=float(model_cfg.get("dropout", 0.0)),
            min_sigma=float(model_cfg.get("min_sigma", 1.0e-4)),
            max_sigma=float(model_cfg["max_sigma"]) if model_cfg.get("max_sigma") is not None else None,
        )
    if model_type == "history_gru_v3":
        return EgoCentricHistoryNoiseModelV3(
            input_dim=int(model_cfg["input_dim"]),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_layers=int(model_cfg.get("num_layers", 1)),
            head_hidden_dims=model_cfg.get("head_hidden_dims", [128, 64]),
            dropout=float(model_cfg.get("dropout", 0.0)),
            min_sigma=float(model_cfg.get("min_sigma", 1.0e-4)),
            max_sigma=float(model_cfg["max_sigma"]) if model_cfg.get("max_sigma") is not None else None,
            output_type=model_cfg.get("output_type", "heteroscedastic"),
        )
    raise ValueError(f"Unknown model type: {model_type}")
