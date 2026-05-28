from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict

import numpy as np


DEFAULT_SIGMA_MODEL_PATHS = {
    "ai_v1": "outputs/noise_modeling/checkpoints/ai_v1_mlp/best.pt",
    "ai_v2": "outputs/noise_modeling/checkpoints/ai_v2_hetero_mlp/best.pt",
    "ai_v3": "outputs/noise_modeling/checkpoints/ai_v3_history_gru/best.pt",
}


class NoiseSigmaRuntime:
    """Online sigma inference for trained ego-centric noise models."""

    def __init__(
        self,
        checkpoint_path: str,
        history: int = 10,
        device: str | None = None,
    ):
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        import torch
        from noise_modeling.data.normalization import load_normalizers
        from noise_modeling.models.factory import build_model

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.history = max(1, int(history))
        self.checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(str(self.checkpoint_path), map_location=self.device)
        self.config = ckpt["config"]
        self.data_config = self.config["data"]
        self.feature_set = str(self.data_config["feature_set"])
        self.use_sequence = bool(self.data_config.get("use_sequence", False))
        self.normalize_features = bool(self.data_config.get("normalize_features", True))
        self.normalize_targets = bool(self.data_config.get("normalize_targets", True))
        self.normalizers = load_normalizers(self._normalization_path())
        self.target_std = np.asarray(ckpt.get("target_std", self.normalizers["target_rel"].std), dtype=np.float32)

        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self._history_by_key: Dict[tuple[str, int], deque[np.ndarray]] = {}

    def predict(self, key: tuple[str, int], features: Dict[str, float]) -> float:
        raw_feature = self._feature_array(features)
        if self.use_sequence:
            model_input = self._sequence_input(key, raw_feature)
        else:
            model_input = self._normalize(raw_feature[None, :], self._feature_norm_key())

        with self.torch.no_grad():
            tensor = self.torch.from_numpy(model_input.astype(np.float32)).to(self.device)
            output = self.model(tensor)

        mu = output["mu"][0].detach().cpu().numpy()
        sigma = output.get("sigma")
        sigma_np = sigma[0].detach().cpu().numpy() if sigma is not None else None
        if self.normalize_targets:
            mu = mu * self.target_std
            if sigma_np is not None:
                sigma_np = sigma_np * self.target_std

        vector = sigma_np if sigma_np is not None else np.abs(mu)
        position = float(math.hypot(float(vector[0]), float(vector[1])))
        velocity = float(math.hypot(float(vector[2]), float(vector[3])))
        return min(1.0, max(0.0, position / 4.0 + velocity / 2.0))

    def _normalization_path(self) -> Path:
        configured = Path(str(self.data_config["normalization_stats"]))
        if configured.exists():
            return configured
        candidate = Path("outputs/noise_modeling/normalization_stats.json")
        if candidate.exists():
            return candidate
        return configured

    def _feature_array(self, values: Dict[str, float]) -> np.ndarray:
        columns = self.normalizers[self._feature_norm_key()].mean.shape[0]
        if self.feature_set == "v1":
            names = [
                "dx_obs",
                "dy_obs",
                "dvx_obs",
                "dvy_obs",
                "distance_obs",
                "sin_bearing_obs",
                "cos_bearing_obs",
            ]
        elif self.feature_set == "v2":
            names = [
                "dx_obs",
                "dy_obs",
                "dvx_obs",
                "dvy_obs",
                "distance_obs",
                "sin_bearing_obs",
                "cos_bearing_obs",
                "relative_speed_obs",
                "closing_rate",
                "ego_vx",
                "ego_vy",
                "ego_speed",
                "vx_obs",
                "vy_obs",
                "neighbor_speed_obs",
                "step_normalized",
                "time_sec_normalized",
                "sigma_v1",
                "sigma_v2",
                "sigma_v3",
            ]
        else:
            names = [
                "dx_obs",
                "dy_obs",
                "dvx_obs",
                "dvy_obs",
                "distance_obs",
                "sin_bearing_obs",
                "cos_bearing_obs",
                "relative_speed_obs",
                "closing_rate",
                "ego_speed",
                "neighbor_speed_obs",
                "delta_time",
            ]

        if len(names) != columns:
            raise ValueError(
                f"Feature dimension mismatch for {self.checkpoint_path}: "
                f"checkpoint expects {columns}, runtime built {len(names)}"
            )
        return np.asarray([float(values.get(name, 0.0)) for name in names], dtype=np.float32)

    def _sequence_input(self, key: tuple[str, int], raw_feature: np.ndarray) -> np.ndarray:
        history = self._history_by_key.setdefault(key, deque(maxlen=self.history))
        history.append(raw_feature)
        frames = list(history)
        if len(frames) < self.history:
            frames = [frames[0]] * (self.history - len(frames)) + frames
        sequence = np.stack(frames, axis=0)[None, :, :]
        return self._normalize(sequence, self._feature_norm_key())

    def _normalize(self, values: np.ndarray, norm_key: str) -> np.ndarray:
        if not self.normalize_features:
            return values
        norm = self.normalizers[norm_key]
        if values.ndim == 3:
            return (values - norm.mean[None, None, :]) / norm.std[None, None, :]
        return (values - norm.mean[None, :]) / norm.std[None, :]

    def _feature_norm_key(self) -> str:
        return f"features_{self.feature_set}"
