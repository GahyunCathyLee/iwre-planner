from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from uncertainty.config import LSTMConfig
from uncertainty.features import build_lstm_features
from uncertainty.lstm_model import LSTMUncertaintyEstimator


class TrackBLSTMRuntime:
    def __init__(self, checkpoint_path: str, history: int = 10, device: str | None = None):
        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(str(Path(checkpoint_path)), map_location=self.device)
        self.config = LSTMConfig(**ckpt.get("lstm_config", {}))
        self.history_length = int(ckpt.get("history_length", history))
        self.feature_mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
        self.feature_std = np.maximum(np.asarray(ckpt["feature_std"], dtype=np.float32), 1.0e-6)
        self.model = LSTMUncertaintyEstimator(self.config).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self._history_by_key: dict[tuple[str, int], deque[dict[str, float]]] = {}

    def predict(self, key: tuple[str, int], features: dict[str, float]) -> float:
        history = self._history_by_key.setdefault(key, deque(maxlen=self.history_length))
        history.append(features)
        frames = list(history)
        if len(frames) < self.history_length:
            frames = [frames[0]] * (self.history_length - len(frames)) + frames
        matrix = build_lstm_features(frames)
        matrix = (matrix - self.feature_mean[None, :]) / self.feature_std[None, :]
        tensor = self.torch.from_numpy(matrix[None, :, :].astype(np.float32)).to(self.device)
        with self.torch.no_grad():
            output = self.model(tensor)
        if "mu" in output:
            value = output["mu"][0]
        else:
            value = output["sigma"][0]
        return float(self.torch.clamp(value, 0.0, 1.0).detach().cpu().item())
