from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from uncertainty.features import build_lstm_features


def _to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "" or value is None:
        return default
    return float(value)


def discover_neighbor_logs(log_root: str | Path, pattern: str = "*_B1*_nbr.csv") -> list[Path]:
    root = Path(log_root)
    return sorted(root.glob(pattern))


class UncertaintyCSVDataset(Dataset):
    def __init__(
        self,
        csv_paths: Iterable[str | Path],
        history_length: int = 10,
        dt: float = 0.05,
        normalize: bool = True,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ):
        self.history_length = int(history_length)
        self.dt = float(dt)
        self.samples: list[tuple[np.ndarray, float]] = []
        for path in csv_paths:
            self._load_path(Path(path))
        if not self.samples:
            raise ValueError("No Track B samples were built. Check CSV paths and sigma_label columns.")
        features = np.stack([sample[0] for sample in self.samples], axis=0)
        if feature_mean is None:
            feature_mean = features.reshape(-1, features.shape[-1]).mean(axis=0)
        if feature_std is None:
            feature_std = features.reshape(-1, features.shape[-1]).std(axis=0)
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = np.maximum(feature_std.astype(np.float32), 1.0e-6)
        if normalize:
            self.samples = [
                ((features_i - self.feature_mean[None, :]) / self.feature_std[None, :], target)
                for features_i, target in self.samples
            ]

    def _load_path(self, path: Path) -> None:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))
        by_vehicle: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            if row.get("sigma_label", "") == "":
                continue
            vehicle_id = row.get("neighbor_id") or row.get("vehicle_id") or "unknown"
            by_vehicle.setdefault(vehicle_id, []).append(row)
        for vehicle_rows in by_vehicle.values():
            vehicle_rows.sort(key=lambda row: (_to_float(row, "step"), _to_float(row, "time_sec")))
            history: list[dict[str, float]] = []
            prev_time = None
            for row in vehicle_rows:
                ego_vx = _to_float(row, "ego_vx")
                ego_vy = _to_float(row, "ego_vy")
                vx_obs = _to_float(row, "vx_obs")
                vy_obs = _to_float(row, "vy_obs")
                time_sec = _to_float(row, "time_sec")
                frame = {
                    "dx": _to_float(row, "dx_obs"),
                    "dy": _to_float(row, "dy_obs"),
                    "dvx": vx_obs - ego_vx,
                    "dvy": vy_obs - ego_vy,
                    "dyaw": 0.0,
                }
                history.append(frame)
                tail = history[-self.history_length :]
                if len(tail) < self.history_length:
                    tail = [tail[0]] * (self.history_length - len(tail)) + tail
                dt = self.dt if prev_time is None else max(time_sec - prev_time, 1.0e-6)
                self.samples.append((build_lstm_features(tail, dt=dt), _to_float(row, "sigma_label")))
                prev_time = time_sec

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        features, target = self.samples[index]
        return {
            "features": torch.from_numpy(features.astype(np.float32)),
            "target": torch.tensor(float(target), dtype=torch.float32),
        }


def split_paths(paths: list[Path], val_fraction: float = 0.15, seed: int = 42) -> tuple[list[Path], list[Path]]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    return shuffled[val_count:], shuffled[:val_count]
