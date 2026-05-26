from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset

from noise_modeling.data.normalization import load_normalizers


FEATURE_KEY_BY_SET = {
    "v1": "features_v1",
    "v2": "features_v2",
    "v3": "features_v3",
}


class NoiseNPZDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        feature_set: str,
        normalization_path: str,
        use_sequence: bool = False,
        normalize_features: bool = True,
        normalize_targets: bool = True,
        limit_shards: int | None = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.feature_set = feature_set
        self.use_sequence = use_sequence
        self.normalize_targets = normalize_targets
        self.normalizers = load_normalizers(normalization_path)

        paths = sorted(glob.glob(str(self.data_root / split / "*.npz")))
        if limit_shards is not None:
            paths = paths[:limit_shards]
        if not paths:
            raise FileNotFoundError(f"No npz shards found under {self.data_root / split}")

        feature_key = "sequence_features_v3" if use_sequence else FEATURE_KEY_BY_SET[feature_set]
        feature_norm_key = "features_v3" if use_sequence else FEATURE_KEY_BY_SET[feature_set]
        arrays = self._load_arrays(paths, feature_key)
        self.features = arrays["features"]
        self.target = arrays["target_rel"]
        self.obs_state = arrays["obs_state_rel"]
        self.true_state = arrays["true_state_rel"]

        if normalize_features:
            norm = self.normalizers[feature_norm_key]
            if self.features.ndim == 3:
                self.features = (self.features - norm.mean[None, None, :]) / norm.std[None, None, :]
            else:
                self.features = norm.transform(self.features)

        if normalize_targets:
            self.target = self.normalizers["target_rel"].transform(self.target)

    @staticmethod
    def _load_arrays(paths: Iterable[str], feature_key: str) -> Dict[str, np.ndarray]:
        features: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        obs_states: List[np.ndarray] = []
        true_states: List[np.ndarray] = []
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                features.append(data[feature_key].astype(np.float32))
                targets.append(data["target_rel"].astype(np.float32))
                obs_states.append(data["obs_state_rel"].astype(np.float32))
                true_states.append(data["true_state_rel"].astype(np.float32))
        return {
            "features": np.concatenate(features, axis=0),
            "target_rel": np.concatenate(targets, axis=0),
            "obs_state_rel": np.concatenate(obs_states, axis=0),
            "true_state_rel": np.concatenate(true_states, axis=0),
        }

    @property
    def target_std(self) -> torch.Tensor:
        return torch.as_tensor(self.normalizers["target_rel"].std, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "features": torch.from_numpy(self.features[index]),
            "target": torch.from_numpy(self.target[index]),
            "obs_state": torch.from_numpy(self.obs_state[index]),
            "true_state": torch.from_numpy(self.true_state[index]),
        }

