from __future__ import annotations

import glob
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset


class InteractionGRIPDataset(Dataset):
    """NPZ shard dataset for ego + 8-neighbor GRIP-style tensors."""

    def __init__(self, data_root: str, split: str, limit_shards: int | None = None):
        self.data_root = Path(data_root)
        paths = sorted(glob.glob(str(self.data_root / split / "*.npz")))
        if limit_shards is not None:
            paths = paths[:limit_shards]
        if not paths:
            raise FileNotFoundError(f"No npz shards found under {self.data_root / split}")
        arrays = self._load(paths)
        self.x = arrays["x"]
        self.adj = arrays["adj"]
        self.target = arrays["target"]
        self.valid_mask = arrays["valid_mask"]
        self.alpha_heuristic = arrays["alpha_heuristic"]
        self.alpha_teacher = arrays.get("alpha_teacher", self.alpha_heuristic)

    @staticmethod
    def _load(paths: Iterable[str]) -> Dict[str, np.ndarray]:
        merged: Dict[str, List[np.ndarray]] = {}
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                for key in ("x", "adj", "target", "valid_mask", "alpha_heuristic", "alpha_teacher"):
                    if key in data:
                        merged.setdefault(key, []).append(data[key].astype(np.float32))
        return {key: np.concatenate(values, axis=0) for key, values in merged.items()}

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "x": torch.from_numpy(self.x[index]),
            "adj": torch.from_numpy(self.adj[index]),
            "target": torch.from_numpy(self.target[index]),
            "valid_mask": torch.from_numpy(self.valid_mask[index]),
            "alpha": torch.from_numpy(self.alpha_teacher[index]),
            "alpha_heuristic": torch.from_numpy(self.alpha_heuristic[index]),
        }

