from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np


class Normalizer:
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.where(std.astype(np.float32) < 1.0e-8, 1.0, std.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_scale(self, values: np.ndarray) -> np.ndarray:
        return values * self.std


def load_normalizers(path: str | Path) -> Dict[str, Normalizer]:
    stats = json.loads(Path(path).read_text())
    result = {}
    for key in ("features_v1", "features_v2", "features_v3", "target_rel"):
        item = stats[key]
        result[key] = Normalizer(
            mean=np.asarray(item["mean"], dtype=np.float32),
            std=np.asarray(item["std"], dtype=np.float32),
        )
    return result

