from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


SLOT_NAMES = [
    "preceding",
    "following",
    "leftPreceding",
    "leftAlongside",
    "leftFollowing",
    "rightPreceding",
    "rightAlongside",
    "rightFollowing",
]
SLOT_TO_INDEX = {name: index for index, name in enumerate(SLOT_NAMES)}


class GRIPAlphaRuntime:
    """Rolling-buffer online alpha inference for the GRIP alpha model.

    The runtime accepts ego-observable neighbor states only. Until enough
    history is available, callers can keep using the existing heuristic alpha.
    """

    def __init__(self, checkpoint_path: str, history: int = 6, device: str | None = None):
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        import torch
        from interaction_modeling.models.grip_alpha import GRIPAlphaEstimator

        self.torch = torch
        self.history = history
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model = GRIPAlphaEstimator(
            in_channels=int(ckpt["in_channels"]),
            hidden_channels=int(ckpt.get("hidden_channels", 64)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.buffer: deque[Dict[str, object]] = deque(maxlen=history)

    def update(self, ego_state: Dict[str, float], neighbors_by_slot: Dict[str, Dict[str, float]]) -> None:
        self.buffer.append({"ego": dict(ego_state), "neighbors": dict(neighbors_by_slot)})

    def ready(self) -> bool:
        return len(self.buffer) >= self.history

    def predict(self) -> Dict[str, float]:
        if not self.ready():
            return {}
        x, adj, valid_mask = self._build_tensor()
        with self.torch.no_grad():
            alpha = self.model(
                self.torch.from_numpy(x[None]).to(self.device),
                self.torch.from_numpy(adj[None]).to(self.device),
                self.torch.from_numpy(valid_mask[None]).to(self.device),
            )[0].detach().cpu().numpy()
        return {slot: float(alpha[index]) for slot, index in SLOT_TO_INDEX.items() if valid_mask[index + 1] > 0.0}

    def _build_tensor(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.zeros((9, self.history, 9), dtype=np.float32)
        valid_mask = np.zeros(9, dtype=np.float32)
        for hi, frame in enumerate(self.buffer):
            ego = frame["ego"]
            ego_vx = float(ego.get("ego_vx", 0.0))
            ego_vy = float(ego.get("ego_vy", 0.0))
            x[2, hi, 0] = ego_vx
            x[3, hi, 0] = ego_vy
            x[7, hi, 0] = 1.0
            x[8, hi, 0] = 1.0
            valid_mask[0] = 1.0
            neighbors = frame["neighbors"]
            for slot, neighbor in neighbors.items():
                if slot not in SLOT_TO_INDEX:
                    continue
                slot_index = SLOT_TO_INDEX[slot]
                node = slot_index + 1
                dx = float(neighbor.get("dx_obs", 0.0))
                dy = float(neighbor.get("dy_obs", 0.0))
                dvx = float(neighbor.get("vx_obs", 0.0)) - ego_vx
                dvy = float(neighbor.get("vy_obs", 0.0)) - ego_vy
                distance = math.hypot(dx, dy)
                closing = (dx * dvx + dy * dvy) / max(distance, 1.0e-6)
                x[:, hi, node] = [
                    dx,
                    dy,
                    dvx,
                    dvy,
                    distance,
                    closing,
                    slot_index / max(len(SLOT_NAMES) - 1, 1),
                    1.0,
                    0.0,
                ]
                if hi == self.history - 1:
                    valid_mask[node] = 1.0

        adj = np.eye(9, dtype=np.float32)
        for node in np.flatnonzero(valid_mask > 0.0):
            adj[0, node] = 1.0
            adj[node, 0] = 1.0
        return x, adj, valid_mask
