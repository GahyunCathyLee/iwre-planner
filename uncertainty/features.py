from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def build_lstm_features(history: Sequence[dict], dt: float = 0.05, eps: float = 1.0e-6) -> np.ndarray:
    rows = []
    dx_values = []
    dy_values = []
    prev_dvx = None
    prev_dvy = None
    for item in history:
        dx = float(item.get("dx", item.get("dx_obs", 0.0)))
        dy = float(item.get("dy", item.get("dy_obs", 0.0)))
        dvx = float(item.get("dvx", item.get("dvx_obs", 0.0)))
        dvy = float(item.get("dvy", item.get("dvy_obs", 0.0)))
        dyaw = float(item.get("dyaw", 0.0))
        dist = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        closing_speed = -(dx * dvx + dy * dvy) / max(dist, eps)
        ttc = dist / closing_speed if closing_speed > eps else 100.0
        ttc = min(ttc, 100.0)
        dx_values.append(dx)
        dy_values.append(dy)
        tail_x = dx_values[-5:]
        tail_y = dy_values[-5:]
        std_pos5 = math.hypot(float(np.std(tail_x)), float(np.std(tail_y)))
        if prev_dvx is None or prev_dvy is None:
            accel = 0.0
        else:
            accel = math.hypot((dvx - prev_dvx) / max(dt, eps), (dvy - prev_dvy) / max(dt, eps))
        rows.append([dx, dy, dvx, dvy, dist, dyaw, ttc, theta, std_pos5, accel])
        prev_dvx = dvx
        prev_dvy = dvy
    return np.asarray(rows, dtype=np.float32)
