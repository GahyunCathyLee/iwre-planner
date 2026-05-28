from __future__ import annotations

import math

import numpy as np

from uncertainty.config import PhysicsConfig


def compute_sigma_phys(
    dx: float,
    dy: float,
    dvx: float,
    dvy: float,
    visible: bool,
    config: PhysicsConfig | None = None,
) -> float:
    cfg = config or PhysicsConfig()
    dist = math.hypot(dx, dy)
    v_rel = math.hypot(dvx, dvy)
    theta = math.atan2(dy, dx)
    sigma = (
        cfg.w_dist * np.clip(dist / max(cfg.max_range, 1.0e-6), 0.0, 1.0)
        + cfg.w_motion * np.clip(v_rel / max(cfg.max_speed, 1.0e-6), 0.0, 1.0)
        + cfg.w_angle * abs(math.sin(theta))
    )
    if not visible:
        sigma = max(float(sigma), cfg.occluded_min_sigma)
    return float(np.clip(sigma, 0.0, 1.0))
