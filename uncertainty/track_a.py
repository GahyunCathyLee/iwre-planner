from __future__ import annotations

import numpy as np

from uncertainty.config import FusionConfig, KalmanConfig, PhysicsConfig
from uncertainty.kalman import KalmanTrackerBank
from uncertainty.physics import compute_sigma_phys


class TrackAEstimator:
    def __init__(
        self,
        physics_config: PhysicsConfig | None = None,
        kalman_config: KalmanConfig | None = None,
        fusion_config: FusionConfig | None = None,
    ):
        self.physics_config = physics_config or PhysicsConfig()
        self.kalman_bank = KalmanTrackerBank(kalman_config or KalmanConfig())
        self.fusion_config = fusion_config or FusionConfig()

    def step(self, vehicle_id: str, dx: float, dy: float, dvx: float, dvy: float, visible: bool = True) -> dict[str, float]:
        sigma_phys = compute_sigma_phys(dx, dy, dvx, dvy, visible, self.physics_config)
        sigma_kf = self.kalman_bank.step_vehicle(str(vehicle_id), dx, dy, dvx, dvy, visible)
        sigma_a = (
            self.fusion_config.alpha_phys * sigma_phys
            + self.fusion_config.alpha_kf * sigma_kf
        )
        return {
            "sigma_phys": float(np.clip(sigma_phys, 0.0, 1.0)),
            "sigma_kf": float(np.clip(sigma_kf, 0.0, 1.0)),
            "sigma_a": float(np.clip(sigma_a, 0.0, 1.0)),
        }
