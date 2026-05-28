from __future__ import annotations

import numpy as np

from uncertainty.config import KalmanConfig


class VehicleKalmanFilter:
    def __init__(self, config: KalmanConfig | None = None):
        self.config = config or KalmanConfig()
        self.x: np.ndarray | None = None
        self.P: np.ndarray | None = None

    def initialize(self, px: float, py: float, vx: float = 0.0, vy: float = 0.0) -> None:
        cfg = self.config
        self.x = np.asarray([px, py, vx, vy], dtype=np.float64)
        self.P = np.diag([cfg.p0_pos, cfg.p0_pos, cfg.p0_vel, cfg.p0_vel]).astype(np.float64)

    def predict(self) -> None:
        if self.x is None or self.P is None:
            raise RuntimeError("Kalman filter must be initialized before predict().")
        cfg = self.config
        dt = cfg.dt
        F = np.asarray(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        Q = np.diag([cfg.q_pos, cfg.q_pos, cfg.q_vel, cfg.q_vel]).astype(np.float64)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    def update(self, px_obs: float, py_obs: float) -> None:
        if self.x is None or self.P is None:
            raise RuntimeError("Kalman filter must be initialized before update().")
        cfg = self.config
        H = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        R = np.diag([cfg.r_pos, cfg.r_pos]).astype(np.float64)
        z = np.asarray([px_obs, py_obs], dtype=np.float64)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        I = np.eye(4, dtype=np.float64)
        self.x = self.x + K @ y
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

    def step(self, px_obs: float, py_obs: float, vx_obs: float, vy_obs: float, visible: bool) -> float:
        if self.x is None:
            self.initialize(px_obs, py_obs, vx_obs, vy_obs)
            return self.sigma()
        self.predict()
        if visible:
            self.update(px_obs, py_obs)
            if self.x is not None:
                self.x[2] = vx_obs
                self.x[3] = vy_obs
        return self.sigma()

    def sigma(self) -> float:
        if self.P is None:
            return 1.0
        sigma = np.sqrt(max(float(self.P[0, 0] + self.P[1, 1]), 0.0)) / max(self.config.sigma_scale, 1.0e-6)
        return float(np.clip(sigma, 0.0, 1.0))


class KalmanTrackerBank:
    def __init__(self, config: KalmanConfig | None = None):
        self.config = config or KalmanConfig()
        self.filters: dict[str, VehicleKalmanFilter] = {}

    def step_vehicle(self, vehicle_id: str, px: float, py: float, vx: float, vy: float, visible: bool) -> float:
        filt = self.filters.setdefault(str(vehicle_id), VehicleKalmanFilter(self.config))
        return filt.step(px, py, vx, vy, visible)
