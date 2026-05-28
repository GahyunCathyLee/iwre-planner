from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PhysicsConfig:
    w_dist: float = 0.4
    w_motion: float = 0.3
    w_angle: float = 0.3
    max_range: float = 100.0
    max_speed: float = 30.0
    occluded_min_sigma: float = 0.8


@dataclass
class KalmanConfig:
    dt: float = 0.05
    q_pos: float = 0.5
    q_vel: float = 1.0
    r_pos: float = 1.0
    p0_pos: float = 10.0
    p0_vel: float = 10.0
    sigma_scale: float = 5.0


@dataclass
class FusionConfig:
    alpha_phys: float = 0.5
    alpha_kf: float = 0.5


@dataclass
class LSTMConfig:
    input_size: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    head_hidden_size: int = 32
    output_mode: str = "gaussian"


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1.0e-3
    min_learning_rate: float = 5.0e-4
    weight_decay: float = 1.0e-4
    scheduler_patience: int = 5
    early_stop_patience: int = 10
