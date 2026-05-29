# Codex Implementation Prompt: Module 1 — Uncertainty Estimation (`σ_i`)

## Goal

Implement **Module 1: Uncertainty Estimation (`σ_i`)** for surrounding vehicles.

The module should support two tracks:

1. **Track A — Rule-based baseline**
   - Physics Estimator: instantaneous observability uncertainty.
   - Kalman Filter Estimator: temporal tracking/familiarity uncertainty.
   - Final output: weighted fusion of both.

2. **Track B — LSTM-based learned estimator**
   - Input: recent 10-step sequence of 10 features per vehicle.
   - Output mode:
     - Scalar mode: one predicted uncertainty value.
     - Gaussian mode: `(μ, log σ²)`, currently preferred.

The final implementation should be clean, modular, testable, and usable from local scripts or training code.

---

# 1. Conceptual Definition

For each surrounding vehicle `i`, estimate an uncertainty value:

```text
σ_i ∈ [0, 1]
```

Interpretation:

- Low `σ_i` means the vehicle is well observed, stable, and predictable.
- High `σ_i` means the vehicle is far, fast, lateral, newly observed, occluded, unstable, or otherwise difficult to estimate.

---

# 2. Track A — Rule-Based Uncertainty Estimation

Track A is an interpretable baseline composed of two sub-models.

```text
σ_A = 0.5 * σ_phys + 0.5 * σ_kf
```

Where:

- `σ_phys`: instantaneous physics/observability uncertainty.
- `σ_kf`: temporal Kalman Filter uncertainty.
- `σ_A`: final Track A uncertainty.

All outputs must be clipped to `[0, 1]`.

---

## 2.1 Sub-model 1: Physics Estimator

### Idea

A vehicle is harder to measure if it is:

- Far away.
- Moving quickly relative to ego.
- Located more laterally relative to ego.
- Occluded or not visible.

This estimator is stateless. It is recomputed at every step.

---

## Inputs

For each vehicle at a single timestep:

```python
dx: float       # relative x position, meters
dy: float       # relative y position, meters
dvx: float      # relative x velocity, m/s
dvy: float      # relative y velocity, m/s
visible: bool   # whether the vehicle is currently visible / observed
```

Derived values:

```python
dist = sqrt(dx**2 + dy**2)
v_rel = sqrt(dvx**2 + dvy**2)
theta = atan2(dy, dx)
```

---

## Formula

```text
σ_phys = w_dist   * clip(dist / max_range, 0, 1)
       + w_motion * clip(v_rel / max_speed, 0, 1)
       + w_angle  * abs(sin(theta))
```

Default parameters:

```python
w_dist = 0.4
w_motion = 0.3
w_angle = 0.3

max_range = 100.0  # meters
max_speed = 30.0   # m/s
occluded_min_sigma = 0.8
```

Weights sum to `1.0`.

If the vehicle is occluded:

```python
if visible is False:
    σ_phys = max(σ_phys, occluded_min_sigma)
```

Finally:

```python
σ_phys = clip(σ_phys, 0.0, 1.0)
```

---

## Expected Behavior

| Situation | Expected `σ_phys` |
|---|---|
| Close, slow, longitudinal vehicle | Low |
| Far vehicle | Higher |
| Fast relative motion | Higher |
| Lateral vehicle, `abs(sin(theta)) ≈ 1` | Higher |
| Occluded vehicle | At least `0.8` |

---

## 2.2 Sub-model 2: Kalman Filter Estimator

### Idea

The longer we observe the same vehicle, the more stable its state estimate becomes.

When a vehicle is:

- Newly detected: uncertainty is high.
- Continuously visible: uncertainty decreases.
- Occluded: only prediction is possible, so uncertainty increases again.

---

## State

Track each vehicle independently.

State vector:

```text
x = [px, py, vx, vy]^T
```

Where:

- `px`, `py`: relative position.
- `vx`, `vy`: relative velocity.

Covariance matrix:

```text
P ∈ R^(4x4)
```

`P` represents "how much we do not know."

---

## Kalman Filter Cycle

Each timestep:

1. **Predict**
   - Predict where the vehicle will be next based on current velocity.
   - Always run predict.

2. **Update**
   - If the vehicle is visible, update using observed position.
   - If the vehicle is occluded, skip update.

---

## Suggested Constant-Velocity Model

Use timestep `dt`, default:

```python
dt = 0.05  # seconds if 10 steps = 0.5 seconds
```

State transition:

```text
F = [[1, 0, dt, 0 ],
     [0, 1, 0,  dt],
     [0, 0, 1,  0 ],
     [0, 0, 0,  1 ]]
```

Observation matrix:

```text
H = [[1, 0, 0, 0],
     [0, 1, 0, 0]]
```

Observation:

```text
z = [px_obs, py_obs]^T
```

---

## Default Kalman Config

```python
q_pos = 0.5   # process noise for position
q_vel = 1.0   # process noise for velocity
r_pos = 1.0   # measurement noise for observed position

p0_pos = 10.0 # initial position uncertainty for new vehicle
p0_vel = 10.0 # initial velocity uncertainty for new vehicle
```

Suggested matrices:

```python
Q = diag([q_pos, q_pos, q_vel, q_vel])
R = diag([r_pos, r_pos])
P0 = diag([p0_pos, p0_pos, p0_vel, p0_vel])
```

---

## Initialization

When a new vehicle ID appears:

```python
x0 = [px_obs, py_obs, vx_obs, vy_obs]
P0 = diag([p0_pos, p0_pos, p0_vel, p0_vel])
```

The first `σ_kf` should be high, approximately close to `1.0`.

---

## Output

Compute Kalman uncertainty from the position covariance:

```text
σ_kf = sqrt(P[0, 0] + P[1, 1]) / 5.0
```

Then:

```python
σ_kf = clip(σ_kf, 0.0, 1.0)
```

---

## Expected Behavior

| Situation | Expected `σ_kf` |
|---|---|
| New vehicle appears | Near `1.0` |
| Vehicle is observed consistently | Gradually decreases |
| Vehicle becomes occluded | Increases again |
| Vehicle reappears | Update reduces uncertainty again |

---

## 2.3 Track A Fusion

Final Track A output:

```text
σ_A = alpha_phys * σ_phys + alpha_kf * σ_kf
```

Default:

```python
alpha_phys = 0.5
alpha_kf = 0.5
```

Clip:

```python
σ_A = clip(σ_A, 0.0, 1.0)
```

---

# 3. Track B — LSTM-Based Uncertainty Estimation

Track B learns uncertainty from recent vehicle motion history.

---

## 3.1 Input Shape

The LSTM receives a sequence:

```text
X ∈ R^(B × T × F)
```

Default values:

```python
B = 32   # batch size
T = 10   # recent 10 timesteps
F = 10   # feature dimension
```

One sample input is:

```text
(10, 10)
```

Meaning:

```text
10 timesteps × 10 features per timestep
```

If 10 steps correspond to 0.5 seconds, then:

```python
dt = 0.05
```

Dataset size:

```text
~730,000 windows
```

With batch size 32:

```text
~22,000 to 23,000 iterations per epoch
```

---

## 3.2 Input Features

Each timestep should contain the following 10 features:

| Index | Feature | Meaning | Category |
|---:|---|---|---|
| 0 | `dx` | Relative x position | Observability |
| 1 | `dy` | Relative y position | Observability |
| 2 | `dvx` | Relative x velocity | Observability |
| 3 | `dvy` | Relative y velocity | Observability |
| 4 | `dist` | Euclidean distance | Observability |
| 5 | `dyaw` | Relative heading angle | Observability |
| 6 | `ttc` | Time-to-collision: `dist / relative_closing_speed` | Risk-relevant |
| 7 | `theta` | Bearing angle: `atan2(dy, dx)` | Observability |
| 8 | `std_pos5` | Standard deviation of position over last 5 steps | Predictability |
| 9 | `accel` | Acceleration magnitude | Predictability |

---

## Feature Engineering Details

### `dist`

```python
dist = sqrt(dx**2 + dy**2)
```

### `theta`

```python
theta = atan2(dy, dx)
```

### `ttc`

Use a numerically stable definition.

Suggested simple version:

```python
relative_speed = sqrt(dvx**2 + dvy**2)
ttc = dist / max(relative_speed, eps)
```

Alternative collision-specific version:

```python
closing_speed = -dot([dx, dy], [dvx, dvy]) / max(dist, eps)
if closing_speed > eps:
    ttc = dist / closing_speed
else:
    ttc = large_value
```

Clip or normalize TTC as needed.

### `std_pos5`

For the last 5 positions:

```python
std_x = std(dx over last 5 steps)
std_y = std(dy over last 5 steps)
std_pos5 = sqrt(std_x**2 + std_y**2)
```

For the first few timesteps, either:

- Use available history only, or
- Pad previous values, or
- Set to 0 until enough history exists.

### `accel`

If velocity history is available:

```python
accel_x = (dvx_t - dvx_{t-1}) / dt
accel_y = (dvy_t - dvy_{t-1}) / dt
accel = sqrt(accel_x**2 + accel_y**2)
```

For the first timestep, set acceleration to `0`.

---

## 3.3 Model Architecture

Use PyTorch.

Recommended architecture:

```text
Input: X ∈ R^(B × 10 × 10)

LSTM:
  input_size = 10
  hidden_size = 64
  num_layers = 2
  dropout = 0.1
  batch_first = True

Use the hidden state at the last timestep:
  h_T ∈ R^(B × 64)

Head:
  Linear(64 → 32)
  ReLU
  Dropout(0.1)
  Linear(32 → output_dim)
```

Where:

- `output_dim = 1` for scalar mode.
- `output_dim = 2` for Gaussian mode.

---

## 3.4 Output Modes

### Mode A: Scalar Mode

The model outputs:

```text
σ_pred
```

Use when a simple uncertainty estimate is enough.

Recommended post-processing:

```python
sigma_pred = sigmoid(raw_output)
```

Prefer `sigmoid` during training for differentiability.

---

### Mode B: Gaussian Mode — Preferred

The model outputs:

```text
(μ, log σ²)
```

Interpretation:

- `μ`: predicted uncertainty target.
- `log σ²`: model's predicted variance/confidence for this prediction.

Recommended output handling:

```python
mu_raw, logvar = output[:, 0], output[:, 1]
mu = sigmoid(mu_raw)  # keep predicted uncertainty in [0, 1]
logvar = clamp(logvar, min=-10, max=5)
```

`logvar` should not be passed through sigmoid. It represents log variance.

---

## 3.5 Loss Functions

### Scalar Mode Loss

Use MSE or Smooth L1:

```python
loss = mse_loss(sigma_pred, target_sigma)
```

### Gaussian Mode Loss

Use Gaussian negative log-likelihood.

For target `y`, predicted mean `μ`, predicted log variance `logvar`:

```text
NLL = 0.5 * (exp(-logvar) * (y - μ)^2 + logvar)
```

Implementation:

```python
def gaussian_nll_loss(mu, logvar, target):
    logvar = torch.clamp(logvar, min=-10.0, max=5.0)
    return 0.5 * (torch.exp(-logvar) * (target - mu) ** 2 + logvar).mean()
```

This lets the model output larger variance for genuinely hard samples, improving calibration.

---

# 4. Training Configuration

Use the following defaults:

```python
loss = "Gaussian NLL"  # for Gaussian mode
optimizer = "AdamW"
learning_rate = 1e-3
min_learning_rate = 5e-4
weight_decay = 1e-4
batch_size = 32
epochs = 20
scheduler = "ReduceLROnPlateau"
scheduler_patience = 5
early_stop_patience = 10
```

Suggested PyTorch setup:

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5,
)
```

Early stopping:

```text
Stop if validation loss does not improve for 10 epochs.
```

Expected training time:

- CPU: several minutes per epoch.
- GPU: tens of seconds per epoch.

---

# 5. Suggested File Structure

Please implement with a clean modular structure like:

```text
uncertainty/
  __init__.py
  config.py
  physics.py
  kalman.py
  track_a.py
  features.py
  lstm_model.py
  losses.py
  train.py
  dataset.py
  evaluate.py
examples/
  demo_track_a.py
  demo_lstm_forward.py
tests/
  test_physics.py
  test_kalman.py
  test_track_a.py
  test_lstm_shapes.py
```

---

# 6. Suggested Dataclasses

## `PhysicsConfig`

```python
@dataclass
class PhysicsConfig:
    w_dist: float = 0.4
    w_motion: float = 0.3
    w_angle: float = 0.3
    max_range: float = 100.0
    max_speed: float = 30.0
    occluded_min_sigma: float = 0.8
```

## `KalmanConfig`

```python
@dataclass
class KalmanConfig:
    dt: float = 0.05
    q_pos: float = 0.5
    q_vel: float = 1.0
    r_pos: float = 1.0
    p0_pos: float = 10.0
    p0_vel: float = 10.0
    sigma_scale: float = 5.0
```

## `FusionConfig`

```python
@dataclass
class FusionConfig:
    alpha_phys: float = 0.5
    alpha_kf: float = 0.5
```

## `LSTMConfig`

```python
@dataclass
class LSTMConfig:
    input_size: int = 10
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    head_hidden_size: int = 32
    output_mode: str = "gaussian"  # "scalar" or "gaussian"
```

## `TrainConfig`

```python
@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_patience: int = 5
    early_stop_patience: int = 10
```

---

# 7. Required Functions and Classes

## Physics

```python
def compute_sigma_phys(
    dx: float,
    dy: float,
    dvx: float,
    dvy: float,
    visible: bool,
    config: PhysicsConfig,
) -> float:
    ...
```

Also support vectorized NumPy input if convenient.

---

## Kalman

```python
class VehicleKalmanFilter:
    def __init__(self, config: KalmanConfig):
        ...

    def initialize(
        self,
        px: float,
        py: float,
        vx: float = 0.0,
        vy: float = 0.0,
    ) -> None:
        ...

    def predict(self) -> None:
        ...

    def update(self, px_obs: float, py_obs: float) -> None:
        ...

    def step(
        self,
        px_obs: float,
        py_obs: float,
        vx_obs: float,
        vy_obs: float,
        visible: bool,
    ) -> float:
        ...
        # returns sigma_kf

    def sigma(self) -> float:
        ...
```

A manager for multiple vehicles is also useful:

```python
class KalmanTrackerBank:
    def __init__(self, config: KalmanConfig):
        self.filters: dict[str, VehicleKalmanFilter] = {}

    def step_vehicle(
        self,
        vehicle_id: str,
        px: float,
        py: float,
        vx: float,
        vy: float,
        visible: bool,
    ) -> float:
        ...
```

---

## Track A

```python
class TrackAEstimator:
    def __init__(
        self,
        physics_config: PhysicsConfig,
        kalman_config: KalmanConfig,
        fusion_config: FusionConfig,
    ):
        ...

    def step(
        self,
        vehicle_id: str,
        dx: float,
        dy: float,
        dvx: float,
        dvy: float,
        visible: bool,
    ) -> dict:
        ...
```

Return dictionary:

```python
{
    "sigma_phys": float,
    "sigma_kf": float,
    "sigma_a": float,
}
```

---

## Feature Builder

```python
def build_lstm_features(
    history: list[dict],
    dt: float = 0.05,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Build a (T, 10) feature matrix for one vehicle.

    Each history element should contain:
      dx, dy, dvx, dvy, dyaw

    Optional:
      visible, timestamp
    """
```

Output order must be:

```text
[dx, dy, dvx, dvy, dist, dyaw, ttc, theta, std_pos5, accel]
```

---

## LSTM Model

```python
class LSTMUncertaintyEstimator(nn.Module):
    def __init__(self, config: LSTMConfig):
        ...

    def forward(self, x: torch.Tensor) -> dict:
        """
        x shape: (B, T, F)

        scalar mode return:
          {"sigma": sigma_pred}

        gaussian mode return:
          {"mu": mu, "logvar": logvar}
        """
```

Expected shapes:

```python
x.shape == (B, 10, 10)
mu.shape == (B,)
logvar.shape == (B,)
sigma.shape == (B,)
```

---

# 8. Tests to Implement

## Physics tests

1. Close and slow visible vehicle should produce low uncertainty.
2. Far vehicle should produce higher uncertainty.
3. Fast vehicle should produce higher uncertainty.
4. Lateral vehicle should produce higher uncertainty.
5. Occluded vehicle should produce uncertainty at least `0.8`.
6. Output must always be in `[0, 1]`.

---

## Kalman tests

1. New vehicle uncertainty should start high.
2. Repeated visible updates should reduce `σ_kf`.
3. Occluded steps should increase `σ_kf`.
4. Output must always be in `[0, 1]`.
5. Covariance matrix should remain symmetric or near-symmetric.

---

## Track A tests

1. Output dictionary must contain:
   - `sigma_phys`
   - `sigma_kf`
   - `sigma_a`
2. `sigma_a` must equal weighted fusion.
3. All outputs must be in `[0, 1]`.

---

## LSTM tests

1. Input shape `(32, 10, 10)` should work.
2. Gaussian mode should output `mu` and `logvar`, each shape `(32,)`.
3. Scalar mode should output `sigma`, shape `(32,)`.
4. `mu` or `sigma` should be in `[0, 1]`.
5. Backpropagation should work with Gaussian NLL loss.

---

# 9. Implementation Notes

- Use NumPy for Track A.
- Use PyTorch for Track B.
- Keep configurations centralized in `config.py`.
- Make all magic numbers configurable.
- Clip all final uncertainty values to `[0, 1]`.
- Keep Track A fully usable without PyTorch.
- Make Track B independent enough to train separately.
- Add type hints and docstrings.
- Include a small demo script showing:
  1. Track A on a toy sequence with visibility changes.
  2. Track B forward pass on random tensor input.

---

# 10. Acceptance Criteria

The implementation is complete when:

- `TrackAEstimator.step(...)` works per vehicle ID and returns all three uncertainty values.
- Kalman uncertainty decreases with stable observation and increases under occlusion.
- LSTM model accepts `(B, 10, 10)` input.
- Gaussian mode returns `(mu, logvar)`.
- Gaussian NLL loss works.
- Unit tests pass.
- A demo script can run locally without external datasets.

---

# 11. Minimal Demo Expectations

Please provide scripts like:

```bash
python examples/demo_track_a.py
python examples/demo_lstm_forward.py
pytest
```

`demo_track_a.py` should print uncertainty over time for one vehicle:

```text
new vehicle: sigma_kf high
stable visible steps: sigma_kf decreases
occluded steps: sigma_kf increases
```

`demo_lstm_forward.py` should create:

```python
x = torch.randn(32, 10, 10)
```

and confirm output shapes.

---

# 12. Important Interpretation

Track A is not meant to be perfect. It is a transparent baseline.

Track B is the learned model. It should learn richer temporal patterns from the recent 10-step sequence and can express confidence through Gaussian output variance.

The two tracks should be implemented so that they can be compared experimentally.
