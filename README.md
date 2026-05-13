# IWRE Planner

CARLA-based simulation framework for **Interaction-Weighted Risk Estimation (IWRE)** in autonomous driving under sensor uncertainty.

This project is designed to compare ego-vehicle planner variants under identical CARLA scenarios, with a focus on whether uncertainty should always lead to conservative behavior or whether uncertainty should be weighted by interaction relevance.

---

## 1. Project Goal

The main goal is to build a simulation framework where the same driving scenario can be executed with different planner variants:

| Planner | Description |
|---|---|
| **B1: IDM only** | Baseline planner using only distance and relative speed. No uncertainty. |
| **B2: IDM + uncertainty only** | Conservative uncertainty-aware planner. Uses uncertainty \(\sigma_i\) directly as risk. |
| **B3: IDM + interaction-weighted risk** | Proposed planner. Uses both uncertainty \(\sigma_i\) and interaction score \(\alpha_i\). |

The key risk formulation is:

```text
risk_i = f(sigma_i, alpha_i)
```

A simple version is:

```text
risk_i = sigma_i × alpha_i
```

The central hypothesis is:

> A neighbor with high uncertainty but low interaction with the ego vehicle should not cause strong conservative behavior.

---

## 2. Current Environment

Current development setup:

```text
OS: Ubuntu 20.04.6 LTS
Development: VS Code SSH
Working directory: ~/iwre-planner
CARLA execution: headless/offscreen mode
Primary map: Town04
Simulation mode: synchronous mode
Default FPS: 20
Default timestep: 0.05 seconds
Default episode length: 500 steps
```

---

## 3. Recommended Project Structure

```text
iwre-planner/
├── README.md
├── PROJECT_PLAN.md
├── configs/
│   └── scenarios/
│       ├── lf_basic.yaml
│       ├── front_sudden_brake.yaml
│       ├── front_sudden_accel.yaml
│       ├── cut_in_left.yaml
│       ├── cut_in_right.yaml
│       ├── ego_lane_change_scripted_left.yaml
│       ├── ego_lane_change_scripted_right.yaml
│       ├── ego_lane_change_conditional_left.yaml
│       ├── ego_lane_change_conditional_right.yaml
│       ├── dense_mixed.yaml
│       └── random_batch.yaml
│
├── carla-simulator/
│   ├── CarlaUE4.sh
│   ├── PythonAPI/
│   └── ...
│
├── src/
│   ├── main.py
│   ├── config.py
│   │
│   ├── scenario/
│   │   ├── scenario_config.py
│   │   ├── scenario_loader.py
│   │   ├── scenario_runner.py
│   │   ├── spawn_manager.py
│   │   ├── behavior_manager.py
│   │   ├── ego_maneuver_manager.py
│   │   └── random_scenario_generator.py
│   │
│   ├── planner/
│   │   ├── idm_planner.py
│   │   ├── b1_idm.py
│   │   ├── b2_uncertainty.py
│   │   └── b3_interaction_risk.py
│   │
│   ├── perception/
│   │   ├── front_vehicle_detector.py
│   │   ├── uncertainty_physics.py
│   │   └── pseudo_perception.py
│   │
│   ├── interaction/
│   │   ├── graph_builder.py
│   │   └── gat_model.py
│   │
│   ├── risk/
│   │   └── risk_modulation.py
│   │
│   ├── control/
│   │   └── vehicle_controller.py
│   │
│   └── utils/
│       ├── carla_utils.py
│       ├── logger.py
│       └── metrics.py
│
├── outputs/
│   ├── logs/
│   ├── metrics/
│   └── scenario_runs/
│
└── scripts/
    ├── run_carla.sh
    ├── run_b1_idm.sh
    ├── run_b2_uncertainty.sh
    ├── run_scenario.sh
    └── run_batch.sh
```

---

## 4. Core Design Principles

### 4.1 Ego Vehicle Control

The ego vehicle should **not** be controlled by CARLA Traffic Manager.

```text
Ego vehicle:
- autopilot OFF
- throttle/brake controlled by this project’s planner
- lateral behavior controlled by scenario or maneuver manager

NPC vehicles:
- autopilot ON, or
- controlled by scripted behavior manager
```

### 4.2 Scenario and Planner Separation

A scenario defines the driving situation.
A planner defines how the ego vehicle reacts.

Scenario responsibilities:

```text
- map
- FPS and episode length
- random seed
- ego spawn position
- neighbor spawn positions
- neighbor count
- scripted neighbor behavior
- ego lateral maneuver intention
- noise configuration
```

Planner responsibilities:

```text
- front or relevant vehicle selection
- uncertainty use
- interaction score use
- risk computation
- effective distance computation
- throttle/brake decision
```

This allows the same scenario to be run with different planners:

```bash
python src/main.py --scenario configs/scenarios/cut_in_left.yaml --planner B1 --run-id cut_in_left_B1
python src/main.py --scenario configs/scenarios/cut_in_left.yaml --planner B2 --run-id cut_in_left_B2
python src/main.py --scenario configs/scenarios/cut_in_left.yaml --planner B3 --run-id cut_in_left_B3
```

---

## 5. CARLA Server Execution

Start the CARLA server in one terminal:

```bash
cd ~/iwre-planner
./scripts/run_carla.sh
```

Recommended `scripts/run_carla.sh`:

```bash
#!/bin/bash

cd $HOME/iwre-planner/carla-simulator

./CarlaUE4.sh \
  -RenderOffScreen \
  -quality-level=Low \
  -carla-rpc-port=2000 \
  -nosound
```

The terminal may appear to be stuck after messages such as:

```text
Disabling core dumps.
```

This is usually normal because the CARLA server is running in the foreground.

---

## 6. Connection Test

Open another VS Code SSH terminal and run:

```bash
cd ~/iwre-planner
conda activate iwre-planner
python src/test_connection.py
```

Example `src/test_connection.py`:

```python
import carla

client = carla.Client("localhost", 2000)
client.set_timeout(10.0)

world = client.get_world()

print("Connected to CARLA")
print("Current map:", world.get_map().name)
```

Expected output:

```text
Connected to CARLA
Current map: Carla/Maps/...
```

---

## 7. Planner Variants

### 7.1 B1: IDM Only

B1 uses the Intelligent Driver Model without uncertainty or interaction weighting.

```text
risk_i = 0
effective_distance = actual_distance
```

B1 uses:

```text
- ego speed
- front vehicle distance
- front vehicle relative speed
```

---

### 7.2 B2: IDM + Uncertainty Only

B2 uses uncertainty directly as risk.

```text
risk_i = sigma_i
effective_distance = actual_distance × (1 - risk_gain × risk_i)
```

This represents a conservative uncertainty-aware planner.

---

### 7.3 B3: IDM + Interaction-Weighted Risk

B3 uses both uncertainty and interaction.

```text
risk_i = sigma_i × alpha_i
effective_distance = actual_distance × (1 - risk_gain × risk_i)
```

This planner should suppress unnecessary conservative reactions to high-uncertainty but low-interaction neighbors.

---

## 8. Scenario Framework

Scenarios should be defined using YAML files.

Example command:

```bash
python src/main.py \
  --scenario configs/scenarios/lf_basic.yaml \
  --planner B1 \
  --run-id lf_basic_seed42_B1 \
  --seed 42
```

Recommended initial scenarios:

| Scenario | Purpose |
|---|---|
| `lf_basic` | Lane following with normal traffic. |
| `front_sudden_brake` | Front vehicle suddenly brakes. |
| `front_sudden_accel` | Front vehicle suddenly accelerates. |
| `cut_in_left` | Left-lane vehicle cuts into ego lane. |
| `cut_in_right` | Right-lane vehicle cuts into ego lane. |
| `dense_mixed` | Dense traffic with mixed behaviors. |
| `ego_lane_change_scripted_left` | Ego follows a scripted left lane change. |
| `ego_lane_change_scripted_right` | Ego follows a scripted right lane change. |
| `ego_lane_change_conditional_left` | Ego changes left only when the target lane gap is safe. |
| `ego_lane_change_conditional_right` | Ego changes right only when the target lane gap is safe. |
| `random_batch` | Randomized scenario generation for dataset collection. |

---

## 9. Ego Lane Change Design

### 9.1 Scripted Ego Lane Change

The scenario fixes ego lateral motion.

```text
ego starts lane change at a configured step
ego follows a predefined lateral path
planner controls longitudinal behavior only
```

This is useful for comparing planner response under identical lateral movement.

---

### 9.2 Conditional Ego Lane Change

The ego vehicle intends to change lanes but waits until the target lane gap is safe.

Basic gap acceptance conditions:

```text
target_front_gap > min_front_gap
target_rear_gap > min_rear_gap
target_rear_ttc > min_rear_ttc
```

Recommended logs:

```text
lane_change_intended
lane_change_started
lane_change_completed
lane_change_start_step
lane_change_wait_steps
rejected_gap_count
target_front_gap
target_rear_gap
target_rear_ttc
```

Later, the gap decision can also become risk-aware:

```text
target_lane_max_risk < risk_threshold
```

---

## 10. Perception and Noise Modeling

CARLA actor APIs such as:

```python
actor.get_transform()
actor.get_velocity()
```

are treated as ground-truth-like simulator values.

They should **not** be treated as noisy ego sensor observations.

Therefore, this project should use a pseudo-perception layer:

```text
CARLA ground truth state
↓
pseudo sensor / observation noise model
↓
noisy observed neighbor state
↓
uncertainty estimation
↓
planner input
```

Noise should generally apply to all neighbors, but the magnitude should vary according to observation conditions.

Recommended factors:

```text
distance_factor
angle_factor
relative_motion_factor
optional occlusion_factor
optional scenario_bias
```

For ablation experiments, specific groups such as low-interaction or far vehicles may receive additional noise bias.

---

## 11. Dataset Logging

The project should generate logs for both planner evaluation and possible uncertainty model training.

### 11.1 Ego Log

Output file:

```text
outputs/logs/<run_id>_ego_log.csv
```

Recommended columns:

```text
run_id,
scenario_id,
planner,
seed,
step,
time_sec,
ego_x,
ego_y,
ego_vx,
ego_vy,
ego_speed,
throttle,
brake,
steer,
front_vehicle_id,
front_distance,
effective_distance,
sigma_front,
alpha_front,
risk_front,
idm_acceleration,
lane_change_intended,
lane_change_started,
lane_change_completed,
target_front_gap,
target_rear_gap,
target_rear_ttc
```

---

### 11.2 Neighbor Observation Log

Output file:

```text
outputs/logs/<run_id>_neighbor_log.csv
```

Recommended columns:

```text
run_id,
scenario_id,
planner,
seed,
step,
time_sec,
ego_id,
neighbor_id,
ego_x,
ego_y,
ego_vx,
ego_vy,
neighbor_true_x,
neighbor_true_y,
neighbor_true_vx,
neighbor_true_vy,
neighbor_obs_x,
neighbor_obs_y,
neighbor_obs_vx,
neighbor_obs_vy,
dx_true,
dy_true,
dx_obs,
dy_obs,
distance_true,
distance_obs,
relative_speed_true,
relative_speed_obs,
relative_angle,
lane_relation,
is_front,
is_adjacent,
is_rear,
is_cut_in_actor,
noise_position_std,
noise_velocity_std,
position_error,
velocity_error,
sigma_label,
sigma_estimated,
alpha,
risk
```

---

## 12. Metrics

Recommended evaluation metrics:

| Metric | Meaning | Desired Direction |
|---|---|---|
| UDR | Unnecessary Deceleration Rate | Lower is better |
| ASM | Average Speed Maintenance | Higher is better |
| Travel Time | Time to complete route or episode segment | Lower is better |
| Collision Rate | Number of collisions per episode | Lower is better |
| Near-Miss Rate | Frequency of TTC below threshold | Lower is better |
| RIC | Risk-Interaction Correlation | Higher is better |
| Lane-change wait time | Waiting time before conditional lane change | Lower if safe |
| Rejected gap count | Number of unsafe gap rejections | Scenario-dependent |
| Minimum TTC during LC | Safety during lane change | Higher is better |

---

## 13. Randomized Data Generation

For uncertainty model training, randomized scenario generation should be supported.

Randomizable factors:

```text
random seed
ego spawn position
neighbor spawn positions
neighbor count
neighbor speed
neighbor behavior type
number of scripted actors
behavior start step
braking intensity
cut-in distance
noise level
high-noise actor ratio
```

Example randomized setting:

```text
seeds: 0-99
neighbor_count: 5, 10, 15, 20
behavior_mix: constant, brake, accel, cut-in
noise_level: low, medium, high
```

---

## 14. Development Roadmap

### Phase 1: Basic CARLA Control

- CARLA connection
- Town04 loading
- ego spawn
- NPC spawn
- synchronous simulation
- CSV logging

Status: completed.

---

### Phase 2: B1 IDM Baseline

- ego autopilot OFF
- NPC autopilot ON
- front vehicle detection
- IDM acceleration calculation
- throttle/brake control
- B1 logging

Status: in progress or partially completed.

---

### Phase 3: Scenario Framework

- YAML scenario config
- scenario loader
- spawn manager
- behavior manager
- ego maneuver manager
- scenario runner
- fixed scenarios

---

### Phase 4: Pseudo-Perception and Noise

- true state logging
- noisy observation generation
- geometry-dependent noise
- sigma label generation
- sigma estimation

---

### Phase 5: B2 Planner

- uncertainty-only risk
- effective distance modification
- B2 logs and metrics

---

### Phase 6: B3 Planner

- interaction score
- risk = sigma × alpha
- interaction-weighted effective distance
- B3 logs and metrics

---

### Phase 7: Evaluation and Metrics

- compute UDR, ASM, travel time, collision rate, near-miss rate, RIC
- compare B1/B2/B3 under identical scenarios
- analyze lane-change timing and safety metrics

---

## 15. Key Rule

The same scenario must be runnable with different planners.

Only planner behavior should change.

When comparing B1, B2, and B3, the following should remain identical:

```text
scenario config
random seed
ego spawn position
neighbor spawn positions
scripted behavior schedule
noise realization, if possible
simulation length
```

This makes it possible to isolate the effect of uncertainty-aware and interaction-weighted risk modulation.
