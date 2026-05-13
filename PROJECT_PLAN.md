# Interaction-Weighted Risk Estimation Planner

## 1. Project Goal

This project builds a CARLA-based simulation framework for autonomous driving under sensor uncertainty.

The main objective is to evaluate whether an ego vehicle should react conservatively to all uncertain surrounding vehicles, or only to vehicles that are both uncertain and interaction-relevant.

The project compares three planner variants under identical scenarios:

- **B1: IDM only**
  - Uses only distance and relative speed.
  - Ignores sensor uncertainty.

- **B2: IDM + uncertainty only**
  - Uses uncertainty score \(\sigma_i\).
  - Treats high-uncertainty neighbors as high-risk.

- **B3: IDM + interaction-weighted risk**
  - Uses both uncertainty \(\sigma_i\) and interaction score \(\alpha_i\).
  - Computes risk using both terms, for example:

    ```text
    risk_i = sigma_i * alpha_i
    ```

  - The planner reacts conservatively only to high-risk neighbors.

---

## 2. Core Research Idea

Conventional uncertainty-aware planning can become overly conservative because it may react strongly to any high-uncertainty neighbor, even if that neighbor is not relevant to the ego vehicle.

This project separates:

- **Uncertainty**: how unreliable the ego vehicle's observation of a neighbor is.
- **Interaction**: how relevant that neighbor is to the ego vehicle's future motion.
- **Risk**: the final quantity used by the planner.

The key hypothesis is:

> A neighbor with high uncertainty but low interaction should not cause strong conservative behavior.

---

## 3. CARLA Setup

The project uses CARLA as a simulation environment.

Current setup:

- Server OS: Ubuntu 20.04.6 LTS
- Development: VS Code SSH
- Working directory: `~/iwre-planner`
- CARLA runs in headless/offscreen mode
- Main map: `Town04`
- Simulation mode: synchronous mode
- Default FPS: 20
- Default step time: 0.05 seconds
- Default episode length: 500 steps

---

## 4. Important Design Decisions

### 4.1 Ego Vehicle Control

The ego vehicle should **not** be controlled by CARLA Traffic Manager.

The ego vehicle should be controlled by this project's planner.

```text
Ego vehicle:
- autopilot OFF
- throttle/brake controlled by planner
- lateral behavior controlled by scenario or maneuver manager

NPC vehicles:
- autopilot ON
- controlled by CARLA Traffic Manager or scripted behavior
```

### 4.2 Scenario and Planner Separation

Scenarios and planners must be separated.

A scenario defines:

- map
- random seed
- ego spawn position
- neighbor spawn positions
- neighbor behaviors
- ego lateral intention
- sensor noise configuration
- episode length

A planner defines:

- how front or relevant vehicles are selected
- how uncertainty is used
- how interaction is used
- how risk is computed
- how effective distance is computed
- how throttle/brake are determined

This separation allows the same scenario to be run with different planners:

```bash
python src/main.py --scenario configs/scenarios/cut_in_left.yaml --planner B1
python src/main.py --scenario configs/scenarios/cut_in_left.yaml --planner B2
python src/main.py --scenario configs/scenarios/cut_in_left.yaml --planner B3
```

---

## 5. Planner Variants

### 5.1 B1: IDM Only

B1 is the baseline planner.

It uses the standard Intelligent Driver Model.

Inputs:

- ego speed
- front vehicle distance
- front vehicle relative speed

It does not use uncertainty or interaction.

```text
risk_i = 0
effective_distance = actual_distance
```

### 5.2 B2: IDM + Uncertainty Only

B2 uses uncertainty but not interaction.

```text
risk_i = sigma_i
effective_distance = actual_distance * (1 - risk_gain * risk_i)
```

This represents a conventional uncertainty-aware conservative planner.

### 5.3 B3: IDM + Interaction-Weighted Risk

B3 uses both uncertainty and interaction.

```text
risk_i = sigma_i * alpha_i
effective_distance = actual_distance * (1 - risk_gain * risk_i)
```

This planner should avoid unnecessary braking caused by high uncertainty from low-interaction vehicles.

---

## 6. Perception and Noise Modeling

CARLA actor APIs such as:

```python
actor.get_transform()
actor.get_velocity()
```

are treated as ground-truth-like simulation states.

They are not treated as noisy sensor observations.

Therefore, this project uses a pseudo-perception layer:

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

Noise should generally apply to all neighbors, because ego observes all neighbors through imperfect sensing.

However, noise magnitude should vary depending on observation conditions.

Recommended noise factors:

- distance factor
- relative angle factor
- relative motion factor
- optional occlusion factor
- optional scenario-specific bias

For ablation experiments, specific low-interaction vehicles may receive additional noise bias to test whether B3 avoids unnecessary conservative behavior.

---

## 7. Scenario Design

The project should support both fixed evaluation scenarios and randomized data-generation scenarios.

### 7.1 Fixed Evaluation Scenarios

These scenarios are used to compare B1, B2, and B3 under identical conditions.

Recommended scenarios:

| Scenario ID | Description |
|---|---|
| `lf_basic` | Lane following with normal traffic |
| `front_sudden_brake` | Front vehicle suddenly brakes |
| `front_sudden_accel` | Front vehicle suddenly accelerates |
| `cut_in_left` | Left-lane vehicle cuts into ego lane |
| `cut_in_right` | Right-lane vehicle cuts into ego lane |
| `dense_mixed` | Dense traffic with mixed behaviors |
| `ego_lane_change_scripted_left` | Ego performs fixed left lane change |
| `ego_lane_change_scripted_right` | Ego performs fixed right lane change |
| `ego_lane_change_conditional_left` | Ego changes left only when target lane gap is safe |
| `ego_lane_change_conditional_right` | Ego changes right only when target lane gap is safe |

---

## 8. Ego Lane Change Design

The project should support two types of ego lane-change scenarios.

### 8.1 Scripted Ego Lane Change

The scenario fixes the ego vehicle's lateral maneuver.

```text
ego starts lane change at configured step
ego follows a predefined lateral path
planner controls longitudinal behavior only
```

This is useful for comparing planner responses under the same lateral maneuver.

### 8.2 Conditional Ego Lane Change

The ego vehicle intends to change lanes, but starts the maneuver only if the target lane is safe.

Basic gap acceptance conditions:

```text
target_front_gap > min_front_gap
target_rear_gap > min_rear_gap
target_rear_ttc > min_rear_ttc
```

If the target lane is unsafe, ego waits.

This allows evaluation of how uncertainty and risk affect lane-change timing.

Important logs:

- lane change intended
- lane change started
- lane change completed
- lane change wait steps
- rejected gap count
- target front gap
- target rear gap
- target rear TTC

---

## 9. Neighbor Behavior Types

Neighbor vehicles can be divided into two groups.

### 9.1 Background NPCs

- Controlled by CARLA Traffic Manager
- Used to create general traffic density

### 9.2 Scripted NPCs

Used for specific experimental events.

Supported or planned behaviors:

- constant speed
- sudden braking
- sudden acceleration
- cut-in from left
- cut-in from right
- cut-out
- merge-in
- dense mixed interaction

---

## 10. Dataset Generation

The project should generate data that can be used both for planner evaluation and possible uncertainty model training.

Two logs are recommended.

### 10.1 Ego Log

File:

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

### 10.2 Neighbor Observation Log

File:

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

## 11. Scenario Configuration

Scenarios should be defined using YAML files.

Example command:

```bash
python src/main.py \
  --scenario configs/scenarios/lf_basic.yaml \
  --planner B1 \
  --run-id lf_basic_seed42_B1 \
  --seed 42
```

Recommended config directory:

```text
configs/scenarios/
```

Recommended initial scenario files:

```text
lf_basic.yaml
front_sudden_brake.yaml
front_sudden_accel.yaml
cut_in_left.yaml
cut_in_right.yaml
ego_lane_change_scripted_left.yaml
ego_lane_change_scripted_right.yaml
ego_lane_change_conditional_left.yaml
ego_lane_change_conditional_right.yaml
dense_mixed.yaml
random_batch.yaml
```

---

## 12. Randomized Data Generation

For uncertainty model training, fixed scenarios are not enough.

The project should also support randomized scenario generation.

Randomized factors:

- random seed
- ego spawn position
- neighbor spawn positions
- neighbor count
- neighbor speed
- behavior type
- number of scripted actors
- behavior start step
- braking intensity
- cut-in distance
- noise level
- high-noise actor ratio

Example randomized setting:

```text
seeds: 0-99
neighbor_count: 5, 10, 15, 20
behavior_mix: constant, brake, accel, cut-in
noise_level: low, medium, high
```

---

## 13. Recommended Implementation Roadmap

### Phase 1: Basic CARLA Control

- CARLA connection
- Town04 loading
- ego spawn
- NPC spawn
- synchronous simulation
- CSV logging

Status: completed.

### Phase 2: B1 IDM Baseline

- ego autopilot OFF
- NPC autopilot ON
- front vehicle detection
- IDM acceleration calculation
- throttle/brake control
- B1 logging

Status: in progress or partially completed.

### Phase 3: Scenario Framework

- YAML scenario config
- scenario loader
- spawn manager
- behavior manager
- ego maneuver manager
- scenario runner
- fixed scenarios

### Phase 4: Pseudo-Perception and Noise

- true state logging
- noisy observation generation
- geometry-dependent noise
- sigma label generation
- sigma estimation

### Phase 5: B2 Planner

- uncertainty-only risk
- effective distance modification
- B2 logs and metrics

### Phase 6: B3 Planner

- interaction score
- risk = sigma * alpha
- interaction-weighted effective distance
- B3 logs and metrics

### Phase 7: Evaluation and Metrics

Metrics:

- UDR: Unnecessary Deceleration Rate
- ASM: Average Speed Maintenance
- Travel Time
- Collision Rate
- Near-Miss Rate
- RIC: Risk-Interaction Correlation
- Lane-change wait time
- Rejected gap count
- Minimum TTC during lane change

---

## 14. Key Rule

The same scenario must be runnable with different planners.

Only planner behavior should change.

Scenario configuration, random seed, spawn positions, scripted behaviors, and noisy observations should remain consistent when comparing B1, B2, and B3.

