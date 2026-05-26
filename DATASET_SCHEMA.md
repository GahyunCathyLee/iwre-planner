# IWRE Planner 데이터셋 구조 설명

이 문서는 `scripts/run_dataset_generation.sh`로 생성한 CARLA 시뮬레이션 로그의 파일 구성과 각 CSV column의 의미를 설명한다. 산출물은 기본적으로 `outputs/logs/` 아래에 저장된다.

## 1. 생성 방식 요약

`scripts/run_dataset_generation.sh`는 여러 시나리오, seed, perception noise scale 조합에 대해 `src/main.py`를 반복 실행한다.

기본 실행 조건은 다음과 같다.

| 항목 | 기본값 | 설명 |
|---|---:|---|
| `PLANNER` | `B1` | IDM 기반 baseline planner |
| `SEEDS` | `81`-`120` | 각 시나리오별 반복 실행 seed |
| `STEPS` | `300` | run당 simulation step 수. 스크립트에서 `--steps`로 YAML 값을 override |
| `NOISE_SCALES` | `0.00`-`2.00` | seed별로 순환 배정되는 observation noise scale |
| `SCENARIOS` | 아래 기본 시나리오 목록 | 지정하지 않으면 script 내부 기본 목록 사용 |
| `RESTART_CARLA` | `1` | run마다 CARLA 재시작 |

기본 시나리오 목록은 다음 8개다.

| `scenario_id` | `scenario_name` | 파일 | 설명 |
|---:|---|---|---|
| 1 | `lf_basic` | `configs/scenarios/lf_basic.yaml` | 일반 lane following |
| 2 | `front_sudden_brake` | `configs/scenarios/front_sudden_brake.yaml` | 선행 차량 급감속 |
| 3 | `front_sudden_accel` | `configs/scenarios/front_sudden_accel.yaml` | 선행 차량 급가속 |
| 4 | `cut_in_left` | `configs/scenarios/cut_in_left.yaml` | 좌측 차선 차량 cut-in |
| 5 | `cut_in_right` | `configs/scenarios/cut_in_right.yaml` | 우측 차선 차량 cut-in |
| 8 | `ego_lane_change_conditional_left` | `configs/scenarios/ego_lane_change_conditional_left.yaml` | 안전 gap 조건 만족 시 ego 좌측 차선 변경 |
| 9 | `ego_lane_change_conditional_right` | `configs/scenarios/ego_lane_change_conditional_right.yaml` | 안전 gap 조건 만족 시 ego 우측 차선 변경 |
| 10 | `dense_mixed` | `configs/scenarios/dense_mixed.yaml` | 밀집 교통 및 혼합 scripted behavior |

## 2. 파일 구성

각 run은 같은 `run_id`를 공유하는 2개의 CSV 파일을 만든다.

```text
outputs/logs/<run_id>_ego.csv
outputs/logs/<run_id>_nbr.csv
```

`run_id` 형식은 다음과 같다.

```text
s<scenario_id>_<planner>_seed<seed>
```

예시는 다음과 같다.

```text
outputs/logs/s4_B1_seed81_ego.csv
outputs/logs/s4_B1_seed81_nbr.csv
```

각 파일의 역할은 다음과 같다.

| 파일 | row 단위 | 내용 |
|---|---|---|
| `<run_id>_ego.csv` | simulation step당 1 row | ego 차량 상태, 제어 입력, 주변 slot에 할당된 neighbor vehicle id |
| `<run_id>_nbr.csv` | simulation step × neighbor 차량당 1 row | neighbor 차량별 ground truth, noisy observation, noise/risk/TTC/near-miss/collision label |

## 3. 공통 좌표/단위 기준

| 항목 | 기준 |
|---|---|
| 위치 `x`, `y`, `dx`, `dy` | CARLA world coordinate 기준, 단위 m |
| 속도 `vx`, `vy`, `speed` | 단위 m/s |
| 가속도 `ax`, `ay` | 단위 m/s^2 |
| 시간 `time_sec` | `step / fps`. 기본 시나리오 YAML은 보통 `fps: 10`, script 실행 시 step 수만 override |
| 소수점 | 대부분 숫자 column은 CSV 저장 시 소수점 3자리 문자열로 기록 |
| `lane_id` | CARLA waypoint의 `lane_id` |

## 4. Ego CSV column 설명

`<run_id>_ego.csv`는 ego 차량을 기준으로 step별 상태와 주변 차량 slot assignment를 기록한다.

| Column | 의미 |
|---|---|
| `run_id` | 현재 run 식별자. 예: `s4_B1_seed81` |
| `step` | simulation step index. 0부터 시작 |
| `ego_x` | ego 차량의 CARLA world x 좌표 |
| `ego_y` | ego 차량의 CARLA world y 좌표 |
| `ego_vx` | ego 차량의 world x축 속도 |
| `ego_vy` | ego 차량의 world y축 속도 |
| `ego_ax` | ego 차량의 world x축 가속도 |
| `ego_ay` | ego 차량의 world y축 가속도 |
| `ego_speed` | ego 차량 속도 크기 |
| `throttle` | planner/controller가 적용한 throttle 명령. 0-1 범위 |
| `brake` | planner/controller가 적용한 brake 명령. 0-1 범위 |
| `steer` | planner/controller가 적용한 steering 명령. 현재 B1에서는 `0.000` |
| `lane_id` | ego 차량 위치의 CARLA lane id |
| `preceding` | ego와 같은 차선에서 ego 앞쪽에 있는 가장 가까운 neighbor vehicle id |
| `following` | ego와 같은 차선에서 ego 뒤쪽에 있는 가장 가까운 neighbor vehicle id |
| `leftPreceding` | ego 기준 좌측 인접 차선 앞쪽의 가장 가까운 neighbor vehicle id |
| `leftAlongside` | ego 기준 좌측 인접 차선 옆쪽의 가장 가까운 neighbor vehicle id |
| `leftFollowing` | ego 기준 좌측 인접 차선 뒤쪽의 가장 가까운 neighbor vehicle id |
| `rightPreceding` | ego 기준 우측 인접 차선 앞쪽의 가장 가까운 neighbor vehicle id |
| `rightAlongside` | ego 기준 우측 인접 차선 옆쪽의 가장 가까운 neighbor vehicle id |
| `rightFollowing` | ego 기준 우측 인접 차선 뒤쪽의 가장 가까운 neighbor vehicle id |

Slot column 값이 비어 있으면 해당 step에서 그 slot에 할당된 neighbor가 없다는 뜻이다.

## 5. Neighbor CSV column 설명

`<run_id>_nbr.csv`는 각 step에서 살아 있는 neighbor 차량마다 1 row를 기록한다. 즉 한 step에 neighbor가 6대 있으면 같은 `step` 값으로 6 row가 생긴다.

### 5.1 식별자 및 시뮬레이션 정보

| Column | 의미 |
|---|---|
| `run_id` | 현재 run 식별자 |
| `scenario_id` | 시나리오 번호 |
| `planner` | 사용 planner. 현재 script 기본값은 `B1` |
| `seed` | 현재 run의 random seed |
| `vehicle_id` | CARLA actor id. `neighbor_id`와 동일하게 기록됨 |
| `neighbor_id` | neighbor 차량 id. 현재는 `vehicle_id`와 동일 |
| `actor_id` | YAML 시나리오에 정의된 actor id. 예: `left_cut_in`, `preceding_candidate` |
| `actor_role` | YAML actor role. 예: `npc`, `scripted`, `slot_candidate` |
| `step` | simulation step index |
| `time_sec` | simulation 시간. `step / fps` |
| `slot` | ego 기준 slot assignment. 값이 비어 있으면 어떤 대표 slot에도 선택되지 않은 차량 |
| `ego_id` | CARLA ego actor id |

### 5.2 Ego 상태

| Column | 의미 |
|---|---|
| `ego_x` | 해당 step의 ego world x 좌표 |
| `ego_y` | 해당 step의 ego world y 좌표 |
| `ego_vx` | 해당 step의 ego world x축 속도 |
| `ego_vy` | 해당 step의 ego world y축 속도 |

### 5.3 Neighbor ground truth

`*_gt`는 CARLA simulator에서 직접 읽은 ground truth 값이다.

| Column | 의미 |
|---|---|
| `x_gt` | neighbor world x 좌표 |
| `y_gt` | neighbor world y 좌표 |
| `vx_gt` | neighbor world x축 속도 |
| `vy_gt` | neighbor world y축 속도 |
| `ax_gt` | neighbor world x축 가속도 |
| `ay_gt` | neighbor world y축 가속도 |
| `dx_gt` | `x_gt - ego_x` |
| `dy_gt` | `y_gt - ego_y` |
| `lane_id` | neighbor 위치의 CARLA lane id |

### 5.4 Neighbor noisy observation

`*_obs`는 ground truth에 perception noise를 더한 관측값이다.

| Column | 의미 |
|---|---|
| `x_obs` | noise가 반영된 neighbor x 관측값 |
| `y_obs` | noise가 반영된 neighbor y 관측값 |
| `vx_obs` | noise가 반영된 neighbor x축 속도 관측값 |
| `vy_obs` | noise가 반영된 neighbor y축 속도 관측값 |
| `dx_obs` | `x_obs - ego_x` |
| `dy_obs` | `y_obs - ego_y` |

### 5.5 Noise 및 uncertainty 관련 값

| Column | 의미 |
|---|---|
| `noise_mode` | 무시해도 되는 value. |
| `noise_scale` | 해당 run에 적용된 noise scale. `0.000`이면 observation noise 없음 (0.000-2.000 범위) |
| `noise_position_std` | 위치 noise sampling에 사용된 표준편차 |
| `noise_velocity_std` | 속도 noise sampling에 사용된 표준편차 |
| `position_error` | 적용된 위치 noise 크기. `sqrt(position_noise_x^2 + position_noise_y^2)` |
| `velocity_error` | 적용된 속도 noise 크기. `sqrt(velocity_noise_x^2 + velocity_noise_y^2)` |
| `sigma_label` | 실제로 주입된 noise error를 기반으로 만든 label. 0-1 범위 |
| `sigma_estimated` | distance와 noise std를 기반으로 추정한 uncertainty. 0-1 범위 |

`noise_position_std`와 `noise_velocity_std`는 거리, 상대 각도, 상대 속도, 차선 관계, cut-in actor 여부 등을 반영해 계산된다.

### 5.6 Risk, TTC, event label

| Column | 의미 |
|---|---|
| `alpha` | ego와 neighbor의 관계 및 거리에 따른 중요도 가중치. 앞차가 가장 크고, 옆차/뒤차/기타 순으로 낮아짐 |
| `risk` | `sigma_estimated * alpha` |
| `ttc_true` | ground truth 기준 time-to-collision. ego 앞 같은 차선 차량이고 ego가 접근 중일 때만 값이 있음 |
| `ttc_obs` | noisy observation 기준 time-to-collision. 조건은 `ttc_true`와 동일 |
| `near_miss_true` | `ttc_true`가 threshold 이하이면 1, 아니면 0 |
| `near_miss_obs` | `ttc_obs`가 threshold 이하이면 1, 아니면 0 |
| `collision` | 해당 step에서 ego collision sensor가 collision event를 감지했으면 1, 아니면 0 |

기본 near-miss TTC threshold는 scenario noise config의 `near_miss_ttc_threshold` 값을 사용하며, 없으면 `2.0`초다.

## 6. Slot assignment 기준

Slot은 ego 차량의 heading 방향을 기준으로 neighbor의 상대 위치를 계산해 분류한다.

| Slot | 기준 |
|---|---|
| `preceding` | 같은 차선 범위이고 ego 앞쪽 |
| `following` | 같은 차선 범위이고 ego 뒤쪽 |
| `leftPreceding` | 좌측 인접 차선이고 ego 앞쪽 |
| `leftAlongside` | 좌측 인접 차선이고 ego 옆쪽 |
| `leftFollowing` | 좌측 인접 차선이고 ego 뒤쪽 |
| `rightPreceding` | 우측 인접 차선이고 ego 앞쪽 |
| `rightAlongside` | 우측 인접 차선이고 ego 옆쪽 |
| `rightFollowing` | 우측 인접 차선이고 ego 뒤쪽 |

각 slot에는 후보 중 longitudinal distance가 가장 가까운 차량 하나만 할당된다. 따라서 `nbr.csv`의 `slot`이 비어 있는 row도 정상이며, 이는 해당 차량이 존재하지만 대표 slot으로 선택되지 않았다는 뜻이다.

## 7. 데이터 사용 시 주의사항

- `ego.csv`와 `nbr.csv`는 `run_id`와 `step`으로 join할 수 있다.
- `nbr.csv`는 차량별 row이므로 같은 `step`이 여러 번 반복된다.
- `vehicle_id`는 CARLA 실행 중 생성된 actor id라서 run 간 전역적으로 고유하다고 가정하면 안 된다. run 간 비교에는 `run_id`와 함께 사용해야 한다.
- `actor_id`는 시나리오 YAML의 actor 이름이므로, 시나리오 의미를 해석할 때 `vehicle_id`보다 안정적이다.
- `noise_scale=0.000`인 run에서는 `x_obs/y_obs/vx_obs/vy_obs`가 ground truth와 동일하고, `position_error`, `velocity_error`, `sigma_label`은 0이다.
- 현재 B1 planner는 longitudinal IDM 제어 중심이며, `steer`는 0으로 기록된다.
