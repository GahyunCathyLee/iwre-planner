import math
import os
import random
import time
from typing import Dict, List, Optional

import carla

from config import (
    ACCEL_TO_THROTTLE_SCALE,
    DECEL_TO_BRAKE_SCALE,
    GLOBAL_DISTANCE_TO_LEADING_VEHICLE,
    IDM_COMFORT_DECEL,
    IDM_DELTA,
    IDM_DESIRED_SPEED,
    IDM_MAX_ACCEL,
    IDM_MAX_DETECTION_DISTANCE,
    IDM_MIN_DISTANCE,
    IDM_TIME_HEADWAY,
    LOG_DIR,
    MAX_BRAKE,
    MAX_THROTTLE,
)
from control.vehicle_controller import VehicleController
from perception.front_vehicle_detector import get_front_vehicle, get_speed
from planner.idm_planner import IDMPlanner
from scenario.behavior_manager import BehaviorManager
from scenario.ego_maneuver_manager import EgoManeuverManager
from scenario.spawn_manager import SpawnManager
from utils.carla_utils import cleanup_actors, reset_world_settings, set_synchronous_mode
from utils.logger import CSVLogger


EGO_LOG_FIELDS = [
    "run_id", "scenario_id", "planner", "seed", "step", "time_sec",
    "ego_x", "ego_y", "ego_vx", "ego_vy", "ego_speed",
    "throttle", "brake", "steer",
    "lane_change_intended", "lane_change_started", "lane_change_completed",
    "lane_change_start_step", "lane_change_wait_steps", "rejected_gap_count",
    "target_front_gap", "target_rear_gap", "target_rear_ttc",
]


NEIGHBOR_LOG_FIELDS = [
    "run_id", "scenario_id", "planner", "seed", "step", "time_sec",
    "ego_id", "neighbor_id",
    "ego_x", "ego_y", "ego_vx", "ego_vy",
    "neighbor_true_x", "neighbor_true_y", "neighbor_true_vx", "neighbor_true_vy",
    "neighbor_obs_x", "neighbor_obs_y", "neighbor_obs_vx", "neighbor_obs_vy",
    "dx_true", "dy_true", "dx_obs", "dy_obs",
    "distance_true", "distance_obs",
    "relative_speed_true", "relative_speed_obs", "relative_angle", "longitudinal_distance", "lateral_distance",
    "lane_relation", "neighbor_slot",
    "is_cut_in_actor",
    "position_error", "velocity_error",
    "sigma_label", "sigma_estimated",
]


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

SLOT_EGO_FIELDS = list(SLOT_NAMES)
EGO_LOG_FIELDS.extend(SLOT_EGO_FIELDS)


class ScenarioRunner:
    def __init__(self, args, scenario_config):
        self.args = args
        self.scenario = scenario_config
        self.planner_name = args.planner.upper()
        self.run_id = args.run_id or f"s{self.scenario.scenario_id}_{self.planner_name}_seed{self.scenario.seed}"
        self.rng = random.Random(self.scenario.seed)

    def run(self):
        if self.planner_name not in ("B1", "B2", "B3"):
            raise ValueError(f"Unsupported planner '{self.args.planner}'. Expected B1, B2, or B3.")

        client = carla.Client(self.args.host, self.args.port)
        client.set_timeout(30.0)
        world = client.load_world(self.scenario.town)
        self.carla_map = world.get_map()
        print(f"[INFO] Loaded map: {world.get_map().name}")

        traffic_manager = client.get_trafficmanager(self.args.tm_port)
        traffic_manager.set_random_device_seed(self.scenario.seed)
        traffic_manager.set_global_distance_to_leading_vehicle(
            float(self.scenario.traffic_manager.get("global_distance_to_leading_vehicle", GLOBAL_DISTANCE_TO_LEADING_VEHICLE))
        )

        actors = []
        ego_logger = None
        neighbor_logger = None
        neighbor_rows = []

        try:
            set_synchronous_mode(world, traffic_manager, self.scenario.fps)

            spawn_manager = SpawnManager(world, traffic_manager, self.rng)
            ego_vehicle, npc_vehicles, scripted_specs_by_id = spawn_manager.spawn(self.scenario)
            actors.append(ego_vehicle)
            actors.extend(npc_vehicles)

            behavior_manager = BehaviorManager(world, traffic_manager, scripted_specs_by_id)
            maneuver_manager = EgoManeuverManager(world, self.scenario.ego_maneuver)
            planner = IDMPlanner(
                desired_speed=IDM_DESIRED_SPEED,
                min_distance=IDM_MIN_DISTANCE,
                time_headway=IDM_TIME_HEADWAY,
                max_accel=IDM_MAX_ACCEL,
                comfort_decel=IDM_COMFORT_DECEL,
                delta=IDM_DELTA,
            )
            controller = VehicleController(
                max_throttle=MAX_THROTTLE,
                max_brake=MAX_BRAKE,
                accel_to_throttle_scale=ACCEL_TO_THROTTLE_SCALE,
                decel_to_brake_scale=DECEL_TO_BRAKE_SCALE,
            )

            ego_log_path = os.path.join(LOG_DIR, f"{self.run_id}_ego_log.csv")
            neighbor_log_path = os.path.join(LOG_DIR, f"{self.run_id}_neighbor_log.csv")
            ego_logger = CSVLogger(ego_log_path, EGO_LOG_FIELDS)
            neighbor_logger = CSVLogger(neighbor_log_path, NEIGHBOR_LOG_FIELDS)

            print(
                f"[INFO] Starting scenario {self.scenario.scenario_id} "
                f"('{self.scenario.scenario_name}') with planner {self.planner_name}"
            )

            for step in range(self.scenario.steps):
                behavior_manager.apply(step)
                world.tick()

                ego_transform = ego_vehicle.get_transform()
                ego_location = ego_transform.location
                ego_velocity = ego_vehicle.get_velocity()
                ego_speed = get_speed(ego_vehicle)

                observations = self._observe_neighbors(ego_vehicle, npc_vehicles, scripted_specs_by_id)
                slot_vehicle_ids = self._assign_neighbor_slots(observations)
                front_vehicle, front_distance, front_speed = get_front_vehicle(
                    ego_vehicle=ego_vehicle,
                    candidate_vehicles=npc_vehicles,
                    max_distance=IDM_MAX_DETECTION_DISTANCE,
                    lane_check=True,
                )
                front_obs = observations.get(front_vehicle.id) if front_vehicle is not None else None

                relative_speed = ego_speed - front_speed if front_speed is not None else 0.0
                sigma_front = front_obs["sigma_estimated"] if front_obs else 0.0
                alpha_front = front_obs["alpha"] if front_obs else 0.0
                risk_front = self._risk(sigma_front, alpha_front)
                effective_distance = self._effective_distance(front_distance, risk_front)

                acceleration = planner.compute_acceleration(
                    ego_speed=ego_speed,
                    front_distance=effective_distance,
                    relative_speed=relative_speed,
                )
                steer = maneuver_manager.compute_steer(ego_vehicle, npc_vehicles, step)
                control = controller.acceleration_to_control(acceleration=acceleration, steer=steer)
                ego_vehicle.apply_control(control)

                time_sec = step / self.scenario.fps
                self._collect_neighbor_rows(neighbor_rows, observations, ego_vehicle, step, time_sec)
                self._log_ego(
                    ego_logger,
                    ego_vehicle,
                    control,
                    step,
                    time_sec,
                    maneuver_manager.log_state(),
                    slot_vehicle_ids,
                )

                if self.args.verbose and step % 20 == 0:
                    print(
                        f"[STEP {step:04d}] speed={ego_speed:.2f} m/s, "
                        f"front_dist={_fmt(front_distance)}, risk={risk_front:.3f}, "
                        f"accel={acceleration:.2f}, throttle={control.throttle:.2f}, brake={control.brake:.2f}"
                    )

            print(f"[INFO] Scenario finished. Ego log: {ego_log_path}")
            for row in sorted(neighbor_rows, key=lambda item: (int(item["neighbor_id"]), int(item["step"]))):
                neighbor_logger.log(row)
            print(f"[INFO] Neighbor log: {neighbor_log_path}")

        finally:
            if ego_logger is not None:
                ego_logger.close()
            if neighbor_logger is not None:
                neighbor_logger.close()
            reset_world_settings(world, traffic_manager)
            cleanup_actors(client, actors)
            time.sleep(0.5)

    def _observe_neighbors(self, ego_vehicle, neighbors, scripted_specs_by_id) -> Dict[int, Dict]:
        observations = {}
        ego_transform = ego_vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed = _speed_from_vector(ego_velocity)
        ego_wp = self._waypoint(ego_loc)

        for actor in neighbors:
            if actor is None or not actor.is_alive:
                continue
            transform = actor.get_transform()
            loc = transform.location
            vel = actor.get_velocity()
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            distance = math.sqrt(dx * dx + dy * dy)
            longitudinal = dx * ego_forward.x + dy * ego_forward.y
            lateral = ego_forward.x * dy - ego_forward.y * dx
            relative_angle = math.atan2(dy, dx) - math.atan2(ego_forward.y, ego_forward.x)
            relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))
            lane_relation = self._lane_relation_from_geometry(ego_wp, lateral)
            is_front = longitudinal > 0.0 and lane_relation == "same"
            is_rear = longitudinal < 0.0 and lane_relation == "same"
            is_adjacent = lane_relation in ("left", "right")
            actor_speed = _speed_from_vector(vel)
            relative_speed_true = ego_speed - actor_speed
            pos_std, vel_std = self._noise_std(distance, abs(relative_angle), abs(relative_speed_true), actor.id, scripted_specs_by_id)

            obs_x = loc.x + self.rng.gauss(0.0, pos_std)
            obs_y = loc.y + self.rng.gauss(0.0, pos_std)
            obs_vx = vel.x + self.rng.gauss(0.0, vel_std)
            obs_vy = vel.y + self.rng.gauss(0.0, vel_std)
            dx_obs = obs_x - ego_loc.x
            dy_obs = obs_y - ego_loc.y
            distance_obs = math.sqrt(dx_obs * dx_obs + dy_obs * dy_obs)
            obs_speed = math.sqrt(obs_vx * obs_vx + obs_vy * obs_vy)
            relative_speed_obs = ego_speed - obs_speed
            position_error = math.sqrt((obs_x - loc.x) ** 2 + (obs_y - loc.y) ** 2)
            velocity_error = math.sqrt((obs_vx - vel.x) ** 2 + (obs_vy - vel.y) ** 2)
            sigma_estimated = min(1.0, (pos_std / 4.0) + (vel_std / 3.0))
            sigma_label = min(1.0, (position_error / max(3.0 * pos_std, 1e-6) + velocity_error / max(3.0 * vel_std, 1e-6)) / 2.0)
            alpha = self._alpha(distance, is_front, is_adjacent, is_rear)
            risk = self._risk(sigma_estimated, alpha)

            observations[actor.id] = {
                "neighbor_id": actor.id,
                "ego_x": ego_loc.x,
                "ego_y": ego_loc.y,
                "ego_vx": ego_velocity.x,
                "ego_vy": ego_velocity.y,
                "neighbor_true_x": loc.x,
                "neighbor_true_y": loc.y,
                "neighbor_true_vx": vel.x,
                "neighbor_true_vy": vel.y,
                "neighbor_obs_x": obs_x,
                "neighbor_obs_y": obs_y,
                "neighbor_obs_vx": obs_vx,
                "neighbor_obs_vy": obs_vy,
                "dx_true": dx,
                "dy_true": dy,
                "dx_obs": dx_obs,
                "dy_obs": dy_obs,
                "distance_true": distance,
                "distance_obs": distance_obs,
                "relative_speed_true": relative_speed_true,
                "relative_speed_obs": relative_speed_obs,
                "relative_angle": relative_angle,
                "lane_relation": lane_relation,
                "longitudinal_distance": longitudinal,
                "lateral_distance": lateral,
                "neighbor_slot": "",
                "is_cut_in_actor": bool(scripted_specs_by_id.get(actor.id) and scripted_specs_by_id[actor.id].cut_in_actor),
                "position_error": position_error,
                "velocity_error": velocity_error,
                "sigma_label": sigma_label,
                "sigma_estimated": sigma_estimated,
                "alpha": alpha,
                "risk": risk,
            }
        return observations

    def _assign_neighbor_slots(self, observations: Dict[int, Dict]) -> Dict[str, int]:
        candidates = {slot: [] for slot in SLOT_NAMES}
        alongside_margin = 8.0

        for actor_id, obs in observations.items():
            lane_relation = obs.get("lane_relation")
            longitudinal = obs.get("longitudinal_distance", 0.0)
            distance = obs.get("distance_true", math.inf)

            slot = None
            if lane_relation == "same":
                if longitudinal > 0.0:
                    slot = "preceding"
                elif longitudinal < 0.0:
                    slot = "following"
            elif lane_relation == "left":
                if longitudinal > alongside_margin:
                    slot = "leftPreceding"
                elif longitudinal < -alongside_margin:
                    slot = "leftFollowing"
                else:
                    slot = "leftAlongside"
            elif lane_relation == "right":
                if longitudinal > alongside_margin:
                    slot = "rightPreceding"
                elif longitudinal < -alongside_margin:
                    slot = "rightFollowing"
                else:
                    slot = "rightAlongside"

            if slot is not None:
                candidates[slot].append((abs(longitudinal), distance, actor_id))

        slot_vehicle_ids = {}
        assigned_actor_ids = set()
        for slot in SLOT_NAMES:
            for _, _, actor_id in sorted(candidates[slot]):
                if actor_id in assigned_actor_ids:
                    continue
                slot_vehicle_ids[slot] = actor_id
                observations[actor_id]["neighbor_slot"] = slot
                assigned_actor_ids.add(actor_id)
                break

        return slot_vehicle_ids

    def _noise_std(self, distance, angle, relative_speed, actor_id, scripted_specs_by_id):
        noise = self.scenario.noise
        mode_scale = {"none": 0.0, "low": 0.5, "medium": 1.0, "high": 1.8}.get(noise.mode, 1.0)
        if mode_scale == 0.0:
            return 0.0, 0.0
        actor_spec = scripted_specs_by_id.get(actor_id)
        actor_key = actor_spec.id if actor_spec else str(actor_id)
        bias = noise.high_noise_bias if actor_key in noise.high_noise_actor_ids else 0.0
        multiplier = (
            1.0
            + noise.distance_factor * distance
            + noise.angle_factor * min(angle / math.pi, 1.0)
            + noise.relative_motion_factor * relative_speed
            + noise.occlusion_factor
            + noise.scenario_bias
            + bias
        )
        return noise.base_position_std * mode_scale * multiplier, noise.base_velocity_std * mode_scale * multiplier

    def _risk(self, sigma, alpha):
        if self.planner_name == "B1":
            return 0.0
        if self.planner_name == "B2":
            return sigma
        return sigma * alpha

    def _effective_distance(self, front_distance: Optional[float], risk: float):
        if front_distance is None:
            return None
        risk_gain = 0.45
        return max(1.0, front_distance * (1.0 - risk_gain * risk))

    def _alpha(self, distance, is_front, is_adjacent, is_rear):
        distance_weight = max(0.1, 1.0 - min(distance, 80.0) / 100.0)
        if is_front:
            return min(1.0, 1.0 * distance_weight)
        if is_adjacent:
            return min(1.0, 0.55 * distance_weight)
        if is_rear:
            return min(1.0, 0.25 * distance_weight)
        return min(1.0, 0.15 * distance_weight)

    def _lane_relation(self, ego_wp, actor_wp):
        if ego_wp is None or actor_wp is None:
            return "other"
        if actor_wp.road_id != ego_wp.road_id or actor_wp.section_id != ego_wp.section_id:
            return "other"
        if actor_wp.lane_id == ego_wp.lane_id:
            return "same"
        left_wp = self._adjacent_same_direction_lane(ego_wp, "left")
        if left_wp is not None and actor_wp.lane_id == left_wp.lane_id:
            return "left"
        right_wp = self._adjacent_same_direction_lane(ego_wp, "right")
        if right_wp is not None and actor_wp.lane_id == right_wp.lane_id:
            return "right"
        return "other"

    def _lane_relation_from_geometry(self, ego_wp, lateral_distance):
        lane_width = ego_wp.lane_width if ego_wp is not None and ego_wp.lane_width else 3.5
        same_limit = 0.60 * lane_width
        adjacent_min = 0.60 * lane_width
        adjacent_max = 1.75 * lane_width

        abs_lateral = abs(lateral_distance)
        if abs_lateral <= same_limit:
            return "same"
        if adjacent_min < abs_lateral <= adjacent_max:
            return "left" if lateral_distance > 0.0 else "right"
        return "other"

    def _adjacent_same_direction_lane(self, waypoint, side):
        candidate = waypoint.get_left_lane() if side == "left" else waypoint.get_right_lane()
        if candidate is None or candidate.lane_type != carla.LaneType.Driving:
            return None
        if waypoint.lane_id * candidate.lane_id <= 0:
            return None
        return candidate

    def _lane_info(self, waypoint):
        if waypoint is None:
            return {"road_id": "", "section_id": "", "lane_id": "", "lane_index": ""}
        return {
            "road_id": waypoint.road_id,
            "section_id": waypoint.section_id,
            "lane_id": waypoint.lane_id,
            "lane_index": abs(waypoint.lane_id),
        }

    def _waypoint(self, location):
        return self.carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)

    def _collect_neighbor_rows(self, rows, observations, ego_vehicle, step, time_sec):
        for obs in observations.values():
            row = {
                "run_id": self.run_id,
                "scenario_id": self.scenario.scenario_id,
                "planner": self.planner_name,
                "seed": self.scenario.seed,
                "step": step,
                "time_sec": f"{time_sec:.3f}",
                "ego_id": ego_vehicle.id,
            }
            row.update({
                key: _fmt(value)
                for key, value in obs.items()
                if key in NEIGHBOR_LOG_FIELDS
            })
            rows.append(row)

    def _log_ego(self, logger, ego_vehicle, control, step, time_sec, lane_state, slot_vehicle_ids):
        transform = ego_vehicle.get_transform()
        velocity = ego_vehicle.get_velocity()
        row = {
            "run_id": self.run_id,
            "scenario_id": self.scenario.scenario_id,
            "planner": self.planner_name,
            "seed": self.scenario.seed,
            "step": step,
            "time_sec": f"{time_sec:.3f}",
            "ego_x": f"{transform.location.x:.3f}",
            "ego_y": f"{transform.location.y:.3f}",
            "ego_vx": f"{velocity.x:.3f}",
            "ego_vy": f"{velocity.y:.3f}",
            "ego_speed": f"{get_speed(ego_vehicle):.3f}",
            "throttle": f"{control.throttle:.3f}",
            "brake": f"{control.brake:.3f}",
            "steer": f"{control.steer:.3f}",
        }
        row.update({key: _fmt(value) for key, value in lane_state.items()})
        for slot in SLOT_NAMES:
            row[slot] = slot_vehicle_ids.get(slot, "")
        logger.log(row)


def _speed_from_vector(velocity):
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isinf(value):
            return ""
        return f"{value:.3f}"
    return value
