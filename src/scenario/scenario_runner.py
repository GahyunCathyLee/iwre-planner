import math
import random
import time

import carla

from config import (
    ACCEL_TO_THROTTLE_SCALE,
    DECEL_TO_BRAKE_SCALE,
    EGO_VEHICLE_FILTER,
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
from scenario.spawn_manager import ScenarioSpawnManager
from utils.carla_utils import cleanup_actors, reset_world_settings, set_synchronous_mode
from utils.logger import CSVLogger


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


class ScenarioRunner:
    def __init__(self, args, scenario_config):
        self.args = args
        self.scenario_config = scenario_config
        self.planner_name = args.planner.upper()
        if self.planner_name != "B1":
            raise NotImplementedError("ScenarioRunner currently supports only planner B1.")
        self.alpha_runtime = None
        if getattr(args, "alpha_model_path", None):
            from interaction.grip_alpha_runtime import GRIPAlphaRuntime

            self.alpha_runtime = GRIPAlphaRuntime(
                checkpoint_path=args.alpha_model_path,
                history=int(getattr(args, "alpha_history", 6)),
            )

    def run(self):
        random.seed(self.scenario_config.seed)
        rng = random.Random(self.scenario_config.seed)

        actors = []
        ego_logger = None
        nbr_rows = []
        world = None
        traffic_manager = None
        collision_steps = set()
        collision_step = {"value": -1}

        client = carla.Client(self.args.host, self.args.port)
        client.set_timeout(30.0)

        try:
            print("[INFO] Connecting to CARLA...")
            world = client.load_world(self.scenario_config.map)
            print(f"[INFO] Loaded map: {world.get_map().name}")

            traffic_manager = client.get_trafficmanager(self.args.tm_port)
            traffic_manager.set_random_device_seed(self.scenario_config.seed)
            traffic_manager.set_global_distance_to_leading_vehicle(
                GLOBAL_DISTANCE_TO_LEADING_VEHICLE
            )

            set_synchronous_mode(world, traffic_manager, self.scenario_config.fps)

            blueprint_library = world.get_blueprint_library()
            spawn_manager = ScenarioSpawnManager(
                world=world,
                blueprint_library=blueprint_library,
                traffic_manager=traffic_manager,
                rng=rng,
            )

            ego_vehicle = spawn_manager.spawn_ego(
                self.scenario_config.ego_spawn,
                EGO_VEHICLE_FILTER,
            )
            actors.append(ego_vehicle)
            collision_sensor = self._spawn_collision_sensor(
                world,
                blueprint_library,
                ego_vehicle,
                collision_steps,
                collision_step,
            )
            if collision_sensor is not None:
                actors.append(collision_sensor)

            actor_entries = spawn_manager.spawn_neighbors(
                self.scenario_config.actors,
                ego_vehicle,
            )
            actors.extend(vehicle for _, vehicle in actor_entries)
            neighbor_vehicles = [vehicle for _, vehicle in actor_entries]
            actor_configs_by_vehicle_id = {
                vehicle.id: actor_config for actor_config, vehicle in actor_entries
            }

            behavior_manager = BehaviorManager(actor_entries)
            idm_planner = IDMPlanner(
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

            run_id = self.args.run_id or self._make_run_id()
            ego_log_path = f"{LOG_DIR}/{run_id}_ego.csv"
            nbr_log_path = f"{LOG_DIR}/{run_id}_nbr.csv"

            ego_logger = CSVLogger(ego_log_path, self._ego_log_fields())
            print(
                f"[INFO] Starting scenario={self.scenario_config.scenario_name}, "
                f"planner={self.planner_name}, run_id={run_id}"
            )

            for step in range(self.scenario_config.steps):
                collision_step["value"] = step
                world.tick()
                behavior_manager.tick(step)

                ego_transform = ego_vehicle.get_transform()
                ego_location = ego_transform.location
                ego_velocity = ego_vehicle.get_velocity()
                ego_acceleration = ego_vehicle.get_acceleration()
                ego_speed = get_speed(ego_vehicle)
                ego_lane_id = self._get_lane_id(world, ego_location)
                slot_assignments, vehicle_slots = self._assign_slots(
                    world,
                    ego_vehicle,
                    neighbor_vehicles,
                )

                front_vehicle, front_distance, front_speed = get_front_vehicle(
                    ego_vehicle=ego_vehicle,
                    candidate_vehicles=neighbor_vehicles,
                    max_distance=IDM_MAX_DETECTION_DISTANCE,
                    lane_check=True,
                )
                relative_speed = 0.0 if front_speed is None else ego_speed - front_speed
                acceleration = idm_planner.compute_acceleration(
                    ego_speed=ego_speed,
                    front_distance=front_distance,
                    relative_speed=relative_speed,
                )
                control = controller.acceleration_to_control(
                    acceleration=acceleration,
                    steer=0.0,
                )
                ego_vehicle.apply_control(control)

                ego_row = {
                    "run_id": run_id,
                    "step": step,
                    "ego_x": f"{ego_location.x:.3f}",
                    "ego_y": f"{ego_location.y:.3f}",
                    "ego_vx": f"{ego_velocity.x:.3f}",
                    "ego_vy": f"{ego_velocity.y:.3f}",
                    "ego_ax": f"{ego_acceleration.x:.3f}",
                    "ego_ay": f"{ego_acceleration.y:.3f}",
                    "ego_speed": f"{ego_speed:.3f}",
                    "throttle": f"{control.throttle:.3f}",
                    "brake": f"{control.brake:.3f}",
                    "steer": f"{control.steer:.3f}",
                    "lane_id": ego_lane_id,
                }
                for slot_name in SLOT_NAMES:
                    vehicle = slot_assignments.get(slot_name)
                    ego_row[slot_name] = vehicle.id if vehicle is not None else ""
                ego_logger.log(ego_row)

                nbr_rows.extend(
                    self._collect_neighbor_rows(
                        world=world,
                        run_id=run_id,
                        step=step,
                        ego_vehicle=ego_vehicle,
                        vehicle_slots=vehicle_slots,
                        neighbor_vehicles=neighbor_vehicles,
                        actor_configs_by_vehicle_id=actor_configs_by_vehicle_id,
                        collision=step in collision_steps,
                    )
                )

                if self.args.verbose and step % 20 == 0:
                    print(
                        f"[STEP {step:04d}] "
                        f"speed={ego_speed:.2f} m/s, "
                        f"neighbors={len(neighbor_vehicles)}, "
                        f"front_dist={front_distance if front_distance is not None else 'None'}, "
                        f"accel={acceleration:.2f}"
                    )

            self._write_neighbor_log(nbr_log_path, nbr_rows)
            print(
                f"[INFO] Scenario finished. Logs saved to: "
                f"{ego_log_path}, {nbr_log_path}"
            )

        finally:
            if ego_logger is not None:
                ego_logger.close()
            if world is not None and traffic_manager is not None:
                reset_world_settings(world, traffic_manager)
            cleanup_actors(client, actors)
            time.sleep(0.5)

    def _make_run_id(self):
        return f"s{self.scenario_config.scenario_id}_{self.planner_name}_seed{self.scenario_config.seed}"

    def _spawn_collision_sensor(
        self,
        world,
        blueprint_library,
        ego_vehicle,
        collision_steps,
        collision_step,
    ):
        collision_bps = blueprint_library.filter("sensor.other.collision")
        if not collision_bps:
            print("[WARN] Collision sensor blueprint not found; collision column will stay 0.")
            return None

        sensor = world.spawn_actor(
            collision_bps[0],
            carla.Transform(),
            attach_to=ego_vehicle,
        )

        def _on_collision(_event):
            collision_steps.add(collision_step["value"])

        sensor.listen(_on_collision)
        return sensor

    def _assign_slots(self, world, ego_vehicle, neighbor_vehicles):
        lane_width = 3.5
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()
        ego_waypoint = world.get_map().get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_waypoint is not None and ego_waypoint.lane_width:
            lane_width = ego_waypoint.lane_width

        candidates = {slot_name: [] for slot_name in SLOT_NAMES}

        for vehicle in neighbor_vehicles:
            if vehicle is None or not vehicle.is_alive:
                continue

            location = vehicle.get_transform().location
            relative = location - ego_location
            longitudinal = (
                relative.x * ego_forward.x
                + relative.y * ego_forward.y
                + relative.z * ego_forward.z
            )
            lateral = ego_forward.x * relative.y - ego_forward.y * relative.x

            same_lane_limit = 0.5 * lane_width
            adjacent_lane_limit = 1.8 * lane_width
            alongside_limit = 8.0

            if abs(lateral) <= same_lane_limit:
                if longitudinal >= 0.0:
                    candidates["preceding"].append((abs(longitudinal), vehicle))
                else:
                    candidates["following"].append((abs(longitudinal), vehicle))
            elif same_lane_limit < lateral <= adjacent_lane_limit:
                if longitudinal > alongside_limit:
                    candidates["leftPreceding"].append((abs(longitudinal), vehicle))
                elif longitudinal < -alongside_limit:
                    candidates["leftFollowing"].append((abs(longitudinal), vehicle))
                else:
                    candidates["leftAlongside"].append((abs(longitudinal), vehicle))
            elif -adjacent_lane_limit <= lateral < -same_lane_limit:
                if longitudinal > alongside_limit:
                    candidates["rightPreceding"].append((abs(longitudinal), vehicle))
                elif longitudinal < -alongside_limit:
                    candidates["rightFollowing"].append((abs(longitudinal), vehicle))
                else:
                    candidates["rightAlongside"].append((abs(longitudinal), vehicle))

        slot_assignments = {}
        vehicle_slots = {}
        for slot_name, slot_candidates in candidates.items():
            if not slot_candidates:
                continue
            _, vehicle = min(slot_candidates, key=lambda item: item[0])
            slot_assignments[slot_name] = vehicle
            vehicle_slots[vehicle.id] = slot_name

        return slot_assignments, vehicle_slots

    def _collect_neighbor_rows(
        self,
        world,
        run_id,
        step,
        ego_vehicle,
        vehicle_slots,
        neighbor_vehicles,
        actor_configs_by_vehicle_id,
        collision,
    ):
        rows = []
        ego_transform = ego_vehicle.get_transform()
        ego_location = ego_transform.location
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed = self._speed(ego_velocity)
        ego_forward = ego_transform.get_forward_vector()
        time_sec = step / self.scenario_config.fps

        for vehicle in neighbor_vehicles:
            if vehicle is None or not vehicle.is_alive:
                continue

            transform = vehicle.get_transform()
            location = transform.location
            velocity = vehicle.get_velocity()
            acceleration = vehicle.get_acceleration()
            dx_gt = location.x - ego_location.x
            dy_gt = location.y - ego_location.y
            distance_true = math.sqrt(dx_gt * dx_gt + dy_gt * dy_gt)
            neighbor_speed = self._speed(velocity)
            relative_speed_true = ego_speed - neighbor_speed
            relative_angle = self._relative_angle(ego_forward, dx_gt, dy_gt)
            longitudinal_true = dx_gt * ego_forward.x + dy_gt * ego_forward.y
            lane_relation = self._lane_relation(vehicle_slots.get(vehicle.id, ""))
            is_front = lane_relation == "same" and longitudinal_true > 0.0
            is_rear = lane_relation == "same" and not is_front
            is_adjacent = lane_relation in ("left", "right")
            actor_config = actor_configs_by_vehicle_id.get(vehicle.id)
            actor_key = actor_config.id if actor_config is not None else str(vehicle.id)

            noise_position_std, noise_velocity_std = self._noise_std(
                distance=distance_true,
                relative_angle=abs(relative_angle),
                relative_speed=abs(relative_speed_true),
                lane_relation=lane_relation,
                actor_config=actor_config,
            )
            position_noise_x = self._noise_rng(actor_key, step, "px").gauss(
                0.0, noise_position_std
            )
            position_noise_y = self._noise_rng(actor_key, step, "py").gauss(
                0.0, noise_position_std
            )
            velocity_noise_x = self._noise_rng(actor_key, step, "vx").gauss(
                0.0, noise_velocity_std
            )
            velocity_noise_y = self._noise_rng(actor_key, step, "vy").gauss(
                0.0, noise_velocity_std
            )
            x_obs = location.x + position_noise_x
            y_obs = location.y + position_noise_y
            vx_obs = velocity.x + velocity_noise_x
            vy_obs = velocity.y + velocity_noise_y
            dx_obs = x_obs - ego_location.x
            dy_obs = y_obs - ego_location.y
            obs_speed = math.sqrt(vx_obs * vx_obs + vy_obs * vy_obs)
            relative_speed_obs = ego_speed - obs_speed
            longitudinal_obs = dx_obs * ego_forward.x + dy_obs * ego_forward.y
            position_error = math.sqrt(position_noise_x ** 2 + position_noise_y ** 2)
            velocity_error = math.sqrt(velocity_noise_x ** 2 + velocity_noise_y ** 2)
            sigma_label = self._sigma_label(
                position_error,
                velocity_error,
                noise_position_std,
                noise_velocity_std,
            )
            sigma_estimated = self._sigma_estimated(
                noise_position_std,
                noise_velocity_std,
                distance_true,
            )
            alpha = self._alpha(distance_true, is_front, is_adjacent, is_rear)
            risk = sigma_estimated * alpha
            ttc_true = self._ttc(longitudinal_true, relative_speed_true, is_front)
            ttc_obs = self._ttc(longitudinal_obs, relative_speed_obs, is_front)
            near_miss_threshold = self._near_miss_ttc_threshold()
            near_miss_true = self._near_miss(ttc_true, near_miss_threshold)
            near_miss_obs = self._near_miss(ttc_obs, near_miss_threshold)

            rows.append({
                "run_id": run_id,
                "scenario_id": self.scenario_config.scenario_id,
                "planner": self.planner_name,
                "seed": self.scenario_config.seed,
                "vehicle_id": vehicle.id,
                "neighbor_id": vehicle.id,
                "actor_id": actor_key,
                "actor_role": actor_config.role if actor_config is not None else "",
                "step": step,
                "time_sec": f"{time_sec:.3f}",
                "slot": vehicle_slots.get(vehicle.id, ""),
                "ego_id": ego_vehicle.id,
                "ego_x": f"{ego_location.x:.3f}",
                "ego_y": f"{ego_location.y:.3f}",
                "ego_vx": f"{ego_velocity.x:.3f}",
                "ego_vy": f"{ego_velocity.y:.3f}",
                "x_gt": f"{location.x:.3f}",
                "y_gt": f"{location.y:.3f}",
                "vx_gt": f"{velocity.x:.3f}",
                "vy_gt": f"{velocity.y:.3f}",
                "ax_gt": f"{acceleration.x:.3f}",
                "ay_gt": f"{acceleration.y:.3f}",
                "dx_gt": f"{dx_gt:.3f}",
                "dy_gt": f"{dy_gt:.3f}",
                "lane_id": self._get_lane_id(world, location),
                "x_obs": f"{x_obs:.3f}",
                "y_obs": f"{y_obs:.3f}",
                "vx_obs": f"{vx_obs:.3f}",
                "vy_obs": f"{vy_obs:.3f}",
                "dx_obs": f"{dx_obs:.3f}",
                "dy_obs": f"{dy_obs:.3f}",
                "noise_mode": self._noise_mode(),
                "noise_scale": f"{self._noise_scale():.3f}",
                "noise_position_std": f"{noise_position_std:.3f}",
                "noise_velocity_std": f"{noise_velocity_std:.3f}",
                "position_error": f"{position_error:.3f}",
                "velocity_error": f"{velocity_error:.3f}",
                "sigma_label": f"{sigma_label:.3f}",
                "sigma_estimated": f"{sigma_estimated:.3f}",
                "alpha": f"{alpha:.3f}",
                "risk": f"{risk:.3f}",
                "ttc_true": self._fmt_optional(ttc_true),
                "ttc_obs": self._fmt_optional(ttc_obs),
                "near_miss_true": int(near_miss_true),
                "near_miss_obs": int(near_miss_obs),
                "collision": int(collision),
            })

        self._apply_runtime_alpha(rows, ego_velocity)
        return rows

    def _apply_runtime_alpha(self, rows, ego_velocity):
        if self.alpha_runtime is None or not rows:
            return

        ego_state = {"ego_vx": ego_velocity.x, "ego_vy": ego_velocity.y}
        neighbors_by_slot = {}
        for row in rows:
            slot = row.get("slot", "")
            if not slot:
                continue
            neighbors_by_slot[slot] = {
                "dx_obs": float(row["dx_obs"]),
                "dy_obs": float(row["dy_obs"]),
                "vx_obs": float(row["vx_obs"]),
                "vy_obs": float(row["vy_obs"]),
            }

        self.alpha_runtime.update(ego_state, neighbors_by_slot)
        alpha_by_slot = self.alpha_runtime.predict()
        if not alpha_by_slot:
            return

        for row in rows:
            slot = row.get("slot", "")
            if slot not in alpha_by_slot:
                continue
            alpha = alpha_by_slot[slot]
            sigma = float(row["sigma_estimated"])
            row["alpha"] = f"{alpha:.3f}"
            row["risk"] = f"{sigma * alpha:.3f}"

    def _noise_mode(self):
        return str(self.scenario_config.noise.get("mode", "none")).lower()

    def _noise_scale(self):
        noise = self.scenario_config.noise or {}
        if "scale" in noise:
            return max(0.0, float(noise["scale"]))

        return {
            "none": 0.0,
            "very_low": 0.25,
            "low": 0.5,
            "medium": 1.0,
            "medium_high": 1.35,
            "high": 1.8,
            "very_high": 2.5,
        }.get(self._noise_mode(), 1.0)

    def _noise_std(
        self,
        distance,
        relative_angle,
        relative_speed,
        lane_relation,
        actor_config,
    ):
        noise = self.scenario_config.noise or {}
        mode_scale = self._noise_scale()
        if mode_scale <= 0.0:
            return 0.0, 0.0

        base_position_std = float(noise.get("base_position_std", 0.25))
        base_velocity_std = float(noise.get("base_velocity_std", 0.10))
        distance_factor = float(noise.get("distance_factor", 0.015))
        angle_factor = float(noise.get("angle_factor", 0.30))
        relative_motion_factor = float(noise.get("relative_motion_factor", 0.02))
        occlusion_factor = float(noise.get("occlusion_factor", 0.0))
        scenario_bias = float(noise.get("scenario_bias", 0.0))

        lane_bias = 0.15 if lane_relation in ("left", "right") else 0.0
        cut_in_bias = 0.20 if actor_config is not None and actor_config.cut_in_actor else 0.0
        multiplier = (
            1.0
            + distance_factor * min(distance, 120.0)
            + angle_factor * min(relative_angle / math.pi, 1.0)
            + relative_motion_factor * min(relative_speed, 30.0)
            + occlusion_factor
            + scenario_bias
            + lane_bias
            + cut_in_bias
        )
        return (
            base_position_std * mode_scale * multiplier,
            base_velocity_std * mode_scale * multiplier,
        )

    def _noise_rng(self, actor_key, step, channel):
        actor_seed = sum((index + 1) * ord(char) for index, char in enumerate(str(actor_key)))
        channel_seed = sum(ord(char) for char in channel)
        seed = (
            int(self.scenario_config.seed) * 1_000_003
            + actor_seed * 9_176
            + int(step) * 101
            + channel_seed
        )
        return random.Random(seed)

    @staticmethod
    def _speed(velocity):
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    @staticmethod
    def _relative_angle(ego_forward, dx, dy):
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0
        ego_heading = math.atan2(ego_forward.y, ego_forward.x)
        neighbor_heading = math.atan2(dy, dx)
        angle = neighbor_heading - ego_heading
        return math.atan2(math.sin(angle), math.cos(angle))

    @staticmethod
    def _lane_relation(slot):
        if slot in ("preceding", "following"):
            return "same"
        if slot.startswith("left"):
            return "left"
        if slot.startswith("right"):
            return "right"
        return "other"

    @staticmethod
    def _sigma_label(position_error, velocity_error, position_std, velocity_std):
        if position_std <= 0.0 and velocity_std <= 0.0:
            return 0.0
        position_term = position_error / max(3.0 * position_std, 1e-6)
        velocity_term = velocity_error / max(3.0 * velocity_std, 1e-6)
        return min(1.0, 0.5 * (position_term + velocity_term))

    @staticmethod
    def _sigma_estimated(position_std, velocity_std, distance):
        return min(1.0, position_std / 4.0 + velocity_std / 2.0 + distance / 300.0)

    @staticmethod
    def _alpha(distance, is_front, is_adjacent, is_rear):
        distance_weight = max(0.1, 1.0 - min(distance, 100.0) / 100.0)
        if is_front:
            return min(1.0, 1.0 * distance_weight)
        if is_adjacent:
            return min(1.0, 0.55 * distance_weight)
        if is_rear:
            return min(1.0, 0.25 * distance_weight)
        return min(1.0, 0.15 * distance_weight)

    @staticmethod
    def _ttc(longitudinal_distance, relative_speed, is_front):
        if not is_front or longitudinal_distance <= 0.0 or relative_speed <= 1e-6:
            return None
        return longitudinal_distance / relative_speed

    def _near_miss_ttc_threshold(self):
        return float(self.scenario_config.noise.get("near_miss_ttc_threshold", 2.0))

    @staticmethod
    def _near_miss(ttc, threshold):
        return ttc is not None and 0.0 < ttc <= threshold

    @staticmethod
    def _fmt_optional(value):
        return "" if value is None or math.isinf(value) else f"{value:.3f}"

    def _write_neighbor_log(self, path, rows):
        logger = CSVLogger(path, self._neighbor_log_fields())
        try:
            for row in sorted(rows, key=lambda item: (int(item["vehicle_id"]), int(item["step"]))):
                logger.log(row)
        finally:
            logger.close()

    @staticmethod
    def _get_lane_id(world, location):
        waypoint = world.get_map().get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        return waypoint.lane_id if waypoint is not None else ""

    @staticmethod
    def _ego_log_fields():
        return [
            "run_id",
            "step",
            "ego_x",
            "ego_y",
            "ego_vx",
            "ego_vy",
            "ego_ax",
            "ego_ay",
            "ego_speed",
            "throttle",
            "brake",
            "steer",
            "lane_id",
            "preceding",
            "following",
            "leftPreceding",
            "leftAlongside",
            "leftFollowing",
            "rightPreceding",
            "rightAlongside",
            "rightFollowing",
        ]

    @staticmethod
    def _neighbor_log_fields():
        return [
            "run_id",
            "scenario_id",
            "planner",
            "seed",
            "vehicle_id",
            "neighbor_id",
            "actor_id",
            "actor_role",
            "step",
            "time_sec",
            "slot",
            "ego_id",
            "ego_x",
            "ego_y",
            "ego_vx",
            "ego_vy",
            "x_gt",
            "y_gt",
            "vx_gt",
            "vy_gt",
            "ax_gt",
            "ay_gt",
            "dx_gt",
            "dy_gt",
            "lane_id",
            "x_obs",
            "y_obs",
            "vx_obs",
            "vy_obs",
            "dx_obs",
            "dy_obs",
            "noise_mode",
            "noise_scale",
            "noise_position_std",
            "noise_velocity_std",
            "position_error",
            "velocity_error",
            "sigma_label",
            "sigma_estimated",
            "alpha",
            "risk",
            "ttc_true",
            "ttc_obs",
            "near_miss_true",
            "near_miss_obs",
            "collision",
        ]
