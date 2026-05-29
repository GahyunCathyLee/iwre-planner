import argparse
import math
import random
import time

import carla

from config import (
    DEFAULT_MAP,
    DEFAULT_FPS,
    DEFAULT_STEPS,
    DEFAULT_NUM_NPC,
    TRAFFIC_MANAGER_PORT,
    GLOBAL_DISTANCE_TO_LEADING_VEHICLE,
    EGO_VEHICLE_FILTER,
    IDM_DESIRED_SPEED,
    IDM_MIN_DISTANCE,
    IDM_TIME_HEADWAY,
    IDM_MAX_ACCEL,
    IDM_COMFORT_DECEL,
    IDM_DELTA,
    IDM_MAX_DETECTION_DISTANCE,
    MAX_THROTTLE,
    MAX_BRAKE,
    ACCEL_TO_THROTTLE_SCALE,
    DECEL_TO_BRAKE_SCALE,
    LOG_DIR,
)

from planner.idm_planner import IDMPlanner
from control.vehicle_controller import VehicleController
from perception.front_vehicle_detector import get_front_vehicle, get_speed
from utils.carla_utils import (
    set_synchronous_mode,
    reset_world_settings,
    spawn_ego_vehicle,
    spawn_npc_vehicles,
    cleanup_actors,
)
from utils.logger import CSVLogger
from scenario.scenario_loader import load_scenario
from scenario.scenario_runner import ScenarioRunner


def get_args():
    parser = argparse.ArgumentParser(description="B1: IDM-only baseline in CARLA")

    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--map", default=DEFAULT_MAP)
    parser.add_argument("--num-vehicles", type=int, default=DEFAULT_NUM_NPC)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--tm-port", type=int, default=TRAFFIC_MANAGER_PORT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenario", default=None)
    parser.add_argument(
        "--noise-mode",
        default=None,
        choices=["none", "very_low", "low", "medium", "medium_high", "high", "very_high"],
    )
    parser.add_argument("--noise-scale", type=float, default=None)
    parser.add_argument("--planner", default="B1", choices=["B1", "B2", "B3", "b1", "b2", "b3"])
    parser.add_argument(
        "--sigma-source",
        default="estimated",
        choices=["estimated", "v1", "v2", "v3", "ai_v1", "ai_v2", "ai_v3", "model", "track_a", "track_b"],
        help="Sigma source for B2/B3. track_a is rule/KF fusion; track_b is the LSTM uncertainty checkpoint.",
    )
    parser.add_argument("--sigma-model-path", default=None, help="Optional trained noise-model checkpoint for online sigma.")
    parser.add_argument("--sigma-history", type=int, default=10)
    parser.add_argument("--sigma-device", default=None, choices=["cpu", "cuda"], help="Optional device override for online sigma models.")
    parser.add_argument("--risk-gain", type=float, default=0.5)
    parser.add_argument("--min-distance-scale", type=float, default=0.2)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--alpha-model-path", default=None, help="Optional GRIP alpha checkpoint for online alpha logging.")
    parser.add_argument("--alpha-history", type=int, default=6)
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--log-path", default=f"{LOG_DIR}/b1_idm_log.csv")

    return parser.parse_args()


def compute_relative_speed(ego_speed, front_speed):
    """
    IDM uses delta_v = ego_speed - front_vehicle_speed.
    Positive value means ego is approaching the front vehicle.
    """

    if front_speed is None:
        return 0.0

    return ego_speed - front_speed


def apply_simple_lane_keeping_placeholder(ego_vehicle):
    """
    B1 minimum implementation uses zero steering.

    This is enough to verify IDM longitudinal control,
    but ego may not follow curved roads perfectly.

    Later, replace this with waypoint-based steering.
    """

    return 0.0


def main():
    args = get_args()

    if args.scenario:
        scenario_config = load_scenario(args.scenario, seed_override=args.seed)
        if args.steps is not None:
            scenario_config.steps = args.steps
        if args.fps is not None:
            scenario_config.fps = args.fps
        if args.noise_mode is not None:
            scenario_config.noise["mode"] = args.noise_mode
        if args.noise_scale is not None:
            scenario_config.noise["scale"] = args.noise_scale
        ScenarioRunner(args, scenario_config).run()
        return

    random.seed(args.seed)

    actors = []
    logger = None

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print("[INFO] Connecting to CARLA...")
    world = client.load_world(args.map)
    print(f"[INFO] Loaded map: {world.get_map().name}")

    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_random_device_seed(args.seed)
    traffic_manager.set_global_distance_to_leading_vehicle(
        GLOBAL_DISTANCE_TO_LEADING_VEHICLE
    )

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    if len(spawn_points) < args.num_vehicles + 1:
        raise RuntimeError("Not enough spawn points in the selected map.")

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

    log_fields = [
        "step",
        "time_sec",
        "ego_x",
        "ego_y",
        "ego_speed",
        "front_vehicle_id",
        "front_distance",
        "front_speed",
        "relative_speed",
        "idm_acceleration",
        "throttle",
        "brake",
        "steer",
    ]

    try:
        fps = args.fps if args.fps is not None else DEFAULT_FPS
        steps = args.steps if args.steps is not None else DEFAULT_STEPS

        set_synchronous_mode(world, traffic_manager, fps)

        ego_vehicle = spawn_ego_vehicle(
            world=world,
            blueprint_library=blueprint_library,
            spawn_points=spawn_points,
            vehicle_filter=EGO_VEHICLE_FILTER,
        )
        actors.append(ego_vehicle)

        npc_vehicles = spawn_npc_vehicles(
            world=world,
            blueprint_library=blueprint_library,
            spawn_points=spawn_points[1:],
            num_vehicles=args.num_vehicles,
            traffic_manager=traffic_manager,
        )
        actors.extend(npc_vehicles)

        # Important: ego is controlled manually by IDM.
        ego_vehicle.set_autopilot(False)

        logger = CSVLogger(args.log_path, log_fields)

        print("[INFO] Starting B1 IDM-only simulation...")

        for step in range(steps):
            world.tick()

            ego_transform = ego_vehicle.get_transform()
            ego_location = ego_transform.location
            ego_speed = get_speed(ego_vehicle)

            front_vehicle, front_distance, front_speed = get_front_vehicle(
                ego_vehicle=ego_vehicle,
                candidate_vehicles=npc_vehicles,
                max_distance=IDM_MAX_DETECTION_DISTANCE,
                lane_check=True,
            )

            relative_speed = compute_relative_speed(
                ego_speed=ego_speed,
                front_speed=front_speed,
            )

            acceleration = idm_planner.compute_acceleration(
                ego_speed=ego_speed,
                front_distance=front_distance,
                relative_speed=relative_speed,
            )

            steer = apply_simple_lane_keeping_placeholder(ego_vehicle)

            control = controller.acceleration_to_control(
                acceleration=acceleration,
                steer=steer,
            )

            ego_vehicle.apply_control(control)

            time_sec = step / fps

            logger.log({
                "step": step,
                "time_sec": f"{time_sec:.3f}",
                "ego_x": f"{ego_location.x:.3f}",
                "ego_y": f"{ego_location.y:.3f}",
                "ego_speed": f"{ego_speed:.3f}",
                "front_vehicle_id": front_vehicle.id if front_vehicle else "",
                "front_distance": f"{front_distance:.3f}" if front_distance is not None else "",
                "front_speed": f"{front_speed:.3f}" if front_speed is not None else "",
                "relative_speed": f"{relative_speed:.3f}",
                "idm_acceleration": f"{acceleration:.3f}",
                "throttle": f"{control.throttle:.3f}",
                "brake": f"{control.brake:.3f}",
                "steer": f"{control.steer:.3f}",
            })

            if args.verbose and step % 20 == 0:
                print(
                    f"[STEP {step:04d}] "
                    f"speed={ego_speed:.2f} m/s, "
                    f"front_dist={front_distance if front_distance is not None else 'None'}, "
                    f"accel={acceleration:.2f}, "
                    f"throttle={control.throttle:.2f}, "
                    f"brake={control.brake:.2f}"
                )

        print(f"[INFO] B1 simulation finished. Log saved to: {args.log_path}")

    finally:
        if logger is not None:
            logger.close()

        reset_world_settings(world, traffic_manager)
        cleanup_actors(client, actors)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
