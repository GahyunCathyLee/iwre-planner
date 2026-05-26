import random

import carla


class ScenarioSpawnManager:
    def __init__(self, world, blueprint_library, traffic_manager, rng=None):
        self.world = world
        self.blueprint_library = blueprint_library
        self.traffic_manager = traffic_manager
        self.rng = rng or random.Random()
        self.carla_map = world.get_map()

    def spawn_ego(self, spawn_config, vehicle_filter):
        spawn_points = self.carla_map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("Selected CARLA map has no spawn points.")

        base_transform = spawn_points[spawn_config.index % len(spawn_points)]
        transform = self._transform_from_config(base_transform, spawn_config)

        ego_bp = self.blueprint_library.filter(vehicle_filter)[0]
        if ego_bp.has_attribute("role_name"):
            ego_bp.set_attribute("role_name", "ego")

        ego_vehicle = self.world.try_spawn_actor(ego_bp, transform)
        if ego_vehicle is None:
            raise RuntimeError(f"Failed to spawn ego vehicle at slot {spawn_config}.")

        ego_vehicle.set_autopilot(False)
        print(f"[INFO] Ego vehicle spawned: {ego_vehicle.id}")
        return ego_vehicle

    def spawn_neighbors(self, actor_configs, ego_vehicle):
        spawned = []
        spawn_points = self.carla_map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("Selected CARLA map has no spawn points.")

        for actor_config in actor_configs:
            probability = max(0.0, min(1.0, actor_config.spawn_probability))
            if self.rng.random() > probability:
                print(f"[INFO] Skipped actor by probability: {actor_config.id}")
                continue

            vehicle = self._spawn_actor(actor_config, spawn_points)
            if vehicle is not None:
                spawned.append((actor_config, vehicle))

        print(f"[INFO] Scenario neighbor vehicles: {len(spawned)}")
        return spawned

    def _spawn_actor(self, actor_config, spawn_points):
        base_transform = spawn_points[actor_config.spawn.index % len(spawn_points)]
        sampled_offset = self._sample_actor_offset(actor_config)
        transform = self._transform_from_config(
            base_transform,
            actor_config.spawn,
            offset_override=sampled_offset,
        )

        vehicle_bp = self.rng.choice(list(self.blueprint_library.filter("vehicle.*")))
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", actor_config.id)

        vehicle = self.world.try_spawn_actor(vehicle_bp, transform)
        if vehicle is None:
            print(
                f"[WARN] Failed to spawn scenario actor: {actor_config.id} "
                f"at slot {actor_config.spawn}, sampled_offset={sampled_offset:.3f}"
            )
            return None

        if actor_config.autopilot:
            vehicle.set_autopilot(True, self.traffic_manager.get_port())
        else:
            vehicle.set_autopilot(False)

        print(
            f"[INFO] Scenario actor spawned: id={actor_config.id}, "
            f"carla_id={vehicle.id}, autopilot={actor_config.autopilot}"
        )
        return vehicle

    def _sample_actor_offset(self, actor_config):
        if actor_config.role != "slot_candidate":
            return actor_config.spawn.offset
        return self.rng.uniform(-10.0, 10.0)

    def _transform_from_config(self, base_transform, spawn_config, offset_override=None):
        waypoint = self.carla_map.get_waypoint(
            base_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return base_transform

        waypoint = self._move_lanes(waypoint, spawn_config.lane_offset)
        offset = spawn_config.offset if offset_override is None else offset_override
        waypoint = self._move_longitudinal(waypoint, offset)

        transform = waypoint.transform
        transform.location.z += 0.4
        return transform

    def _move_lanes(self, waypoint, lane_offset):
        current = waypoint
        steps = abs(int(lane_offset))

        for _ in range(steps):
            next_waypoint = (
                current.get_left_lane() if lane_offset > 0 else current.get_right_lane()
            )
            if next_waypoint is None or next_waypoint.lane_type != carla.LaneType.Driving:
                break
            current = next_waypoint

        return current

    def _move_longitudinal(self, waypoint, offset):
        distance = abs(float(offset))
        if distance < 1e-6:
            return waypoint

        candidates = waypoint.next(distance) if offset >= 0.0 else waypoint.previous(distance)
        if not candidates:
            return waypoint

        return candidates[0]
