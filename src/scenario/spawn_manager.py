import random
from typing import Dict, List, Tuple

import carla

from scenario.scenario_config import ActorSpec, ScenarioConfig, SpawnSpec


class SpawnManager:
    def __init__(self, world, traffic_manager, rng: random.Random):
        self.world = world
        self.traffic_manager = traffic_manager
        self.rng = rng
        self.blueprints = world.get_blueprint_library()
        self.map = world.get_map()
        self.spawn_points = list(self.map.get_spawn_points())

    def spawn(self, scenario: ScenarioConfig) -> Tuple[carla.Vehicle, List[carla.Vehicle], Dict[int, ActorSpec]]:
        if not self.spawn_points:
            raise RuntimeError("Selected CARLA map has no spawn points.")

        ego_spawn = self._sample_spawn_spec(scenario.ego.spawn)
        ego = self._spawn_vehicle(
            ego_spawn,
            scenario.ego.vehicle_filter,
            role_name="ego",
            used_indices=set(),
        )
        ego.set_autopilot(False)
        ego_waypoint = self.map.get_waypoint(
            ego.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        used_indices = {ego_spawn.index or 0}
        actors: List[carla.Vehicle] = []
        actor_specs_by_id: Dict[int, ActorSpec] = {}

        for spec in scenario.actors:
            if self.rng.random() > spec.spawn_probability:
                print(f"[INFO] Skipped optional actor {spec.id}")
                continue

            spawn_spec = self._sample_spawn_spec(spec.spawn)
            vehicle = self._try_spawn_actor_spec(
                ego_waypoint,
                spawn_spec,
                spawn_spec.vehicle_filter,
                role_name=spec.id,
            )
            if vehicle is None:
                print(f"[WARN] Skipped actor {spec.id}: requested spawn failed")
                continue
            vehicle.set_autopilot(spec.autopilot, self.traffic_manager.get_port())
            actors.append(vehicle)
            actor_specs_by_id[vehicle.id] = spec
            if spawn_spec.index is not None:
                used_indices.add(spawn_spec.index)

        background_needed = 0
        if scenario.background_spawn.mode != "none":
            background_needed = max(0, scenario.neighbor_count - len(actors))
        actors.extend(self._spawn_background(background_needed, used_indices, ego, scenario.background_spawn))

        return ego, actors, actor_specs_by_id

    def _sample_spawn_spec(self, spec: SpawnSpec) -> SpawnSpec:
        index = spec.index
        if spec.index_choices:
            index = self.rng.choice(spec.index_choices)

        offset = spec.offset
        if len(spec.offset_range) == 2:
            low, high = spec.offset_range
            if low > high:
                low, high = high, low
            offset = self.rng.uniform(low, high)

        lane_offset = spec.lane_offset
        if spec.lane_offset_choices:
            lane_offset = self.rng.choice(spec.lane_offset_choices)

        return SpawnSpec(
            index=index,
            index_choices=list(spec.index_choices),
            offset=offset,
            offset_range=list(spec.offset_range),
            lane_offset=lane_offset,
            lane_offset_choices=list(spec.lane_offset_choices),
            vehicle_filter=spec.vehicle_filter,
        )

    def _spawn_background(self, count: int, used_indices: set, ego_vehicle, background_config) -> List[carla.Vehicle]:
        vehicles = []
        ego_waypoint = self.map.get_waypoint(
            ego_vehicle.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if ego_waypoint is not None and background_config.mode in ("ego_local", "ego_local_random"):
            local_specs = self._ego_local_background_specs(background_config, count, ego_waypoint)
            for spec in local_specs:
                if len(vehicles) >= count:
                    break
                vehicle = self._try_spawn_near_waypoint(ego_waypoint, spec, "vehicle.*", role_name="background")
                if vehicle is not None:
                    vehicle.set_autopilot(True, self.traffic_manager.get_port())
                    vehicles.append(vehicle)

        if len(vehicles) < count and background_config.allow_global_fallback:
            vehicles.extend(self._spawn_global_background(count - len(vehicles), used_indices))

        if len(vehicles) < count:
            print(
                f"[WARN] Requested {count} ego-local background NPCs, "
                f"spawned {len(vehicles)}. Skipping far random fallback."
            )
        print(f"[INFO] Spawned {len(vehicles)} ego-local background NPC vehicles")
        return vehicles

    def _ego_local_background_specs(self, background_config, count: int, anchor_waypoint=None) -> List[SpawnSpec]:
        if background_config.mode == "ego_local_random":
            return self._ego_local_random_background_specs(background_config, count, anchor_waypoint)

        specs = []
        offsets = list(background_config.forward_offsets) + list(background_config.rear_offsets)

        for lane_offset in background_config.lane_offsets:
            for offset in offsets:
                jitter = self.rng.uniform(-background_config.jitter, background_config.jitter)
                specs.append(
                    SpawnSpec(
                        index=None,
                        offset=offset + jitter,
                        lane_offset=lane_offset,
                    )
                )

        same_lane_front = [spec for spec in specs if spec.lane_offset == 0 and spec.offset > 0.0]
        same_lane_rear = [spec for spec in specs if spec.lane_offset == 0 and spec.offset < 0.0]
        adjacent_front = [spec for spec in specs if spec.lane_offset != 0 and spec.offset > 0.0]
        adjacent_alongside = []
        adjacent_rear = [spec for spec in specs if spec.lane_offset != 0 and spec.offset < 0.0]

        for lane_offset in background_config.lane_offsets:
            if lane_offset == 0:
                continue
            for offset in background_config.alongside_offsets:
                adjacent_alongside.append(
                    SpawnSpec(
                        index=None,
                        offset=offset + self.rng.uniform(-background_config.jitter, background_config.jitter),
                        lane_offset=lane_offset,
                    )
                )

        for group in (same_lane_front, same_lane_rear, adjacent_front, adjacent_alongside, adjacent_rear):
            self.rng.shuffle(group)

        return same_lane_front + same_lane_rear + adjacent_alongside + adjacent_front + adjacent_rear

    def _ego_local_random_background_specs(self, background_config, count: int, anchor_waypoint=None) -> List[SpawnSpec]:
        # Filter to lane offsets that are actually reachable from the anchor waypoint.
        # Avoids generating specs for lanes that don't exist, which would all silently fail.
        if anchor_waypoint is not None:
            available = [
                lo for lo in background_config.lane_offsets
                if lo == 0 or self._apply_lane_offset(anchor_waypoint, lo) is not None
            ]
            lane_offsets = available if available else background_config.lane_offsets
        else:
            lane_offsets = background_config.lane_offsets

        specs = []
        used_offsets_by_lane = {lane_offset: [] for lane_offset in lane_offsets}
        attempts = 0

        while len(specs) < count and attempts < background_config.max_attempts:
            attempts += 1
            lane_offset = self.rng.choice(lane_offsets)
            offset = self._sample_local_offset(background_config, lane_offset)

            used_offsets = used_offsets_by_lane.setdefault(lane_offset, [])
            if any(abs(offset - used) < background_config.min_spacing for used in used_offsets):
                continue

            used_offsets.append(offset)
            specs.append(
                SpawnSpec(
                    index=None,
                    offset=offset,
                    lane_offset=lane_offset,
                )
            )

        if len(specs) < count:
            print(
                f"[WARN] Generated {len(specs)} local background specs for requested {count}. "
                f"Consider widening background_spawn ranges."
            )

        specs.sort(key=lambda spec: (abs(spec.offset), spec.lane_offset))
        return specs

    def _sample_local_offset(self, background_config, lane_offset: int) -> float:
        if lane_offset == 0:
            choose_front = self.rng.random() < 0.65
            low, high = background_config.front_range if choose_front else background_config.rear_range
        else:
            low, high = background_config.adjacent_range

        if low > high:
            low, high = high, low
        return self.rng.uniform(low, high)

    def _spawn_global_background(self, count: int, used_indices: set) -> List[carla.Vehicle]:
        vehicles = []
        candidate_indices = [i for i in range(len(self.spawn_points)) if i not in used_indices]
        self.rng.shuffle(candidate_indices)

        for idx in candidate_indices:
            if len(vehicles) >= count:
                break
            vehicle = self._try_spawn_vehicle(SpawnSpec(index=idx), "vehicle.*", role_name="background")
            if vehicle is not None:
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                vehicles.append(vehicle)

        return vehicles

    def _spawn_vehicle(self, spec: SpawnSpec, vehicle_filter: str, role_name: str, used_indices: set):
        preferred = self._try_spawn_vehicle(spec, vehicle_filter, role_name)
        if preferred is not None:
            print(f"[INFO] Spawned {role_name}: {preferred.id}")
            return preferred

        indices = [i for i in range(len(self.spawn_points)) if i not in used_indices]
        self.rng.shuffle(indices)
        for idx in indices:
            fallback = self._try_spawn_vehicle(SpawnSpec(index=idx), vehicle_filter, role_name)
            if fallback is not None:
                print(f"[WARN] Preferred spawn failed for {role_name}; used spawn index {idx}")
                return fallback

        raise RuntimeError(f"Failed to spawn vehicle for role {role_name}.")

    def _try_spawn_actor_spec(self, ego_waypoint, spec: SpawnSpec, vehicle_filter: str, role_name: str):
        if ego_waypoint is not None:
            vehicle = self._try_spawn_near_waypoint(ego_waypoint, spec, vehicle_filter, role_name)
        else:
            vehicle = self._try_spawn_vehicle(spec, vehicle_filter, role_name)
        if vehicle is None:
            return None
        print(f"[INFO] Spawned {role_name}: {vehicle.id}")
        return vehicle

    def _try_spawn_vehicle(self, spec: SpawnSpec, vehicle_filter: str, role_name: str):
        blueprint = self._choose_blueprint(vehicle_filter)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", role_name)

        transform = self._resolve_transform(spec)
        return self.world.try_spawn_actor(blueprint, transform)

    def _try_spawn_near_waypoint(self, anchor_waypoint, spec: SpawnSpec, vehicle_filter: str, role_name: str):
        blueprint = self._choose_blueprint(vehicle_filter)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", role_name)

        waypoint = self._apply_lane_offset(anchor_waypoint, spec.lane_offset)
        if waypoint is None:
            return None

        for offset_delta in (0.0, 4.0, -4.0, 8.0, -8.0):
            candidate_offset = spec.offset + offset_delta
            candidate_waypoint = waypoint
            if abs(candidate_offset) > 1e-6:
                candidates = candidate_waypoint.next(abs(candidate_offset)) if candidate_offset >= 0 else candidate_waypoint.previous(abs(candidate_offset))
                if not candidates:
                    continue
                candidate_waypoint = candidates[0]

            if not self._is_valid_ego_local_candidate(anchor_waypoint, candidate_waypoint, spec, candidate_offset):
                continue

            transform = candidate_waypoint.transform
            transform.location.z += 0.4
            vehicle = self.world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                if self._is_valid_ego_local_actor(anchor_waypoint, vehicle):
                    return vehicle
                vehicle.destroy()

        return None

    def _is_valid_ego_local_candidate(self, anchor_waypoint, candidate_waypoint, spec: SpawnSpec, candidate_offset: float) -> bool:
        anchor_transform = anchor_waypoint.transform
        anchor_loc = anchor_transform.location
        forward = anchor_transform.get_forward_vector()
        candidate_loc = candidate_waypoint.transform.location

        dx = candidate_loc.x - anchor_loc.x
        dy = candidate_loc.y - anchor_loc.y
        longitudinal = dx * forward.x + dy * forward.y
        lateral = forward.x * dy - forward.y * dx

        lane_width = anchor_waypoint.lane_width or 3.5
        expected_lateral = abs(spec.lane_offset) * lane_width
        longitudinal_tolerance = 18.0
        lateral_tolerance = max(2.0, 0.75 * lane_width)

        if abs(longitudinal - candidate_offset) > longitudinal_tolerance:
            return False
        if abs(abs(lateral) - expected_lateral) > lateral_tolerance:
            return False
        if abs(longitudinal) > 160.0 or abs(lateral) > 2.2 * lane_width:
            return False
        return True

    def _is_valid_ego_local_actor(self, anchor_waypoint, actor) -> bool:
        anchor_transform = anchor_waypoint.transform
        anchor_loc = anchor_transform.location
        forward = anchor_transform.get_forward_vector()
        actor_loc = actor.get_transform().location

        dx = actor_loc.x - anchor_loc.x
        dy = actor_loc.y - anchor_loc.y
        longitudinal = dx * forward.x + dy * forward.y
        lateral = forward.x * dy - forward.y * dx
        lane_width = anchor_waypoint.lane_width or 3.5

        return abs(longitudinal) <= 160.0 and abs(lateral) <= 2.2 * lane_width

    def _choose_blueprint(self, vehicle_filter: str):
        candidates = list(self.blueprints.filter(vehicle_filter))
        if not candidates:
            candidates = list(self.blueprints.filter("vehicle.*"))
        return self.rng.choice(candidates)

    def _resolve_transform(self, spec: SpawnSpec):
        index = spec.index if spec.index is not None else 0
        base_transform = self.spawn_points[index % len(self.spawn_points)]
        waypoint = self.map.get_waypoint(
            base_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return base_transform

        waypoint = self._apply_lane_offset(waypoint, spec.lane_offset)
        if waypoint is None:
            return base_transform
        if abs(spec.offset) > 1e-6:
            candidates = waypoint.next(abs(spec.offset)) if spec.offset >= 0 else waypoint.previous(abs(spec.offset))
            if candidates:
                waypoint = candidates[0]

        transform = waypoint.transform
        transform.location.z += 0.4
        return transform

    def _apply_lane_offset(self, waypoint, lane_offset: int):
        current = waypoint
        steps = abs(lane_offset)
        for _ in range(steps):
            nxt = current.get_left_lane() if lane_offset > 0 else current.get_right_lane()
            if nxt is None or nxt.lane_type != carla.LaneType.Driving:
                return None
            if current.lane_id * nxt.lane_id <= 0:
                return None
            current = nxt
        return current
