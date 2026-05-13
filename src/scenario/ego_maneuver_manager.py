import math

import carla


class EgoManeuverManager:
    def __init__(self, world, config):
        self.world = world
        self.map = world.get_map()
        self.config = config
        self.started = False
        self.completed = False
        self.start_step = None
        self.wait_steps = 0
        self.rejected_gap_count = 0
        self.target_front_gap = None
        self.target_rear_gap = None
        self.target_rear_ttc = None

    def compute_steer(self, ego_vehicle, neighbors, step: int) -> float:
        mode = self.config.mode
        intended = mode in ("scripted_lane_change", "conditional_lane_change")

        if mode == "scripted_lane_change" and step >= self.config.start_step and not self.completed:
            self._start_if_needed(step)
        elif mode == "conditional_lane_change" and not self.started and not self.completed:
            if step >= self.config.start_step:
                safe = self._evaluate_gap(ego_vehicle, neighbors)
                if safe:
                    self._start_if_needed(step)
                else:
                    self.wait_steps += 1
                    self.rejected_gap_count += 1
        else:
            self._evaluate_gap(ego_vehicle, neighbors)

        if self.started and not self.completed:
            if step - self.start_step >= self.config.duration_steps:
                self.completed = True
            else:
                return self._lane_change_steer(ego_vehicle)

        if intended and not self.completed:
            return self._lane_keep_steer(ego_vehicle)

        return self._lane_keep_steer(ego_vehicle)

    def log_state(self):
        return {
            "lane_change_intended": self.config.mode in ("scripted_lane_change", "conditional_lane_change"),
            "lane_change_started": self.started,
            "lane_change_completed": self.completed,
            "lane_change_start_step": self.start_step if self.start_step is not None else "",
            "lane_change_wait_steps": self.wait_steps,
            "rejected_gap_count": self.rejected_gap_count,
            "target_front_gap": self.target_front_gap,
            "target_rear_gap": self.target_rear_gap,
            "target_rear_ttc": self.target_rear_ttc,
        }

    def _start_if_needed(self, step: int):
        if not self.started:
            self.started = True
            self.start_step = step

    def _lane_change_steer(self, ego_vehicle) -> float:
        target_wp = self._target_lane_waypoint(ego_vehicle)
        if target_wp is None:
            return self._direction_sign() * 0.12
        return self._steer_to_waypoint(ego_vehicle, target_wp, gain=0.08, max_abs=0.35)

    def _lane_keep_steer(self, ego_vehicle) -> float:
        wp = self.map.get_waypoint(
            ego_vehicle.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is None:
            return 0.0
        return self._steer_to_waypoint(ego_vehicle, wp, gain=0.05, max_abs=0.18)

    def _steer_to_waypoint(self, ego_vehicle, waypoint, gain: float, max_abs: float) -> float:
        ego_transform = ego_vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()
        target = waypoint.transform.location
        dx = target.x - ego_loc.x
        dy = target.y - ego_loc.y
        lateral_error = ego_forward.x * dy - ego_forward.y * dx
        steer = gain * lateral_error
        return max(-max_abs, min(max_abs, steer))

    def _target_lane_waypoint(self, ego_vehicle):
        wp = self.map.get_waypoint(
            ego_vehicle.get_transform().location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if wp is None:
            return None
        return wp.get_left_lane() if self.config.direction == "left" else wp.get_right_lane()

    def _evaluate_gap(self, ego_vehicle, neighbors) -> bool:
        target_wp = self._target_lane_waypoint(ego_vehicle)
        if target_wp is None or target_wp.lane_type != carla.LaneType.Driving:
            self.target_front_gap = 0.0
            self.target_rear_gap = 0.0
            self.target_rear_ttc = 0.0
            return False

        ego_transform = ego_vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_forward = ego_transform.get_forward_vector()
        ego_speed = _speed(ego_vehicle)
        front_gap = math.inf
        rear_gap = math.inf
        rear_ttc = math.inf

        for actor in neighbors:
            if actor is None or not actor.is_alive:
                continue
            actor_loc = actor.get_transform().location
            actor_wp = self.map.get_waypoint(actor_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if actor_wp is None:
                continue
            if actor_wp.road_id != target_wp.road_id or actor_wp.lane_id != target_wp.lane_id:
                continue
            rel = actor_loc - ego_loc
            longitudinal = rel.x * ego_forward.x + rel.y * ego_forward.y + rel.z * ego_forward.z
            distance = ego_loc.distance(actor_loc)
            if longitudinal >= 0.0:
                front_gap = min(front_gap, distance)
            else:
                rear_gap = min(rear_gap, distance)
                closing_speed = _speed(actor) - ego_speed
                if closing_speed > 0.1:
                    rear_ttc = min(rear_ttc, distance / closing_speed)

        self.target_front_gap = front_gap
        self.target_rear_gap = rear_gap
        self.target_rear_ttc = rear_ttc

        return (
            front_gap > self.config.min_front_gap
            and rear_gap > self.config.min_rear_gap
            and rear_ttc > self.config.min_rear_ttc
        )

    def _direction_sign(self) -> float:
        return -1.0 if self.config.direction == "left" else 1.0


def _speed(vehicle) -> float:
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
