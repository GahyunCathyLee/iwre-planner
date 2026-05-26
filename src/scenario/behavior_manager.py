import carla


class BehaviorManager:
    """Minimal scripted behavior runner for the first scenario milestone."""

    def __init__(self, actor_entries):
        self.actor_entries = actor_entries

    def tick(self, step):
        for actor_config, vehicle in self.actor_entries:
            if not vehicle.is_alive:
                continue

            behavior = actor_config.behavior or {}
            behavior_type = behavior.get("type", "none")
            if behavior_type in ("none", "constant_speed"):
                continue

            start_step = int(behavior.get("start_step", 0))
            duration_steps = int(behavior.get("duration_steps", 0))
            active = step >= start_step and (
                duration_steps <= 0 or step < start_step + duration_steps
            )
            if not active:
                continue

            control = carla.VehicleControl()
            if behavior_type == "sudden_brake":
                control.brake = float(behavior.get("brake", 0.7))
            elif behavior_type == "sudden_accel":
                control.throttle = float(behavior.get("throttle", 0.5))
            elif behavior_type in ("cut_in_left", "cut_in_right"):
                direction = -1.0 if behavior_type == "cut_in_left" else 1.0
                control.throttle = float(behavior.get("throttle", 0.35))
                control.steer = direction * abs(float(behavior.get("steer", 0.15)))
            else:
                continue

            vehicle.apply_control(control)

