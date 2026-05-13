import carla


class BehaviorManager:
    def __init__(self, world, traffic_manager, scripted_specs_by_actor_id):
        self.world = world
        self.traffic_manager = traffic_manager
        self.scripted_specs_by_actor_id = scripted_specs_by_actor_id

    def apply(self, step: int):
        for actor_id, spec in self.scripted_specs_by_actor_id.items():
            actor = self._get_actor(actor_id)
            if actor is None or not actor.is_alive:
                continue

            behavior = spec.behavior or {}
            behavior_type = behavior.get("type", "none")
            if behavior_type in ("none", ""):
                continue
            start = int(behavior.get("start_step", 0))
            duration = int(behavior.get("duration_steps", 60))
            active = start <= step < start + duration

            if behavior_type == "sudden_brake" and active:
                actor.set_autopilot(False)
                actor.apply_control(carla.VehicleControl(throttle=0.0, brake=float(behavior.get("brake", 0.65)), steer=0.0))
            elif behavior_type == "sudden_accel" and active:
                actor.set_autopilot(False)
                actor.apply_control(carla.VehicleControl(throttle=float(behavior.get("throttle", 0.55)), brake=0.0, steer=0.0))
            elif behavior_type in ("cut_in_left", "cut_in_right") and active:
                actor.set_autopilot(False)
                direction = -1.0 if behavior_type == "cut_in_left" else 1.0
                steer = direction * float(behavior.get("steer", 0.18))
                actor.apply_control(carla.VehicleControl(throttle=float(behavior.get("throttle", 0.35)), brake=0.0, steer=steer))
            elif behavior_type == "constant_speed" and active:
                actor.set_autopilot(False)
                actor.apply_control(carla.VehicleControl(throttle=float(behavior.get("throttle", 0.25)), brake=0.0, steer=0.0))
            elif behavior_type != "constant_speed":
                actor.set_autopilot(True, self.traffic_manager.get_port())

    def _get_actor(self, actor_id):
        return self.world.get_actors().find(actor_id)
