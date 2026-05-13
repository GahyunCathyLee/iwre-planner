import carla


class VehicleController:
    """
    Converts desired longitudinal acceleration into CARLA VehicleControl.
    Steering is currently fixed to 0 for B1 minimum implementation.

    This is intentionally simple.
    Later, this can be replaced by waypoint-following or lane-keeping control.
    """

    def __init__(
        self,
        max_throttle=0.75,
        max_brake=0.8,
        accel_to_throttle_scale=1.5,
        decel_to_brake_scale=4.0,
    ):
        self.max_throttle = max_throttle
        self.max_brake = max_brake
        self.accel_to_throttle_scale = accel_to_throttle_scale
        self.decel_to_brake_scale = decel_to_brake_scale

    def acceleration_to_control(self, acceleration, steer=0.0):
        control = carla.VehicleControl()

        control.steer = float(max(-1.0, min(1.0, steer)))

        if acceleration >= 0.0:
            throttle = acceleration / max(self.accel_to_throttle_scale, 1e-6)
            control.throttle = float(max(0.0, min(self.max_throttle, throttle)))
            control.brake = 0.0
        else:
            brake = abs(acceleration) / max(self.decel_to_brake_scale, 1e-6)
            control.throttle = 0.0
            control.brake = float(max(0.0, min(self.max_brake, brake)))

        control.hand_brake = False
        control.reverse = False
        control.manual_gear_shift = False

        return control