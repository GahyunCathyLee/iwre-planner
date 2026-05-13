import math


class IDMPlanner:
    """
    Intelligent Driver Model planner.

    B1 baseline:
    - No uncertainty
    - No interaction score
    - No risk modulation
    - Ego reacts only to the front vehicle distance and relative speed
    """

    def __init__(
        self,
        desired_speed=13.9,
        min_distance=2.0,
        time_headway=1.5,
        max_accel=1.5,
        comfort_decel=2.0,
        delta=4.0,
    ):
        self.v0 = desired_speed
        self.s0 = min_distance
        self.T = time_headway
        self.a_max = max_accel
        self.b = comfort_decel
        self.delta = delta

    def compute_acceleration(
        self,
        ego_speed,
        front_distance=None,
        relative_speed=0.0,
    ):
        """
        Compute IDM acceleration.

        Parameters
        ----------
        ego_speed : float
            Ego speed in m/s.
        front_distance : float or None
            Distance to the front vehicle in meters.
            If None, ego performs free-road driving.
        relative_speed : float
            ego_speed - front_vehicle_speed.
            Positive means ego is approaching the front vehicle.

        Returns
        -------
        float
            Desired longitudinal acceleration in m/s^2.
        """

        ego_speed = max(ego_speed, 0.0)

        free_road_term = (ego_speed / max(self.v0, 1e-6)) ** self.delta

        if front_distance is None or front_distance <= 0.0:
            interaction_term = 0.0
        else:
            desired_gap = self.compute_desired_gap(
                ego_speed=ego_speed,
                relative_speed=relative_speed,
            )

            interaction_term = (desired_gap / max(front_distance, 1e-6)) ** 2

        acceleration = self.a_max * (1.0 - free_road_term - interaction_term)

        return acceleration

    def compute_desired_gap(self, ego_speed, relative_speed):
        """
        IDM desired dynamic gap:
        s* = s0 + vT + v * delta_v / (2 * sqrt(a_max * b))
        """

        braking_term = (
            ego_speed * relative_speed
            / (2.0 * math.sqrt(max(self.a_max * self.b, 1e-6)))
        )

        desired_gap = self.s0 + ego_speed * self.T + braking_term

        return max(self.s0, desired_gap)