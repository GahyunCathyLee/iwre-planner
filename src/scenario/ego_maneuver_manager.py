class EgoManeuverManager:
    """Placeholder for lateral ego maneuvers.

    The first scenario milestone keeps ego steering at zero while the IDM
    planner owns longitudinal control.
    """

    def __init__(self, config=None):
        self.config = config or {}

    def tick(self, step, ego_vehicle, neighbor_vehicles):
        return {
            "steer": 0.0,
            "lane_change_intended": False,
            "lane_change_started": False,
            "lane_change_completed": False,
        }

