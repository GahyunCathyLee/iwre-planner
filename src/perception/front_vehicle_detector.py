import math
import carla


def get_speed(vehicle):
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def vector_dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


def get_front_vehicle(
    ego_vehicle,
    candidate_vehicles,
    max_distance=80.0,
    lane_check=True,
):
    """
    Find the closest vehicle in front of the ego vehicle.

    For B1, this uses:
    - ego forward vector
    - relative position
    - optional same-lane check using CARLA waypoint lane_id

    Returns
    -------
    front_vehicle : carla.Vehicle or None
    front_distance : float or None
    front_speed : float or None
    """

    world = ego_vehicle.get_world()
    carla_map = world.get_map()

    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    ego_forward = ego_transform.get_forward_vector()

    ego_waypoint = carla_map.get_waypoint(
        ego_location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )

    closest_vehicle = None
    closest_distance = None
    closest_speed = None

    for vehicle in candidate_vehicles:
        if vehicle is None:
            continue

        if vehicle.id == ego_vehicle.id:
            continue

        if not vehicle.is_alive:
            continue

        vehicle_location = vehicle.get_transform().location

        relative_vector = vehicle_location - ego_location
        longitudinal_distance = vector_dot(relative_vector, ego_forward)

        # Vehicle is behind ego
        if longitudinal_distance <= 0.0:
            continue

        distance = ego_location.distance(vehicle_location)

        if distance > max_distance:
            continue

        if lane_check:
            vehicle_waypoint = carla_map.get_waypoint(
                vehicle_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            if vehicle_waypoint is None or ego_waypoint is None:
                continue

            same_road = vehicle_waypoint.road_id == ego_waypoint.road_id
            same_lane = vehicle_waypoint.lane_id == ego_waypoint.lane_id

            if not (same_road and same_lane):
                continue

        if closest_distance is None or distance < closest_distance:
            closest_vehicle = vehicle
            closest_distance = distance
            closest_speed = get_speed(vehicle)

    return closest_vehicle, closest_distance, closest_speed