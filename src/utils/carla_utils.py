import random
import carla


def set_synchronous_mode(world, traffic_manager, fps):
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(settings)

    traffic_manager.set_synchronous_mode(True)


def reset_world_settings(world, traffic_manager):
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    traffic_manager.set_synchronous_mode(False)


def spawn_ego_vehicle(world, blueprint_library, spawn_points, vehicle_filter):
    ego_bp = blueprint_library.filter(vehicle_filter)[0]

    if ego_bp.has_attribute("role_name"):
        ego_bp.set_attribute("role_name", "ego")

    ego_spawn_point = spawn_points[0]
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)

    if ego_vehicle is None:
        raise RuntimeError("Failed to spawn ego vehicle.")

    print(f"[INFO] Ego vehicle spawned: {ego_vehicle.id}")

    return ego_vehicle


def spawn_npc_vehicles(
    world,
    blueprint_library,
    spawn_points,
    num_vehicles,
    traffic_manager,
):
    vehicle_bps = blueprint_library.filter("vehicle.*")
    npc_vehicles = []

    spawn_points = list(spawn_points)
    random.shuffle(spawn_points)

    for spawn_point in spawn_points:
        if len(npc_vehicles) >= num_vehicles:
            break

        bp = random.choice(vehicle_bps)

        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "npc")

        vehicle = world.try_spawn_actor(bp, spawn_point)

        if vehicle is not None:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            npc_vehicles.append(vehicle)
            print(f"[INFO] NPC vehicle spawned: {vehicle.id}")

    print(f"[INFO] Total NPC vehicles: {len(npc_vehicles)}")

    return npc_vehicles


def cleanup_actors(client, actors):
    print("[INFO] Cleaning up actors...")

    valid_actors = [actor for actor in actors if actor is not None]

    if valid_actors:
        client.apply_batch([
            carla.command.DestroyActor(actor) for actor in valid_actors
        ])

    print("[INFO] Cleanup complete.")