import random
from typing import Any, Dict, Optional


def generate_random_scenario_dict(template: Dict[str, Any], seed_override: Optional[int] = None) -> Dict[str, Any]:
    seed = int(seed_override if seed_override is not None else template.get("seed", 42))
    rng = random.Random(seed)
    behavior_mix = template.get("behavior_mix", ["constant_speed", "sudden_brake", "sudden_accel"])

    actors = _slot_candidate_actors()
    scripted_count = rng.randint(1, 3)
    for idx in range(scripted_count):
        behavior_type = rng.choice(behavior_mix)
        actors.append(
            {
                "id": f"random_scripted_{idx}",
                "role": "scripted",
                "spawn": {
                    "index": 10 + idx * 3,
                    "offset": rng.uniform(8.0, 35.0),
                    "lane_offset": rng.choice([-1, 0, 1]),
                },
                "autopilot": behavior_type == "constant_speed",
                "cut_in_actor": behavior_type.startswith("cut_in"),
                "behavior": {
                    "type": behavior_type,
                    "start_step": rng.randint(80, 220),
                    "duration_steps": rng.randint(30, 90),
                    "target_speed": rng.uniform(6.0, 14.0),
                    "brake": rng.uniform(0.35, 0.75),
                    "throttle": rng.uniform(0.35, 0.65),
                },
            }
        )

    return {
        "scenario_id": int(template.get("generated_scenario_id", template.get("scenario_id", 90))),
        "scenario_name": f"random_seed_{seed}",
        "map": template.get("map", "Town04"),
        "fps": template.get("fps", 10.0),
        "steps": template.get("steps", 200),
        "seed": seed,
        "neighbor_count": 0,
        "ego": template.get("ego", {"spawn": {"index": 0}}),
        "actors": actors,
        "background_spawn": {"mode": "none"},
        "noise": template.get("noise", {"mode": "medium"}),
        "ego_maneuver": template.get("ego_maneuver", {"mode": "lane_keep"}),
        "traffic_manager": template.get("traffic_manager", {}),
    }


def _slot_candidate_actors():
    return [
        _slot_actor("preceding_candidate", [42, 80], 0, 0.62),
        _slot_actor("following_candidate", [-35, -18], 0, 0.54),
        _slot_actor("leftPreceding_candidate", [25, 65], 1, 0.73),
        _slot_actor("leftAlongside_candidate", [-6, 6], 1, 0.61),
        _slot_actor("leftFollowing_candidate", [-38, -18], 1, 0.49),
        _slot_actor("rightPreceding_candidate", [25, 65], -1, 0.69),
        _slot_actor("rightAlongside_candidate", [-6, 6], -1, 0.57),
        _slot_actor("rightFollowing_candidate", [-38, -18], -1, 0.45),
    ]


def _slot_actor(actor_id, offset_range, lane_offset, spawn_probability):
    return {
        "id": actor_id,
        "role": "slot_candidate",
        "spawn": {"index": 0, "offset_range": offset_range, "lane_offset": lane_offset},
        "autopilot": True,
        "spawn_probability": spawn_probability,
        "behavior": {"type": "none"},
    }
