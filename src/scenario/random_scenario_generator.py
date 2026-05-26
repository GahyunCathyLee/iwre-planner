import copy
import random


DEFAULT_RANDOM_SLOTS = [
    ("same_lane_front", 32.0, 0),
    ("same_lane_far_front", 55.0, 0),
    ("same_lane_rear", -24.0, 0),
    ("left_front", 34.0, 1),
    ("left_alongside", 2.0, 1),
    ("left_rear", -26.0, 1),
    ("right_front", 34.0, -1),
    ("right_alongside", 2.0, -1),
    ("right_rear", -26.0, -1),
]


def generate_random_scenario_dict(template, seed_override=None):
    data = copy.deepcopy(template)
    seed = int(seed_override if seed_override is not None else data.get("seed", 42))
    rng = random.Random(seed)
    behavior_mix = list(data.get("behavior_mix", ["constant_speed"]))

    actors = []
    for slot_name, base_offset, lane_offset in DEFAULT_RANDOM_SLOTS:
        if rng.random() > 0.75:
            continue

        behavior_type = rng.choice(behavior_mix) if behavior_mix else "constant_speed"
        behavior = _behavior_for_type(behavior_type, rng)
        offset = base_offset + rng.uniform(-5.0, 5.0)

        actors.append({
            "id": f"random_{slot_name}",
            "role": "slot_candidate",
            "spawn": {
                "index": data.get("ego", {}).get("spawn", {}).get("index", 0),
                "offset": round(offset, 2),
                "lane_offset": lane_offset,
            },
            "autopilot": behavior["type"] in ("none", "constant_speed"),
            "spawn_probability": 1.0,
            "behavior": behavior,
            "cut_in_actor": behavior["type"].startswith("cut_in"),
        })

    if not actors:
        actors.append({
            "id": "random_same_lane_front",
            "role": "slot_candidate",
            "spawn": {
                "index": data.get("ego", {}).get("spawn", {}).get("index", 0),
                "offset": 35.0,
                "lane_offset": 0,
            },
            "autopilot": True,
            "spawn_probability": 1.0,
            "behavior": {"type": "none"},
        })

    data["seed"] = seed
    data["actors"] = actors
    return data


def _behavior_for_type(behavior_type, rng):
    if behavior_type == "sudden_brake":
        return {
            "type": "sudden_brake",
            "start_step": rng.randint(90, 150),
            "duration_steps": rng.randint(50, 90),
            "brake": round(rng.uniform(0.55, 0.8), 2),
        }
    if behavior_type == "sudden_accel":
        return {
            "type": "sudden_accel",
            "start_step": rng.randint(90, 150),
            "duration_steps": rng.randint(50, 100),
            "throttle": round(rng.uniform(0.45, 0.7), 2),
        }
    if behavior_type in ("cut_in_left", "cut_in_right"):
        return {
            "type": behavior_type,
            "start_step": rng.randint(80, 140),
            "duration_steps": rng.randint(60, 100),
            "throttle": round(rng.uniform(0.3, 0.45), 2),
            "steer": round(rng.uniform(0.12, 0.2), 2),
        }
    if behavior_type == "constant_speed":
        return {"type": "constant_speed"}
    return {"type": "none"}

