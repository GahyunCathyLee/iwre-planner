from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only in missing dependency envs.
    yaml = None

from scenario.random_scenario_generator import generate_random_scenario_dict
from scenario.scenario_config import ActorConfig, ScenarioConfig, SpawnConfig


def _require_mapping(value, field_name):
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a mapping.")
    return value


def _load_yaml(path):
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load scenario files. "
            "Run with the iwre-planner conda environment."
        )

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Scenario file must contain a mapping: {path}")

    return data


def _parse_spawn(data):
    data = _require_mapping(data, "spawn")
    return SpawnConfig(
        index=int(data.get("index", 0)),
        offset=float(data.get("offset", 0.0)),
        lane_offset=int(data.get("lane_offset", 0)),
    )


def _parse_actor(data):
    data = _require_mapping(data, "actor")
    if "id" not in data:
        raise ValueError("Every scenario actor requires an id.")

    return ActorConfig(
        id=str(data["id"]),
        role=str(data.get("role", "npc")),
        spawn=_parse_spawn(data.get("spawn", {})),
        autopilot=bool(data.get("autopilot", True)),
        spawn_probability=float(data.get("spawn_probability", 1.0)),
        behavior=_require_mapping(data.get("behavior", {}), "behavior"),
        cut_in_actor=bool(data.get("cut_in_actor", False)),
    )


def load_scenario(path, seed_override=None):
    path = Path(path)
    data = _load_yaml(path)

    if data.get("scenario_name") == "random_batch" or data.get("type") == "random_batch":
        data = generate_random_scenario_dict(data, seed_override)

    if "scenario_id" not in data:
        raise ValueError(f"scenario_id is required: {path}")
    if "scenario_name" not in data:
        raise ValueError(f"scenario_name is required: {path}")

    ego = _require_mapping(data.get("ego", {}), "ego")
    actors = [_parse_actor(actor) for actor in data.get("actors", []) or []]

    scenario = ScenarioConfig(
        scenario_id=int(data["scenario_id"]),
        scenario_name=str(data["scenario_name"]),
        map=str(data.get("map", "Town04")),
        fps=float(data.get("fps", 20.0)),
        steps=int(data.get("steps", 500)),
        seed=int(data.get("seed", 42)),
        type=str(data.get("type", "fixed")),
        ego_spawn=_parse_spawn(ego.get("spawn", {})),
        background_spawn=_require_mapping(data.get("background_spawn", {}), "background_spawn"),
        noise=_require_mapping(data.get("noise", {}), "noise"),
        ego_maneuver=_require_mapping(data.get("ego_maneuver", {}), "ego_maneuver"),
        actors=actors,
        raw=data,
    )

    if seed_override is not None:
        scenario.seed = int(seed_override)

    return scenario
