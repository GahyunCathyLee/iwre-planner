from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SpawnSpec:
    index: Optional[int] = None
    index_choices: List[int] = field(default_factory=list)
    offset: float = 0.0
    offset_range: List[float] = field(default_factory=list)
    lane_offset: int = 0
    lane_offset_choices: List[int] = field(default_factory=list)
    vehicle_filter: str = "vehicle.*"


@dataclass
class ActorSpec:
    id: str
    role: str = "background"
    spawn: SpawnSpec = field(default_factory=SpawnSpec)
    autopilot: bool = True
    spawn_probability: float = 1.0
    behavior: Dict[str, Any] = field(default_factory=dict)
    cut_in_actor: bool = False


@dataclass
class EgoSpec:
    spawn: SpawnSpec = field(default_factory=SpawnSpec)
    vehicle_filter: str = "vehicle.tesla.model3"


@dataclass
class NoiseConfig:
    mode: str = "medium"
    mode_choices: List[str] = field(default_factory=list)
    base_position_std: float = 0.25
    base_velocity_std: float = 0.10
    distance_factor: float = 0.015
    angle_factor: float = 0.40
    relative_motion_factor: float = 0.04
    occlusion_factor: float = 0.0
    scenario_bias: float = 0.0
    high_noise_actor_ids: List[str] = field(default_factory=list)
    high_noise_bias: float = 0.0


@dataclass
class EgoManeuverConfig:
    mode: str = "lane_keep"
    direction: str = "left"
    start_step: int = 120
    duration_steps: int = 80
    min_front_gap: float = 18.0
    min_rear_gap: float = 12.0
    min_rear_ttc: float = 4.0
    max_wait_steps: int = 250


@dataclass
class BackgroundSpawnConfig:
    mode: str = "ego_local"
    lane_offsets: List[int] = field(default_factory=lambda: [0, 1, -1])
    forward_offsets: List[float] = field(default_factory=lambda: [16.0, 24.0, 36.0, 50.0, 66.0, 84.0])
    rear_offsets: List[float] = field(default_factory=lambda: [-16.0, -26.0, -40.0, -58.0])
    alongside_offsets: List[float] = field(default_factory=lambda: [-6.0, 0.0, 6.0])
    front_range: List[float] = field(default_factory=lambda: [14.0, 90.0])
    rear_range: List[float] = field(default_factory=lambda: [-60.0, -14.0])
    adjacent_range: List[float] = field(default_factory=lambda: [-45.0, 90.0])
    min_spacing: float = 10.0
    max_attempts: int = 200
    jitter: float = 2.0
    allow_global_fallback: bool = False


@dataclass
class ScenarioConfig:
    scenario_id: int
    scenario_name: str = "unnamed_scenario"
    town: str = "Town04"
    fps: float = 10.0
    steps: int = 200
    seed: int = 42
    neighbor_count: int = 0
    ego: EgoSpec = field(default_factory=EgoSpec)
    actors: List[ActorSpec] = field(default_factory=list)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    ego_maneuver: EgoManeuverConfig = field(default_factory=EgoManeuverConfig)
    background_spawn: BackgroundSpawnConfig = field(default_factory=BackgroundSpawnConfig)
    traffic_manager: Dict[str, Any] = field(default_factory=dict)


def _spawn_from_dict(data: Optional[Dict[str, Any]]) -> SpawnSpec:
    data = data or {}
    return SpawnSpec(
        index=data.get("index"),
        index_choices=[int(value) for value in data.get("index_choices", [])],
        offset=float(data.get("offset", 0.0)),
        offset_range=[float(value) for value in data.get("offset_range", [])],
        lane_offset=int(data.get("lane_offset", 0)),
        lane_offset_choices=[int(value) for value in data.get("lane_offset_choices", [])],
        vehicle_filter=data.get("vehicle_filter", "vehicle.*"),
    )


def scenario_from_dict(data: Dict[str, Any]) -> ScenarioConfig:
    ego_data = data.get("ego", {})
    noise_data = data.get("noise", {})
    maneuver_data = data.get("ego_maneuver", {})
    background_data = data.get("background_spawn", {})
    actor_specs = []
    seed = int(data.get("seed", 42))
    noise_mode_choices = list(noise_data.get("mode_choices", []))
    noise_mode = noise_data.get("mode", "medium")
    if noise_mode_choices:
        import random

        noise_mode = random.Random(seed + 1009).choice(noise_mode_choices)

    for idx, actor_data in enumerate(data.get("actors", [])):
        behavior = actor_data.get("behavior", {}) or {}
        actor_specs.append(
            ActorSpec(
                id=actor_data.get("id", f"actor_{idx}"),
                role=actor_data.get("role", "background"),
                spawn=_spawn_from_dict(actor_data.get("spawn")),
                autopilot=bool(actor_data.get("autopilot", True)),
                spawn_probability=float(actor_data.get("spawn_probability", 1.0)),
                behavior=behavior,
                cut_in_actor=bool(actor_data.get("cut_in_actor", behavior.get("type", "").startswith("cut_in"))),
            )
        )

    return ScenarioConfig(
        scenario_id=int(data.get("scenario_id", -1)),
        scenario_name=data.get("scenario_name", data.get("name", "unnamed_scenario")),
        town=data.get("map", data.get("town", "Town04")),
        fps=float(data.get("fps", 10.0)),
        steps=int(data.get("steps", 200)),
        seed=seed,
        neighbor_count=int(data.get("neighbor_count", 0)),
        ego=EgoSpec(
            spawn=_spawn_from_dict(ego_data.get("spawn")),
            vehicle_filter=ego_data.get("vehicle_filter", "vehicle.tesla.model3"),
        ),
        actors=actor_specs,
        noise=NoiseConfig(
            mode=noise_mode,
            mode_choices=noise_mode_choices,
            base_position_std=float(noise_data.get("base_position_std", 0.25)),
            base_velocity_std=float(noise_data.get("base_velocity_std", 0.10)),
            distance_factor=float(noise_data.get("distance_factor", 0.015)),
            angle_factor=float(noise_data.get("angle_factor", 0.40)),
            relative_motion_factor=float(noise_data.get("relative_motion_factor", 0.04)),
            occlusion_factor=float(noise_data.get("occlusion_factor", 0.0)),
            scenario_bias=float(noise_data.get("scenario_bias", 0.0)),
            high_noise_actor_ids=list(noise_data.get("high_noise_actor_ids", [])),
            high_noise_bias=float(noise_data.get("high_noise_bias", 0.0)),
        ),
        ego_maneuver=EgoManeuverConfig(
            mode=maneuver_data.get("mode", "lane_keep"),
            direction=maneuver_data.get("direction", "left"),
            start_step=int(maneuver_data.get("start_step", 120)),
            duration_steps=int(maneuver_data.get("duration_steps", 80)),
            min_front_gap=float(maneuver_data.get("min_front_gap", 18.0)),
            min_rear_gap=float(maneuver_data.get("min_rear_gap", 12.0)),
            min_rear_ttc=float(maneuver_data.get("min_rear_ttc", 4.0)),
            max_wait_steps=int(maneuver_data.get("max_wait_steps", 250)),
        ),
        background_spawn=BackgroundSpawnConfig(
            mode=background_data.get("mode", "ego_local"),
            lane_offsets=[int(value) for value in background_data.get("lane_offsets", [0, 1, -1])],
            forward_offsets=[float(value) for value in background_data.get("forward_offsets", [16.0, 24.0, 36.0, 50.0, 66.0, 84.0])],
            rear_offsets=[float(value) for value in background_data.get("rear_offsets", [-16.0, -26.0, -40.0, -58.0])],
            alongside_offsets=[float(value) for value in background_data.get("alongside_offsets", [-6.0, 0.0, 6.0])],
            front_range=[float(value) for value in background_data.get("front_range", [14.0, 90.0])],
            rear_range=[float(value) for value in background_data.get("rear_range", [-60.0, -14.0])],
            adjacent_range=[float(value) for value in background_data.get("adjacent_range", [-45.0, 90.0])],
            min_spacing=float(background_data.get("min_spacing", 10.0)),
            max_attempts=int(background_data.get("max_attempts", 200)),
            jitter=float(background_data.get("jitter", 2.0)),
            allow_global_fallback=bool(background_data.get("allow_global_fallback", False)),
        ),
        traffic_manager=dict(data.get("traffic_manager", {})),
    )
