from dataclasses import dataclass, field


@dataclass
class SpawnConfig:
    index: int = 0
    offset: float = 0.0
    lane_offset: int = 0


@dataclass
class ActorConfig:
    id: str
    role: str = "npc"
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    autopilot: bool = True
    spawn_probability: float = 1.0
    behavior: dict = field(default_factory=dict)
    cut_in_actor: bool = False


@dataclass
class ScenarioConfig:
    scenario_id: int
    scenario_name: str
    map: str = "Town04"
    fps: float = 20.0
    steps: int = 500
    seed: int = 42
    type: str = "fixed"
    ego_spawn: SpawnConfig = field(default_factory=SpawnConfig)
    background_spawn: dict = field(default_factory=dict)
    noise: dict = field(default_factory=dict)
    ego_maneuver: dict = field(default_factory=dict)
    actors: list[ActorConfig] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

