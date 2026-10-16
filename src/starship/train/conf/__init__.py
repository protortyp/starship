"""Hydra config: `train.yaml` layered on the structured `TrainConfig` schema registered here."""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

from starship.sim import SimConfig
from starship.train.encoding import ObsConfig
from starship.train.reward import RewardConfig


@dataclass
class TrainConfig:
    experiment: str = "landing"
    num_envs: int = 4096
    device: str | None = None
    seed: int = 0
    iterations: int = 500
    sim: SimConfig = field(default_factory=SimConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    runner: dict[str, Any] = field(default_factory=dict)


ConfigStore.instance().store(name="train_schema", node=TrainConfig)
