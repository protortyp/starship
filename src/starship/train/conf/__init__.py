"""Hydra config: `train.yaml` layered on the structured `TrainConfig` schema registered here."""

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

from starship.sim import SimConfig
from starship.train.curriculum import CurriculumConfig
from starship.train.encoding import ObsConfig
from starship.train.reward import RewardConfig


@dataclass
class TrainConfig:
    experiment: str = "landing"
    num_envs: int = 4096
    device: str | None = None
    seed: int = 0
    iterations: int = 500
    eval_interval: int = 25
    eval_envs: int = 256
    # Reload model_best.pt when the stage's eval landing rate stays this far below its best for
    # rollback_patience evals (0 disables).
    rollback_drop: float = 0.5
    rollback_patience: int = 2
    # model_<it>.pt from an earlier run to continue from (weights, optimizer and normalizers).
    checkpoint: str | None = None
    sim: SimConfig = field(default_factory=SimConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    runner: dict[str, Any] = field(default_factory=dict)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


ConfigStore.instance().store(name="train_schema", node=TrainConfig)
