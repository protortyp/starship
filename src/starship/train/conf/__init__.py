"""Hydra config: `train.yaml` layered on the structured `TrainConfig` schema registered here."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

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
    # Reload model_best.pt when the stage's eval landing rate stays below (1 - rollback_drop) x its
    # best for rollback_patience evals (0 disables).
    rollback_drop: float = 0.5
    rollback_patience: int = 2
    # Write Rerun recordings of eval episodes every `record_every` evals (0 disables).
    record_every: int = 4
    record_envs: int = 16
    # model_<it>.pt from an earlier run to continue from (weights, optimizer and normalizers).
    checkpoint: str | None = None
    sim: SimConfig = field(default_factory=SimConfig)
    obs: ObsConfig = field(default_factory=ObsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    runner: dict[str, Any] = field(default_factory=dict)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)


ConfigStore.instance().store(name="train_schema", node=TrainConfig)


def load_run_config(run: Path):
    """A finished run's Hydra config on the current schema. Settings the schema no longer has (e.g.
    after a model upgrade) are dropped, so older runs stay loadable for viewing and diagnosis."""
    schema = OmegaConf.structured(TrainConfig)
    saved = OmegaConf.load(Path(run) / ".hydra/config.yaml")

    def known(node, ref):
        # Free-form sections (runner: dict[str, Any]) have no schema to check against.
        if not isinstance(node, DictConfig) or OmegaConf.get_type(ref) is dict:
            return node
        return OmegaConf.create({k: known(v, ref[k]) for k, v in node.items() if k in ref})

    return OmegaConf.merge(schema, known(saved, schema))
