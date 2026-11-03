"""Staged curriculum: each stage adds overrides on top of all earlier ones, and the learner is
promoted to the next stage once the current one is solved (eval landing rate above a threshold
for several consecutive evals)."""

import copy
from dataclasses import dataclass, field
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from starship.sim import SimConfig
from starship.train.reward import RewardConfig


@dataclass
class CurriculumConfig:
    enabled: bool = False
    promote_landed: float = 0.9
    promote_evals: int = 2
    # Stage to begin at when the checkpoint carries no curriculum state, e.g. to skip stages
    # an earlier run already solved.
    start_stage: int = 0
    # Each stage: {"name": str, "overrides": ["sim.spawn.y=3000", "reward.time_cost=0.25"]}.
    stages: list[Any] = field(default_factory=list)


class Curriculum:
    def __init__(self, cfg: DictConfig, curriculum: CurriculumConfig | DictConfig):
        """`cfg` holds the base `sim` and `reward` configs the stage overrides apply to."""
        # Resolve first: interpolations like reward.gamma point outside the sim/reward subtree.
        resolved = copy.deepcopy(cfg)
        OmegaConf.resolve(resolved)
        self._base = OmegaConf.masked_copy(resolved, ["sim", "reward"])
        raw_stages = [OmegaConf.to_container(s, resolve=True) for s in curriculum.stages]
        self.stages = self._expand_stages(raw_stages)
        if not self.stages:
            raise ValueError("a curriculum needs at least one stage")
        self.promote_landed = curriculum.promote_landed
        self.promote_evals = curriculum.promote_evals
        if not 0 <= curriculum.start_stage < len(self.stages):
            raise ValueError(f"start_stage {curriculum.start_stage} is not a valid stage")
        self.stage = curriculum.start_stage
        self.passes = 0

    def _expand_stages(self, stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Expand numeric linear-interpolation blocks into ordinary cumulative stages."""
        merged = copy.deepcopy(self._base)
        expanded: list[dict[str, Any]] = []
        for stage in stages:
            interpolation = stage.get("interpolate")
            if interpolation is None:
                expanded.append(stage)
                # Track valid ordinary overrides so a later interpolation starts at the value that
                # will actually be active. Invalid keys remain rejected by configs(), as before.
                for override in stage.get("overrides") or []:
                    key, separator, raw_value = override.partition("=")
                    if separator and key.startswith(("sim.", "reward.")):
                        OmegaConf.update(merged, key, yaml.safe_load(raw_value), force_add=False)
                continue

            if stage.get("overrides"):
                raise ValueError("an interpolated curriculum stage cannot also define overrides")
            steps = interpolation.get("steps")
            if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
                raise ValueError("curriculum interpolation steps must be a positive integer")
            targets: list[tuple[str, float | int]] = []
            starts: list[float | int] = []
            for override in interpolation.get("overrides") or []:
                key, separator, raw_value = override.partition("=")
                if not separator or not key.startswith(("sim.", "reward.")):
                    raise ValueError(
                        f"curriculum interpolation override {override!r} must start with "
                        "sim. or reward."
                    )
                start = OmegaConf.select(merged, key)
                target = yaml.safe_load(raw_value)
                if (
                    not isinstance(start, (int, float))
                    or isinstance(start, bool)
                    or not isinstance(target, (int, float))
                    or isinstance(target, bool)
                ):
                    raise ValueError(f"curriculum interpolation {key!r} must be numeric")
                starts.append(start)
                targets.append((key, target))
            if not targets:
                raise ValueError("a curriculum interpolation needs at least one override")

            name = stage["name"]
            for step in range(1, steps + 1):
                fraction = step / steps
                overrides = []
                for start, (key, target) in zip(starts, targets, strict=True):
                    value = target if step == steps else start + (target - start) * fraction
                    overrides.append(f"{key}={value}")
                expanded.append({"name": f"{name} ({step}/{steps})", "overrides": overrides})
            for key, target in targets:
                OmegaConf.update(merged, key, target, force_add=False)
        return expanded

    @property
    def name(self) -> str:
        return self.stages[self.stage]["name"]

    @property
    def is_final(self) -> bool:
        return self.stage == len(self.stages) - 1

    def configs(self, stage: int | None = None) -> tuple[SimConfig, RewardConfig]:
        """Sim and reward configs with the overrides of stages 0..stage applied."""
        stage = self.stage if stage is None else stage
        merged = OmegaConf.masked_copy(self._base, ["sim", "reward"])
        for s in self.stages[: stage + 1]:
            for override in s.get("overrides") or []:
                key, _, value = override.partition("=")
                if not key.startswith(("sim.", "reward.")):
                    raise ValueError(f"curriculum override {key!r} must start with sim. or reward.")
                OmegaConf.update(merged, key, yaml.safe_load(value), force_add=False)
        return OmegaConf.to_object(merged.sim), OmegaConf.to_object(merged.reward)

    def target_configs(self) -> tuple[SimConfig, RewardConfig]:
        return self.configs(len(self.stages) - 1)

    def observe_eval(self, landed: float) -> bool:
        """Record an eval of the current stage; True when the learner should be promoted."""
        self.passes = self.passes + 1 if landed >= self.promote_landed else 0
        return not self.is_final and self.passes >= self.promote_evals

    def advance(self) -> None:
        self.stage += 1
        self.passes = 0

    def state_dict(self) -> dict:
        return {"stage": self.stage, "passes": self.passes}

    def load_state_dict(self, state: dict) -> None:
        self.stage = min(state["stage"], len(self.stages) - 1)
        self.passes = state.get("passes", 0)
