"""Replay fixed actions against a supplied checkout; run in that checkout's PYTHONPATH.

Arguments: CHECKOUT OUTPUT. PROBE_DEVICE defaults to cpu; cuda:0 also supported.
Uses default observation scales and original landing stages, not the active bridge recipe.
This probes simulator/reward compatibility, not learned-policy success or PPO equivalence.
"""

import dataclasses
import json
import os
import pathlib
import sys

import numpy as np
import torch
import warp
from omegaconf import OmegaConf

import starship
from starship.sim.config import SimConfig
from starship.train.actions import ACTION_DIM
from starship.train.encoding import ObsConfig
from starship.train.env import RocketEnv
from starship.train.reward import RewardConfig

base = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
(out / "provenance.json").write_text(
    json.dumps(
        {
            "package": starship.__file__,
            "device": os.environ.get("PROBE_DEVICE", "cpu"),
            "torch": torch.__version__,
            "warp": warp.__version__,
            "envs_per_stage": 16,
            "steps": 400,
            "stages": 5,
        },
        indent=2,
    )
)
assert pathlib.Path(starship.__file__).resolve().is_relative_to(base.resolve()), (
    "Wrong imported checkout; set PYTHONPATH to CHECKOUT/src"
)
stages = OmegaConf.load(base / "src/starship/train/conf/curriculum/landing.yaml").stages
for stage in range(5):
    cfg = SimConfig()
    rew = RewardConfig()
    for item in stages[: stage + 1]:
        for override in item.overrides:
            key, value = override.split("=", 1)
            keys = key.split(".")
            obj = {"sim": cfg, "reward": rew}[keys[0]]
            for k in keys[1:-1]:
                obj = getattr(obj, k)
            setattr(
                obj,
                keys[-1],
                OmegaConf.from_dotlist([f"value={value}"]).value,
            )
    env = RocketEnv(
        16, cfg, ObsConfig(), rew, device=os.environ.get("PROBE_DEVICE", "cpu"), seed=4711 + stage
    )
    rng = np.random.default_rng(912 + stage)
    record = {
        k: []
        for k in [
            "obs",
            "reward",
            "terms",
            "terminated",
            "truncated",
            "terminal_status",
            "pos",
            "vel",
            "rot",
            "omega",
        ]
    }
    initial = env.observe().numpy().copy()
    for _ in range(400):
        actions = np.zeros((16, ACTION_DIM), np.float32)
        actions[:, :3] = rng.integers(0, 2, (16, 3))
        actions[:, 3:11] = rng.normal(0, 0.35, (16, 8))
        result = env.step(actions)
        snap = env.sim.snapshot()
        for key in ["pos", "vel", "rot", "omega"]:
            record[key].append(getattr(snap, key).copy())
        record["obs"].append(result.obs.numpy()[:, :24].copy())
        record["reward"].append(result.reward.numpy().copy())
        record["terms"].append(result.reward_terms.numpy()[:, :8].copy())
        for key in ["terminated", "truncated"]:
            record[key].append(getattr(result, key).numpy().copy())
        record["terminal_status"].append(env.sim.terminal.status.numpy().copy())
    np.savez_compressed(
        out / f"stage-{stage}.npz",
        initial=initial[:, :24],
        **{k: np.stack(v) for k, v in record.items()},
    )
    (out / f"stage-{stage}-config.json").write_text(
        json.dumps({"sim": dataclasses.asdict(cfg), "reward": dataclasses.asdict(rew)}, indent=2)
    )
    print("stage", stage, "done", flush=True)
