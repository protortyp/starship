"""Controller that runs a trained policy exported by `starship-train`."""

from pathlib import Path

import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf
from tensordict import TensorDict

from starship.sim import Simulator, StateSnapshot
from starship.train.encoding import OBS_DIM, ObsConfig, make_obs_scales, observe
from starship.train.rsl_env import actions_to_bitmask


def load_obs_config(policy_path: Path) -> ObsConfig:
    """Use the observation scales the policy was trained with, if the run's config is present."""
    hydra_cfg = policy_path.parent / ".hydra" / "config.yaml"
    if not hydra_cfg.exists():
        return ObsConfig()
    return ObsConfig(**OmegaConf.to_container(OmegaConf.load(hydra_cfg).obs))


class PolicyController:
    """Encodes observations from the simulator's live device state; the snapshot is only used to
    find envs that just (re)spawned, whose recurrent state must start fresh."""

    def __init__(self, sim: Simulator, policy_path: str | Path):
        policy_path = Path(policy_path)
        self.sim = sim
        self.device = wp.device_to_torch(sim.device)
        # A pickled module from our own training runs, so full unpickling is expected here.
        self.policy = torch.load(policy_path, map_location=self.device, weights_only=False).eval()
        self.policy.reset()
        self._scales = make_obs_scales(load_obs_config(policy_path))
        self._obs = wp.zeros((sim.num_envs, OBS_DIM), dtype=float, device=sim.device)

    def __call__(self, s: StateSnapshot) -> np.ndarray:
        observe(self.sim, self._scales, self._obs)
        obs = TensorDict({"policy": wp.to_torch(self._obs)}, batch_size=[self.sim.num_envs])
        with torch.no_grad():
            self.policy.reset(torch.as_tensor(s.steps == 0, device=self.device).long())
            actions = self.policy(obs, stochastic_output=False)
        return actions_to_bitmask(actions).cpu().numpy()
