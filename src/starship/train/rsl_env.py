"""rsl_rl VecEnv adapter over RocketEnv."""

from dataclasses import asdict

import torch
import warp as wp
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from starship.sim import LEFT, MAIN, RIGHT
from starship.train.env import RocketEnv

# Policy output heads, in order, mapped to thruster bits.
THRUSTER_BITS = (MAIN, LEFT, RIGHT)
HEAD_NAMES = ("main", "left", "right")


def actions_to_bitmask(actions: torch.Tensor, mirrored: torch.Tensor | None = None) -> torch.Tensor:
    """Policy outputs to thruster bits; in mirrored episodes left and right swap places."""
    bits = torch.tensor(THRUSTER_BITS, dtype=torch.int32, device=actions.device)
    mask = ((actions > 0.5).int() * bits).sum(dim=-1, dtype=torch.int32)
    if mirrored is None:
        return mask
    swapped = (mask & MAIN) | ((mask & LEFT) << 1) | ((mask & RIGHT) >> 1)
    return torch.where(mirrored.bool(), swapped, mask)


class RslRocketEnv(VecEnv):
    num_actions = len(THRUSTER_BITS)

    def __init__(self, env: RocketEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.device = wp.device_to_torch(env.device)
        self.max_episode_length = env.config.max_steps
        self.cfg = {
            "sim": asdict(env.config),
            "obs": asdict(env.obs_config),
            "reward": asdict(env.reward_config),
        }
        # Zero-copy views of persistent Warp buffers, overwritten in place on every step.
        term = env.sim.terminal
        self.episode_length_buf = wp.to_torch(env.sim.state.steps)
        self._mirrored = wp.to_torch(env.sim.state.mirrored)
        self.reward = wp.to_torch(env.reward)
        self.terminal_reward = wp.to_torch(env.terminal_reward)
        self.reward_terms = wp.to_torch(env.reward_terms)
        self.final_status = wp.to_torch(term.status)
        self.final_vel = wp.to_torch(term.vel)
        self.final_theta = wp.to_torch(term.theta)
        self.final_omega = wp.to_torch(term.omega)
        self.final_fuel = wp.to_torch(term.fuel)
        self.final_steps = wp.to_torch(term.steps)
        self.actions = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.learner_scale = env.reward_config.learner_scale
        self._obs = wp.to_torch(env.obs)
        self._terminated = wp.to_torch(env.terminated)
        self._truncated = wp.to_torch(env.truncated)
        env.reset()

    def set_configs(self, sim_config=None, reward_config=None) -> None:
        self.env.set_configs(sim_config, reward_config)
        self.max_episode_length = self.env.config.max_steps
        self.learner_scale = self.env.reward_config.learner_scale
        self.cfg["sim"] = asdict(self.env.config)
        self.cfg["reward"] = asdict(self.env.reward_config)

    def _observations(self) -> TensorDict:
        # Cloned because rsl_rl keeps the observation it acted on until after the next step.
        return TensorDict({"policy": self._obs.clone()}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        return self._observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self.actions.copy_(actions_to_bitmask(actions, self._mirrored))
        self.env.step(wp.from_torch(self.actions))
        truncated = self._truncated.bool()
        dones = self._terminated.bool() | truncated
        rewards = self.reward * self.learner_scale
        return self._observations(), rewards, dones.long(), {"time_outs": truncated}
