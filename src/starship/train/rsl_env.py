"""rsl_rl VecEnv adapter over RocketEnv."""

from dataclasses import asdict

import torch
import warp as wp
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from starship.sim import LANDED, LEFT, MAIN, RIGHT
from starship.train.env import RocketEnv

# Policy output columns, in order, mapped to thruster bits.
THRUSTER_BITS = (MAIN, LEFT, RIGHT)


def actions_to_bitmask(actions: torch.Tensor) -> torch.Tensor:
    bits = torch.tensor(THRUSTER_BITS, dtype=torch.int32, device=actions.device)
    return ((actions > 0.5).int() * bits).sum(dim=-1, dtype=torch.int32)


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
        self.episode_length_buf = wp.to_torch(env.sim.state.steps)
        self._obs = wp.to_torch(env.obs)
        self._reward = wp.to_torch(env.reward)
        self._terminated = wp.to_torch(env.terminated)
        self._truncated = wp.to_torch(env.truncated)
        self._final_status = wp.to_torch(env.sim.terminal.status)
        self._final_vel = wp.to_torch(env.sim.terminal.vel)
        self._final_fuel = wp.to_torch(env.sim.terminal.fuel)
        self._actions = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        env.reset()

    def _observations(self) -> TensorDict:
        # Cloned because rsl_rl keeps the observation it acted on until after the next step.
        return TensorDict({"policy": self._obs.clone()}, batch_size=[self.num_envs])

    def get_observations(self) -> TensorDict:
        return self._observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        self._actions.copy_(actions_to_bitmask(actions))
        self.env.step(wp.from_torch(self._actions))
        terminated = self._terminated.bool()
        truncated = self._truncated.bool()
        dones = terminated | truncated
        extras = {"time_outs": truncated.clone(), "log": self._episode_metrics(dones)}
        return self._observations(), self._reward.clone(), dones.long(), extras

    def _episode_metrics(self, dones: torch.Tensor) -> dict[str, torch.Tensor]:
        if not dones.any():
            return {}
        landed = self._final_status[dones] == LANDED
        metrics = {"landing/landed_rate": landed.float()}
        if landed.any():
            metrics["landing/touchdown_speed"] = self._final_vel[dones][landed].norm(dim=-1)
            metrics["landing/fuel_left_t"] = self._final_fuel[dones][landed] / 1000.0
        return metrics
