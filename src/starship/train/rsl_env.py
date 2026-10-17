"""rsl_rl adapter; metric tensors are live device views, rollout observations are owned copies."""

import torch
import warp as wp
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from starship.sim import SimConfig
from starship.sim.physics import ATTITUDE_FIELDS
from starship.train.actions import ACTION_DIM
from starship.train.env import RocketEnv
from starship.train.reward import RewardConfig


class RslRocketEnv(VecEnv):
    def __init__(self, env: RocketEnv):
        self.env = env
        self.num_envs = env.num_envs
        self.num_actions = ACTION_DIM
        self.device = str(env.device)
        self.cfg = env.config
        self.episode_length_buf = wp.to_torch(env.sim.state.steps)
        for name in (
            "reward",
            "terminal_reward",
            "reward_terms",
            "actions",
            "squashed_actions",
            "final_status",
            "final_steps",
            "final_pos",
            "final_vel",
            "final_rot",
            "final_omega",
            "final_tanks",
            "final_time",
            "mirrored",
            "final_mirrored",
            "final_obs",
        ):
            setattr(self, name, wp.to_torch(getattr(env, name)))
        attitude = wp.to_torch(env.final_attitude)
        for i, name in enumerate(ATTITUDE_FIELDS):
            setattr(self, f"final_{name}", attitude[:, i])
        wind = wp.to_torch(env.final_wind)
        self.final_wind_speed, self.final_turbulence = wind[:, 0], wind[:, 1]
        # Physical engine order; final_mirrored identifies the policy frame of ended episodes.
        self.engine_on = wp.to_torch(env.sim.commands.engine_on)
        self._obs = wp.to_torch(env.obs)
        self._terminated = wp.to_torch(env.terminated)
        self._truncated = wp.to_torch(env.truncated)

    @property
    def max_episode_length(self) -> int:
        return self.env.sim.params.max_steps

    def set_configs(self, sim_config: SimConfig, reward_config: RewardConfig) -> None:
        self.env.set_configs(sim_config, reward_config)
        self.cfg = sim_config

    def get_observations(self) -> TensorDict:
        if self.env.device.is_cuda:
            wp.stream_from_torch(self.device).wait_stream(wp.get_stream(self.env.device))
        return TensorDict({"policy": self._obs.clone()}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        # Use Torch's stream for Warp launches so shared buffers need no host synchronization.
        if self.env.device.is_cuda:
            with wp.ScopedStream(wp.stream_from_torch(self.device), sync_exit=True):
                self.env.step(actions)
        else:
            self.env.step(actions)
        dones = self._terminated.bool() | self._truncated.bool()
        return (
            self.get_observations(),
            self.reward.clone(),
            dones,
            # Deadlines are penalized terminal failures, so PPO must not bootstrap them.
            {"time_outs": torch.zeros_like(dones), "truncated": self._truncated.bool()},
        )
