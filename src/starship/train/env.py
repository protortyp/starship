"""Batched RL environment over the Warp simulator; all buffers stay on the simulator's device."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from starship.sim import NUM_ACTIONS, SimConfig, Simulator
from starship.train.encoding import (
    OBS_DIM,
    ObsConfig,
    episode_end_kernel,
    make_obs_scales,
    observe,
)
from starship.train.reward import (
    RewardConfig,
    make_reward_params,
    reset_potential_kernel,
    reward_kernel,
)


@dataclass(frozen=True)
class StepResult:
    """Device arrays describing one batched step; they are overwritten by the next step.

    `obs` already belongs to the next episode for envs that just finished (they auto-reset);
    their last observation of the finished episode is in `final_obs`.
    """

    obs: wp.array
    reward: wp.array
    final_obs: wp.array
    terminated: wp.array
    truncated: wp.array


class RocketEnv:
    num_actions = NUM_ACTIONS
    obs_dim = OBS_DIM

    def __init__(
        self,
        num_envs: int,
        sim_config: SimConfig | None = None,
        obs_config: ObsConfig | None = None,
        reward_config: RewardConfig | None = None,
        device: str | None = None,
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self.sim = Simulator(num_envs, sim_config, device=device, seed=seed, auto_reset=True)
        self.device = self.sim.device
        self.obs_config = obs_config or ObsConfig()
        self._scales = make_obs_scales(self.obs_config)
        self.reward_config = reward_config or RewardConfig()
        self._reward_params = make_reward_params(self.reward_config)

        def buffer(shape, dtype=float):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        self.obs = buffer((num_envs, OBS_DIM))
        self.final_obs = buffer((num_envs, OBS_DIM))
        self.reward = buffer(num_envs)
        self._last_potential = buffer(num_envs)
        self.terminated = buffer(num_envs, wp.int32)
        self.truncated = buffer(num_envs, wp.int32)

    @property
    def config(self) -> SimConfig:
        return self.sim.config

    def reset(self) -> wp.array:
        self.sim.reset()
        wp.launch(
            reset_potential_kernel,
            dim=self.num_envs,
            inputs=[self.sim.params, self._reward_params, self.sim.state, self._last_potential],
            device=self.device,
        )
        self._observe()
        return self.obs

    def step(self, actions: wp.array | np.ndarray) -> StepResult:
        self.sim.step(actions)
        wp.launch(
            episode_end_kernel,
            dim=self.num_envs,
            inputs=[
                self.sim.params,
                self._scales,
                self.sim.terminal,
                self.final_obs,
                self.terminated,
                self.truncated,
            ],
            device=self.device,
        )
        wp.launch(
            reward_kernel,
            dim=self.num_envs,
            inputs=[
                self.sim.params,
                self._reward_params,
                self.sim.state,
                self.sim.terminal,
                self._last_potential,
                self.reward,
            ],
            device=self.device,
        )
        self._observe()
        return StepResult(self.obs, self.reward, self.final_obs, self.terminated, self.truncated)

    def _observe(self) -> None:
        observe(self.sim, self._scales, self.obs)
