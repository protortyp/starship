"""Batched simulator: N independent rockets stepped in parallel by a single Warp kernel launch."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from starship.sim.config import SimConfig, make_params
from starship.sim.physics import reset_kernel, step_kernel


@dataclass(frozen=True)
class StateSnapshot:
    """Host-side copy of the simulator state, one entry per environment."""

    pos: np.ndarray
    vel: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    status: np.ndarray
    steps: np.ndarray


class Simulator:
    """Holds the state of `num_envs` rockets on a Warp device.

    Finished environments (status != FLYING) stay frozen in their terminal state until reset, so
    the final touchdown state can be inspected after the episode ends.
    """

    def __init__(
        self,
        num_envs: int,
        config: SimConfig | None = None,
        device: str | None = None,
        seed: int = 0,
    ):
        self.num_envs = num_envs
        self.config = config or SimConfig()
        self.device = wp.get_device(device)
        self.params = make_params(self.config)
        self._seed = seed
        self._reset_count = 0

        def zeros(dtype):
            return wp.zeros(num_envs, dtype=dtype, device=self.device)

        self.pos = zeros(wp.vec2)
        self.vel = zeros(wp.vec2)
        self.theta = zeros(float)
        self.omega = zeros(float)
        self.status = zeros(wp.int32)
        self.steps = zeros(wp.int32)
        self.actions = zeros(wp.int32)
        self._all_envs = wp.ones(num_envs, dtype=wp.int32, device=self.device)

        self.reset()

    def _state_arrays(self):
        return [self.pos, self.vel, self.theta, self.omega, self.status, self.steps]

    def reset(self, mask: wp.array | np.ndarray | None = None) -> None:
        """Respawn every environment, or only those where `mask` is nonzero."""
        if mask is None:
            mask = self._all_envs
        elif not isinstance(mask, wp.array):
            mask = wp.array(np.asarray(mask, dtype=np.int32), device=self.device)
        # Each reset draws fresh spawn noise while staying reproducible for a given seed.
        seed = self._seed * 1_000_003 + self._reset_count
        self._reset_count += 1
        wp.launch(
            reset_kernel,
            dim=self.num_envs,
            inputs=[self.params, seed, mask, *self._state_arrays()],
            device=self.device,
        )

    def step(self, actions: wp.array | np.ndarray | int) -> None:
        """Advance every flying environment by one policy step (`config.substeps` physics steps).

        `actions` holds one thruster bitmask per environment (see `config.MAIN/LEFT/RIGHT`).
        """
        if isinstance(actions, wp.array):
            wp.copy(self.actions, actions)
        else:
            self.actions.assign(np.broadcast_to(np.asarray(actions, np.int32), self.num_envs))
        wp.launch(
            step_kernel,
            dim=self.num_envs,
            inputs=[self.params, self.actions, *self._state_arrays()],
            device=self.device,
        )

    def snapshot(self) -> StateSnapshot:
        # On CPU devices `numpy()` aliases the Warp buffer, so copy to decouple from later steps.
        return StateSnapshot(*(a.numpy().copy() for a in self._state_arrays()))
