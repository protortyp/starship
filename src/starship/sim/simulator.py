"""Batched simulator: N independent rockets stepped in parallel by a single Warp kernel launch."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from starship.sim.config import SimConfig, make_params
from starship.sim.physics import State, Terminal, init_rng_kernel, reset_kernel, step_kernel


@dataclass(frozen=True)
class StateSnapshot:
    """Host-side copy of the simulator state, one entry per environment."""

    pos: np.ndarray
    vel: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    status: np.ndarray
    steps: np.ndarray
    fuel: np.ndarray


@dataclass(frozen=True)
class TerminalSnapshot:
    """Host-side copy of the terminal buffers; entries are meaningful only where `done` is set."""

    done: np.ndarray
    status: np.ndarray
    pos: np.ndarray
    vel: np.ndarray
    theta: np.ndarray
    omega: np.ndarray
    steps: np.ndarray
    fuel: np.ndarray


class Simulator:
    """Holds the state of `num_envs` rockets on a Warp device.

    When an episode ends, the step that ended it sets `terminal.done` and records the final state
    in `terminal`. With `auto_reset`, the env then respawns within the same kernel launch, so
    rollouts never need host round-trips. Without it, finished envs stay frozen until `reset`.
    """

    def __init__(
        self,
        num_envs: int,
        config: SimConfig | None = None,
        device: str | None = None,
        seed: int = 0,
        auto_reset: bool = False,
    ):
        self.num_envs = num_envs
        self.config = config or SimConfig()
        self.device = wp.get_device(device)
        self.params = make_params(self.config)
        self.auto_reset = auto_reset

        def zeros(dtype):
            return wp.zeros(num_envs, dtype=dtype, device=self.device)

        self.state = State()
        self.state.pos = zeros(wp.vec2)
        self.state.vel = zeros(wp.vec2)
        self.state.theta = zeros(float)
        self.state.omega = zeros(float)
        self.state.status = zeros(wp.int32)
        self.state.steps = zeros(wp.int32)
        self.state.fuel = zeros(float)
        self.state.rng = zeros(wp.uint32)

        self.terminal = Terminal()
        self.terminal.done = zeros(wp.int32)
        self.terminal.status = zeros(wp.int32)
        self.terminal.pos = zeros(wp.vec2)
        self.terminal.vel = zeros(wp.vec2)
        self.terminal.theta = zeros(float)
        self.terminal.omega = zeros(float)
        self.terminal.steps = zeros(wp.int32)
        self.terminal.fuel = zeros(float)

        self.actions = zeros(wp.int32)
        self._all_envs = wp.ones(num_envs, dtype=wp.int32, device=self.device)

        wp.launch(init_rng_kernel, dim=num_envs, inputs=[seed, self.state.rng], device=self.device)
        self.reset()

    def reset(self, mask: wp.array | np.ndarray | None = None) -> None:
        """Respawn every environment, or only those where `mask` is nonzero."""
        if mask is None:
            mask = self._all_envs
        elif not isinstance(mask, wp.array):
            mask = wp.array(np.asarray(mask, dtype=np.int32), device=self.device)
        wp.launch(
            reset_kernel,
            dim=self.num_envs,
            inputs=[self.params, mask, self.state, self.terminal],
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
            inputs=[self.params, int(self.auto_reset), self.actions, self.state, self.terminal],
            device=self.device,
        )

    def snapshot(self) -> StateSnapshot:
        s = self.state
        arrays = (s.pos, s.vel, s.theta, s.omega, s.status, s.steps, s.fuel)
        # On CPU devices `numpy()` aliases the Warp buffer, so copy to decouple from later steps.
        return StateSnapshot(*(a.numpy().copy() for a in arrays))

    def terminal_snapshot(self) -> TerminalSnapshot:
        t = self.terminal
        arrays = (t.done, t.status, t.pos, t.vel, t.theta, t.omega, t.steps, t.fuel)
        return TerminalSnapshot(*(a.numpy().copy() for a in arrays))
