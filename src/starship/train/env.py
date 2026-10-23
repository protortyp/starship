"""Batched training environment with terminal observations captured before automatic respawn."""

from dataclasses import dataclass

import numpy as np
import torch
import warp as wp

from starship.sim import FLYING, SimConfig, Simulator
from starship.sim.physics import (
    ATTITUDE_FIELDS,
    Attitude,
    Commands,
    Params,
    State,
    Terminal,
    advance,
    init_rng_kernel,
    spawn,
)
from starship.train.actions import (
    ACTION_DIM,
    CONTINUOUS_HEADS,
    make_action_ranges,
    map_actions_kernel,
)
from starship.train.encoding import (
    OBS_DIM,
    ObsConfig,
    ObsScales,
    encode,
    make_obs_scales,
    observe_kernel,
)
from starship.train.model_state import STATE_DIM, pack_state
from starship.train.reward import (
    REWARD_TERMS,
    RewardConfig,
    RewardParams,
    make_reward_params,
    potential,
    reward_kernel,
)


@wp.kernel
def _respawn_kernel(
    state: State,
    terminal: Terminal,
    force: int,
    mirror: int,
    rng: wp.array(dtype=wp.uint32),
    mirrored: wp.array(dtype=int),
    landing_load: wp.array(dtype=float),
    previous: wp.array2d(dtype=float),
    has_previous: wp.array(dtype=int),
):
    i = wp.tid()
    if force != 0 or terminal.done[i] != 0:
        random = rng[i]
        mirrored[i] = 0
        if mirror != 0 and wp.randf(random) < 0.5:
            mirrored[i] = 1
        rng[i] = random
        landing_load[i] = wp.dot(state.tanks[i], wp.vec4(1.0))
        has_previous[i] = 0
        for j in range(CONTINUOUS_HEADS):
            previous[i, j] = 0.0


@wp.kernel
def _step_kernel(
    p: Params,
    r: RewardParams,
    scales: ObsScales,
    commands: Commands,
    state: State,
    terminal: Terminal,
    mirrored: wp.array(dtype=int),
    landing_load: wp.array(dtype=float),
    before: wp.array(dtype=wp.vec3),
    fuel_before: wp.array(dtype=float),
    time_before: wp.array(dtype=float),
    attitude_before: wp.array(dtype=Attitude),
    final_obs: wp.array2d(dtype=float),
    final_mirrored: wp.array(dtype=int),
    model_before: wp.array2d(dtype=float),
    model_after: wp.array2d(dtype=float),
):
    i = wp.tid()
    if model_before.shape[0] > 0:
        pack_state(state, i, model_before)
    before[i] = potential(p, r, state.body[i], state.tanks[i])
    fuel_before[i] = wp.dot(state.tanks[i], wp.vec4(1.0))
    time_before[i] = state.time[i]
    attitude_before[i] = state.attitude[i]
    terminal.done[i] = 0
    for j in range(OBS_DIM):
        final_obs[i, j] = 0.0
    final_mirrored[i] = mirrored[i]
    advance(p, state, commands, i)
    if model_after.shape[0] > 0:
        pack_state(state, i, model_after)
    if state.status[i] != FLYING:
        terminal.done[i] = 1
        terminal.status[i] = state.status[i]
        terminal.body[i] = state.body[i]
        terminal.attitude[i] = state.attitude[i]
        terminal.tanks[i] = state.tanks[i]
        terminal.time[i] = state.time[i]
        terminal.steps[i] = state.steps[i]
        terminal.wind[i] = state.wind[i]
        # The simulator's terminal struct has no actuators, so encode before spawn discards them.
        encode(
            state.body[i],
            state.engine[i],
            state.flaps[i],
            state.tanks[i],
            landing_load[i],
            mirrored[i],
            scales,
            final_obs,
            i,
        )
        spawn(p, state, i)


@wp.kernel
def _terminal_views_kernel(
    terminal: Terminal,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    rot: wp.array(dtype=wp.quat),
    omega: wp.array(dtype=wp.vec3),
    attitude: wp.array2d(dtype=float),
    wind: wp.array2d(dtype=float),
):
    i = wp.tid()
    b = terminal.body[i]
    pos[i] = b.pos
    vel[i] = b.vel
    rot[i] = b.rot
    omega[i] = b.omega
    a = terminal.attitude[i]
    attitude[i, 0] = a.roll_travel
    attitude[i, 1] = a.horizontal_travel
    attitude[i, 2] = a.peak_roll_rate
    attitude[i, 3] = a.peak_horizontal_rate
    attitude[i, 4] = float(a.flips)
    attitude[i, 5] = float(a.reflips)
    attitude[i, 6] = a.peak_dynamic_pressure
    attitude[i, 7] = a.peak_mach
    attitude[i, 8] = a.peak_load
    attitude[i, 9] = a.flip_altitude
    attitude[i, 10] = a.ignition_altitude
    attitude[i, 11] = a.start_offset
    w = terminal.wind[i]
    wind[i, 0] = wp.length(w.surface)
    wind[i, 1] = w.turbulence


@dataclass
class StepResult:
    """Device buffers, valid until the next step or reset; final_obs is zero for ongoing envs."""

    obs: wp.array
    reward: wp.array
    terminated: wp.array
    truncated: wp.array
    final_obs: wp.array
    terminal_reward: wp.array
    reward_terms: wp.array


class RocketEnv:
    def __init__(
        self,
        num_envs: int,
        sim_config: SimConfig | None = None,
        obs_config: ObsConfig | None = None,
        reward_config: RewardConfig | None = None,
        device: str | None = None,
        seed: int = 0,
        record_model_state: bool = False,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.num_envs = num_envs
        self.config = sim_config or SimConfig()
        self.obs_config = obs_config or ObsConfig()
        self.reward_config = reward_config or RewardConfig()
        self.obs_scales = make_obs_scales(self.obs_config)
        self.reward_params = make_reward_params(self.reward_config)
        self.sim = Simulator(num_envs, self.config, device, seed, auto_reset=True)
        self.device = self.sim.device
        shape = (num_envs if record_model_state else 0, STATE_DIM)
        self.model_before = wp.zeros(shape, dtype=float, device=self.device)
        self.model_after = wp.zeros(shape, dtype=float, device=self.device)

        def zeros(dtype=float, width=None):
            shape = num_envs if width is None else (num_envs, width)
            return wp.zeros(shape, dtype=dtype, device=self.device)

        self.obs = zeros(width=OBS_DIM)
        self.reward = zeros()
        self.terminal_reward = zeros()
        self.reward_terms = zeros(width=len(REWARD_TERMS))
        self.terminated = zeros(int)
        self.truncated = zeros(int)
        self.final_obs = zeros(width=OBS_DIM)
        self.mirrored = zeros(int)
        self.final_mirrored = zeros(int)
        self.actions = zeros(width=ACTION_DIM)
        # Squashed targets stay in policy coordinates; reflection preserves their squared changes.
        self.squashed_actions = zeros(width=CONTINUOUS_HEADS)
        self.previous_actions = zeros(width=CONTINUOUS_HEADS)
        self._has_previous = zeros(int)
        self.landing_load = zeros()
        self._mirror_rng = zeros(wp.uint32)
        self._potential_before = zeros(wp.vec3)
        self._fuel_before = zeros()
        self._time_before = zeros()
        self._attitude_before = zeros(Attitude)
        self._ranges = make_action_ranges(self.device)
        self.final_pos = zeros(wp.vec3)
        self.final_vel = zeros(wp.vec3)
        self.final_rot = zeros(wp.quat)
        self.final_omega = zeros(wp.vec3)
        self.final_attitude = zeros(width=len(ATTITUDE_FIELDS))
        self.final_wind = zeros(width=2)  # episode's surface wind speed and turbulence level
        term = self.sim.terminal
        self.final_status, self.final_steps = term.status, term.steps
        self.final_tanks, self.final_time = term.tanks, term.time
        self._launch(init_rng_kernel, [seed ^ 0x5EED, self._mirror_rng])
        self._respawn(force=True)
        self.observe()

    def _launch(self, kernel, inputs):
        wp.launch(kernel, dim=self.num_envs, inputs=inputs, device=self.device)

    def _respawn(self, force=False):
        self._launch(
            _respawn_kernel,
            [
                self.sim.state,
                self.sim.terminal,
                int(force),
                int(self.obs_config.mirror),
                self._mirror_rng,
                self.mirrored,
                self.landing_load,
                self.previous_actions,
                self._has_previous,
            ],
        )

    def observe(self) -> wp.array:
        self._launch(
            observe_kernel,
            [self.sim.state, self.landing_load, self.mirrored, self.obs_scales, self.obs],
        )
        return self.obs

    def set_configs(self, sim_config: SimConfig, reward_config: RewardConfig) -> None:
        """Apply curriculum parameters without resetting running episodes or their fuel baseline."""
        reward_params = make_reward_params(reward_config)
        self.sim.set_config(sim_config)
        self.config, self.reward_config = sim_config, reward_config
        self.reward_params = reward_params

    def reset(self) -> wp.array:
        self.sim.reset()
        for buffer in (
            self.sim.terminal.done,
            self.sim.terminal.status,
            self.sim.terminal.steps,
            self.sim.terminal.time,
            self.sim.terminal.body,
            self.sim.terminal.attitude,
            self.sim.terminal.tanks,
            self.sim.terminal.wind,
            self.reward,
            self.terminal_reward,
            self.reward_terms,
            self.terminated,
            self.truncated,
            self.final_obs,
            self.final_mirrored,
            self.final_pos,
            self.final_vel,
            self.final_rot,
            self.final_omega,
            self.final_attitude,
            self.final_wind,
            self.actions,
            self.squashed_actions,
        ):
            buffer.zero_()
        self._respawn(force=True)
        return self.observe()

    def step(self, actions: wp.array | np.ndarray | torch.Tensor) -> StepResult:
        if tuple(actions.shape) != (self.num_envs, ACTION_DIM):
            raise ValueError(f"actions must have shape ({self.num_envs}, {ACTION_DIM})")
        if isinstance(actions, torch.Tensor):
            actions = wp.from_torch(
                actions.detach().to(device=str(self.device), dtype=torch.float32).contiguous()
            )
            if self.device.is_cuda:
                wp.get_stream(self.device).wait_stream(wp.stream_from_torch(str(self.device)))
        elif not isinstance(actions, wp.array):
            actions = wp.array(np.asarray(actions, dtype=np.float32), device=self.device)
        if actions.dtype != wp.float32:
            raise ValueError("Warp actions must have dtype float32")
        wp.copy(self.actions, actions)
        self._launch(
            map_actions_kernel,
            [self.actions, self.mirrored, self._ranges, self.sim.commands, self.squashed_actions],
        )
        self._launch(
            _step_kernel,
            [
                self.sim.params,
                self.reward_params,
                self.obs_scales,
                self.sim.commands,
                self.sim.state,
                self.sim.terminal,
                self.mirrored,
                self.landing_load,
                self._potential_before,
                self._fuel_before,
                self._time_before,
                self._attitude_before,
                self.final_obs,
                self.final_mirrored,
                self.model_before,
                self.model_after,
            ],
        )
        self._launch(
            reward_kernel,
            [
                self.sim.params,
                self.reward_params,
                self.sim.state,
                self.sim.terminal,
                self._potential_before,
                self._fuel_before,
                self._time_before,
                self._attitude_before,
                self.landing_load,
                self.squashed_actions,
                self.previous_actions,
                self._has_previous,
                self.reward,
                self.terminal_reward,
                self.reward_terms,
                self.terminated,
                self.truncated,
            ],
        )
        self._launch(
            _terminal_views_kernel,
            [
                self.sim.terminal,
                self.final_pos,
                self.final_vel,
                self.final_rot,
                self.final_omega,
                self.final_attitude,
                self.final_wind,
            ],
        )
        self._respawn()
        self.observe()
        return StepResult(
            self.obs,
            self.reward,
            self.terminated,
            self.truncated,
            self.final_obs,
            self.terminal_reward,
            self.reward_terms,
        )
