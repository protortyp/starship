"""Vectorized scripted controllers operating on host-side state snapshots."""

from dataclasses import dataclass

import numpy as np

from starship.sim import LEFT, MAIN, NUM_ACTIONS, RIGHT, SimConfig, StateSnapshot


@dataclass
class HeuristicController:
    """Belly-flop, late flip, and velocity-profile suicide burn with bang-bang thrusters.

    The rocket stays flat to ride the high drag down, flips upright once its tail is below
    `flip_altitude`, and fires the main engine whenever it descends faster than the speed it
    could still cancel with `brake_fraction` of the engine's net deceleration.
    """

    config: SimConfig
    flip_altitude: float = 800.0
    brake_fraction: float = 0.6
    touchdown_speed: float = 1.0
    # Lateral guidance: tilt toward the pad proportionally to offset and drift.
    tilt_per_meter: float = 0.002
    tilt_per_mps: float = 0.03
    max_tilt: float = 0.25
    attitude_gain: float = 0.8
    max_rate: float = 0.6
    rate_deadband: float = 0.02
    max_burn_tilt: float = 0.5

    def __call__(self, s: StateSnapshot) -> np.ndarray:
        r = self.config.rocket
        x, y = s.pos[:, 0], s.pos[:, 1]
        vx, vy = s.vel[:, 0], s.vel[:, 1]
        tail_altitude = y - 0.5 * r.length

        target = np.clip(
            self.tilt_per_meter * x + self.tilt_per_mps * vx, -self.max_tilt, self.max_tilt
        )
        target = np.where(tail_altitude > self.flip_altitude, 0.5 * np.pi, target)
        error = np.arctan2(np.sin(s.theta - target), np.cos(s.theta - target))
        rate_cmd = np.clip(-self.attitude_gain * error, -self.max_rate, self.max_rate)

        actions = np.zeros(len(x), dtype=np.int32)
        actions |= np.where(s.omega < rate_cmd - self.rate_deadband, RIGHT, 0)
        actions |= np.where(s.omega > rate_cmd + self.rate_deadband, LEFT, 0)

        brake = self.brake_fraction * (r.main_thrust / r.mass - self.config.world.gravity)
        v_profile = -(np.sqrt(2.0 * brake * np.maximum(tail_altitude, 0.0)) + self.touchdown_speed)
        burn = (vy < v_profile) & (np.abs(s.theta) < self.max_burn_tilt)
        actions |= np.where(burn, MAIN, 0)
        return actions.astype(np.int32)


@dataclass
class RandomController:
    seed: int = 0

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def __call__(self, s: StateSnapshot) -> np.ndarray:
        return self._rng.integers(0, NUM_ACTIONS, size=len(s.status), dtype=np.int32)


class IdleController:
    def __call__(self, s: StateSnapshot) -> np.ndarray:
        return np.zeros(len(s.status), dtype=np.int32)
