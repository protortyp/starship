"""Scripted controllers that produce `Simulator.set_commands` keyword arguments."""

import numpy as np

from starship.sim import Snapshot


def body_axis_z(rot: np.ndarray) -> np.ndarray:
    """World z component of the body axis b1 for (N, 4) quaternions (x, y, z, w)."""
    x, y, z, w = rot.T
    return 2.0 * (x * z - w * y)


class BellyFlopController:
    """Holds the belly-flop with the flaps: PD on the pitch of the body axis. Opening the aft
    flaps moves the centre of pressure aft, which pushes a rising nose back down."""

    def __init__(self, kp: float = 1.0, kd: float = 1.0, aft_trim=0.9, forward_trim=0.35):
        self.kp, self.kd = kp, kd
        self.aft_trim, self.forward_trim = aft_trim, forward_trim

    def __call__(self, s: Snapshot) -> dict[str, np.ndarray]:
        u = np.clip(self.kp * body_axis_z(s.rot) + self.kd * s.omega[:, 1], -0.7, 0.7)
        aft, fwd = self.aft_trim - u, self.forward_trim + u
        return {"flaps": np.stack([aft, aft, fwd, fwd], axis=1).astype(np.float32)}


class IdleController:
    def __call__(self, s: Snapshot) -> dict[str, np.ndarray]:
        return {}


CONTROLLERS = {"bellyflop": BellyFlopController, "idle": IdleController}
