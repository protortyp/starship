"""Rocket outline and thruster plume geometry in world coordinates, vectorized over envs."""

import numpy as np

from starship.sim import LEFT, MAIN, RIGHT, RocketConfig


def body_frame(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Body axis (tail to nose) and body normal (rocket's right side when upright), (N, 2) each."""
    theta = np.asarray(theta, dtype=np.float64)
    a = np.stack([-np.sin(theta), np.cos(theta)], axis=-1)
    n = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    return a, n


def _to_world(pos, a, n, body: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float64)
    return (
        pos[:, None, :]
        + body[None, :, 0, None] * a[:, None, :]
        + body[None, :, 1, None] * n[:, None, :]
    )


def hull_outline(pos: np.ndarray, theta: np.ndarray, rocket: RocketConfig) -> np.ndarray:
    """Closed-polygon outline with a tapered nose, (N, K, 2) for pos (N, 2) and theta (N,)."""
    half_l, half_d = 0.5 * rocket.length, 0.5 * rocket.diameter
    nose = 0.25 * rocket.length
    body = np.array(
        [
            (-half_l, -half_d),
            (half_l - nose, -half_d),
            (half_l - 0.45 * nose, -0.8 * half_d),
            (half_l - 0.1 * nose, -0.4 * half_d),
            (half_l, 0.0),
            (half_l - 0.1 * nose, 0.4 * half_d),
            (half_l - 0.45 * nose, 0.8 * half_d),
            (half_l - nose, half_d),
            (-half_l, half_d),
        ]
    )
    a, n = body_frame(theta)
    return _to_world(pos, a, n, body)


def plumes(
    pos: np.ndarray, theta: float, action: int, rocket: RocketConfig, scale: float = 1.0
) -> list[np.ndarray]:
    """Triangles (3, 2) for each firing thruster of one rocket; `scale` stretches the length."""
    pos = np.asarray(pos, dtype=np.float64)
    a, n = (v[0] for v in body_frame(np.array([theta])))
    triangles = []

    def triangle(base, direction, half_width, length):
        side = np.array([-direction[1], direction[0]])
        tip = base + direction * length * scale
        triangles.append(np.array([base + side * half_width, tip, base - side * half_width]))

    if action & MAIN:
        triangle(pos - 0.5 * rocket.length * a, -a, 0.35 * rocket.diameter, 0.6 * rocket.length)
    nozzle = pos + (rocket.side_arm - 0.3 * rocket.length) * a
    # The left thruster pushes the rocket toward +n, so its exhaust leaves on the -n side.
    if action & LEFT:
        triangle(nozzle - 0.5 * rocket.diameter * n, -n, 1.2, 0.35 * rocket.length)
    if action & RIGHT:
        triangle(nozzle + 0.5 * rocket.diameter * n, n, 1.2, 0.35 * rocket.length)
    return triangles
