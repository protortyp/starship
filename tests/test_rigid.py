import numpy as np
import pytest

from starship.sim.rigid import G0, simulate

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def rotate(q, v):
    x, y, z, w = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)


def test_free_fall_matches_kinematics():
    (b,) = simulate([((0, 0, 1000), (5, 0, 0), IDENTITY, (0, 0, 0))], dt=0.001, steps=5000)
    t = 5.0
    assert b["pos"][0] == pytest.approx(5 * t, rel=1e-4)
    # Gravity falls off by ~0.03 % over 1 km, below the integration error checked here.
    assert b["pos"][2] == pytest.approx(1000 - 0.5 * G0 * t * t, abs=0.2)
    assert b["vel"][2] == pytest.approx(-G0 * t, rel=1e-3)


def test_constant_torque_spins_up_about_principal_axis():
    inertia = (2.0, 50.0, 50.0)
    (b,) = simulate(
        [((0, 0, 1e6), (0, 0, 0), IDENTITY, (0, 0, 0))],
        torque=(0, 10.0, 0),
        inertia=inertia,
        dt=0.001,
        steps=1000,
    )
    assert b["omega"][1] == pytest.approx(10.0 / 50.0 * 1.0, rel=1e-3)


def test_torque_free_tumble_conserves_momentum_and_energy():
    inertia = np.array([3e6, 4.8e7, 5.0e7])
    omega0 = np.array([0.5, 0.3, -0.2])
    (b,) = simulate(
        [((0, 0, 1e6), (0, 0, 0), IDENTITY, tuple(omega0))],
        inertia=tuple(inertia),
        dt=0.01,
        steps=2000,
    )
    q = np.asarray(b["rot"], dtype=float)
    omega = np.asarray(b["omega"], dtype=float)
    momentum0 = inertia * omega0
    momentum = rotate(q, inertia * omega)
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-5)
    assert np.allclose(momentum, momentum0, rtol=5e-3, atol=5e-3 * np.linalg.norm(momentum0))
    energy0 = 0.5 * np.dot(inertia * omega0, omega0)
    energy = 0.5 * np.dot(inertia * omega, omega)
    assert energy == pytest.approx(energy0, rel=5e-3)


def test_rotation_maps_body_axis_into_world():
    # Spin about body b2 by 90 deg: the axis b1 (engines to nose) ends up along world -z.
    rate = np.pi / 2
    (b,) = simulate(
        [((0, 0, 1e6), (0, 0, 0), IDENTITY, (0, rate, 0))], dt=0.001, steps=1000, inertia=(1, 1, 1)
    )
    assert np.allclose(rotate(b["rot"], (1, 0, 0)), (0, 0, -1), atol=1e-3)
