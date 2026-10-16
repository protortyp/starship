import math
from dataclasses import replace

import numpy as np
import pytest

from starship.sim import (
    CRASHED,
    FLYING,
    LANDED,
    LEFT,
    MAIN,
    OUT_OF_BOUNDS,
    RIGHT,
    TIMEOUT,
    SimConfig,
    Simulator,
    SpawnConfig,
    WorldConfig,
)
from tests import reference

VACUUM = WorldConfig(sea_level_density=0.0)
ZERO_G_VACUUM = WorldConfig(gravity=0.0, sea_level_density=0.0)


def make_sim(num_envs=1, seed=0, **overrides) -> Simulator:
    return Simulator(num_envs, replace(SimConfig(), **overrides), device="cpu", seed=seed)


def run(sim: Simulator, action, seconds: float) -> None:
    for _ in range(round(seconds / sim.config.policy_dt)):
        sim.step(action)


def test_vacuum_free_fall_matches_kinematics():
    spawn = SpawnConfig(y=3000.0, vy=-10.0)
    sim = make_sim(world=VACUUM, spawn=spawn)
    t = 5.0
    run(sim, 0, t)
    s = sim.snapshot()
    g = sim.config.world.gravity
    # Semi-implicit Euler overshoots the exact parabola by g * dt * t / 2.
    expected_y = spawn.y + spawn.vy * t - 0.5 * g * t * t - 0.5 * g * sim.config.dt * t
    assert s.pos[0, 1] == pytest.approx(expected_y, abs=1e-2)
    assert s.vel[0, 1] == pytest.approx(spawn.vy - g * t, abs=1e-3)
    assert s.vel[0, 0] == 0.0
    assert s.omega[0] == 0.0


@pytest.mark.parametrize(
    ("theta", "cd_attr", "area_attr"),
    [
        (0.5 * math.pi, "drag_coeff_normal", "area_normal"),
        (0.0, "drag_coeff_axial", "area_axial"),
    ],
)
def test_falls_at_terminal_velocity(theta, cd_attr, area_attr):
    sim = make_sim(spawn=SpawnConfig(y=5000.0, vy=0.0, theta=theta))
    run(sim, 0, 30.0)
    s = sim.snapshot()
    r = sim.config.rocket
    rho = reference.density(sim.config, float(s.pos[0, 1]))
    v_terminal = math.sqrt(
        2 * r.mass * sim.config.world.gravity / (rho * getattr(r, cd_attr) * getattr(r, area_attr))
    )
    assert s.status[0] == FLYING
    # Density increases on the way down, so the rocket lags slightly behind the local value.
    assert -s.vel[0, 1] == pytest.approx(v_terminal, rel=0.02)


def test_main_engine_thrusts_along_body_axis_without_torque():
    theta = 0.3
    sim = make_sim(world=ZERO_G_VACUUM, spawn=SpawnConfig(vy=0.0, theta=theta))
    t = 1.0
    run(sim, MAIN, t)
    s = sim.snapshot()
    r = sim.config.rocket
    expected = r.main_thrust / r.mass * t * np.array([-math.sin(theta), math.cos(theta)])
    np.testing.assert_allclose(s.vel[0], expected, rtol=1e-5)
    assert s.omega[0] == 0.0
    assert s.theta[0] == pytest.approx(theta)


@pytest.mark.parametrize(("action", "sign"), [(LEFT, -1.0), (RIGHT, 1.0)])
def test_side_thrusters_rotate_in_opposite_directions(action, sign):
    sim = make_sim(world=ZERO_G_VACUUM, spawn=SpawnConfig(vy=0.0, theta=0.0))
    t = 1.0
    run(sim, action, t)
    r = sim.config.rocket
    expected_omega = sign * r.side_arm * r.side_thrust / r.inertia * t
    assert sim.snapshot().omega[0] == pytest.approx(expected_omega, rel=1e-5)


def test_both_side_thrusters_cancel():
    sim = make_sim(world=ZERO_G_VACUUM, spawn=SpawnConfig(vy=0.0, theta=0.4))
    run(sim, LEFT | RIGHT, 1.0)
    s = sim.snapshot()
    assert s.omega[0] == 0.0
    np.testing.assert_allclose(s.vel[0], 0.0, atol=1e-6)


def test_rotation_is_damped_by_air():
    spawn = SpawnConfig(y=1000.0, vy=0.0, omega=1.0)
    sim = make_sim(world=replace(SimConfig().world, gravity=0.0), spawn=spawn)
    run(sim, 0, 2.0)
    assert 0.0 < sim.snapshot().omega[0] < spawn.omega


def upright_touchdown(**spawn_overrides) -> int:
    spawn = SpawnConfig(**{"y": 15.5, "vy": -1.0, "theta": 0.0, **spawn_overrides})
    sim = make_sim(world=ZERO_G_VACUUM, spawn=spawn)
    run(sim, 0, 2.0)
    return int(sim.snapshot().status[0])


def test_gentle_upright_touchdown_on_pad_lands():
    assert upright_touchdown() == LANDED


@pytest.mark.parametrize(
    "spawn_overrides",
    [
        {"vy": -10.0},
        {"vx": 5.0},
        {"x": 40.0},
        {"theta": math.radians(20.0), "y": 16.0},
        {"omega": 0.5},
    ],
)
def test_bad_touchdown_crashes(spawn_overrides):
    assert upright_touchdown(**spawn_overrides) == CRASHED


def test_flat_rocket_touches_down_at_half_diameter():
    sim = make_sim(world=ZERO_G_VACUUM, spawn=SpawnConfig(y=10.0, vy=-1.0))
    run(sim, 0, 10.0)
    s = sim.snapshot()
    assert s.status[0] == CRASHED
    assert s.pos[0, 1] == pytest.approx(0.5 * sim.config.rocket.diameter, abs=0.02)


def test_terminated_envs_stay_frozen():
    sim = make_sim(world=ZERO_G_VACUUM, spawn=SpawnConfig(y=10.0, vy=-5.0))
    run(sim, 0, 3.0)
    before = sim.snapshot()
    run(sim, MAIN | LEFT, 1.0)
    after = sim.snapshot()
    assert before.status[0] == CRASHED
    np.testing.assert_array_equal(before.pos, after.pos)
    np.testing.assert_array_equal(before.steps, after.steps)


def test_timeout():
    sim = make_sim(world=ZERO_G_VACUUM, spawn=SpawnConfig(vy=0.0), max_time=2.0)
    run(sim, 0, 2.0)
    assert sim.snapshot().status[0] == TIMEOUT


def test_out_of_bounds():
    sim = make_sim(world=ZERO_G_VACUUM, spawn=SpawnConfig(vx=500.0, vy=0.0), max_abs_x=100.0)
    run(sim, 0, 1.0)
    assert sim.snapshot().status[0] == OUT_OF_BOUNDS


NOISY_SPAWN = SpawnConfig(
    x_noise=500.0,
    y_noise=500.0,
    vx_noise=20.0,
    vy_noise=20.0,
    theta_noise=0.5,
    omega_noise=0.2,
)


def test_spawn_noise_stays_within_bounds_and_varies():
    sim = make_sim(num_envs=256, spawn=NOISY_SPAWN)
    s = sim.snapshot()
    sp = NOISY_SPAWN
    assert np.all(np.abs(s.pos[:, 0] - sp.x) <= sp.x_noise)
    assert np.all(np.abs(s.vel[:, 1] - sp.vy) <= sp.vy_noise)
    assert np.all(np.abs(s.omega - sp.omega) <= sp.omega_noise)
    assert np.unique(s.pos[:, 0]).size == 256


def test_reset_is_reproducible_per_seed():
    a = make_sim(num_envs=8, seed=1, spawn=NOISY_SPAWN).snapshot()
    b = make_sim(num_envs=8, seed=1, spawn=NOISY_SPAWN).snapshot()
    c = make_sim(num_envs=8, seed=2, spawn=NOISY_SPAWN).snapshot()
    np.testing.assert_array_equal(a.pos, b.pos)
    assert not np.array_equal(a.pos, c.pos)


def test_masked_reset_only_touches_selected_envs():
    sim = make_sim(num_envs=4)
    run(sim, 0, 1.0)
    before = sim.snapshot()
    sim.reset(np.array([1, 0, 1, 0]))
    after = sim.snapshot()
    np.testing.assert_array_equal(after.steps, [0, before.steps[1], 0, before.steps[3]])
    np.testing.assert_array_equal(after.pos[1], before.pos[1])
    assert after.pos[0, 1] == pytest.approx(sim.config.spawn.y)


def test_matches_numpy_reference_under_random_actions():
    num_envs, num_steps = 16, 400
    sim = make_sim(num_envs=num_envs, spawn=NOISY_SPAWN)
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 8, size=(num_steps, num_envs), dtype=np.int32)

    s = sim.snapshot()
    ref = [(s.pos[i], s.vel[i], s.theta[i], s.omega[i], FLYING) for i in range(num_envs)]
    for k in range(num_steps):
        sim.step(actions[k])
        ref = [
            reference.step(sim.config, *r[:4], actions[k, i]) if r[4] == FLYING else r
            for i, r in enumerate(ref)
        ]

    s = sim.snapshot()
    for i, (pos, vel, theta, omega, status) in enumerate(ref):
        assert s.status[i] == status
        np.testing.assert_allclose(s.pos[i], pos, rtol=1e-3, atol=0.05)
        np.testing.assert_allclose(s.vel[i], vel, rtol=1e-3, atol=0.01)
        assert s.theta[i] == pytest.approx(theta, abs=1e-3)
        assert s.omega[i] == pytest.approx(omega, abs=1e-4)
