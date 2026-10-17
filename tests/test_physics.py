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
from starship.sim.config import G0
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
def test_drag_balances_gravity_at_terminal_velocity(theta, cd_attr, area_attr):
    cfg = SimConfig()
    r = cfg.rocket
    y = 1000.0
    rho = reference.density(cfg, y)
    v_terminal = math.sqrt(
        2 * r.mass * cfg.world.gravity / (rho * getattr(r, cd_attr) * getattr(r, area_attr))
    )
    sim = make_sim(spawn=SpawnConfig(y=y, vy=-v_terminal, theta=theta))
    sim.step(0)
    # Without drag the rocket would gain g * policy_dt ~ 0.5 m/s in this step.
    assert sim.snapshot().vel[0, 1] == pytest.approx(-v_terminal, abs=0.01)


def test_main_engine_follows_rocket_equation_along_body_axis():
    theta = 0.3
    sim = make_sim(world=ZERO_G_VACUUM, spawn=SpawnConfig(vy=0.0, theta=theta))
    t = 2.0
    run(sim, MAIN, t)
    s = sim.snapshot()
    r = sim.config.rocket
    burned = r.main_mass_flow * t
    assert s.fuel[0] == pytest.approx(r.fuel_mass - burned, rel=1e-5)
    # Tsiolkovsky: dv = Isp g0 ln(m0 / m1); the explicit mass update lags by one substep.
    dv = r.main_isp * G0 * math.log(r.mass / (r.mass - burned))
    expected = dv * np.array([-math.sin(theta), math.cos(theta)])
    np.testing.assert_allclose(s.vel[0], expected, rtol=1e-3)
    assert s.omega[0] == 0.0
    assert s.theta[0] == pytest.approx(theta)


def test_empty_tank_disables_thrusters():
    rocket = replace(SimConfig().rocket, fuel_mass=0.0)
    sim = make_sim(rocket=rocket, world=ZERO_G_VACUUM, spawn=SpawnConfig(vy=0.0))
    run(sim, MAIN | LEFT, 1.0)
    s = sim.snapshot()
    np.testing.assert_array_equal(s.vel[0], 0.0)
    assert s.omega[0] == 0.0


def test_thrust_stops_when_fuel_runs_out():
    rocket = SimConfig().rocket
    rocket = replace(rocket, fuel_mass=rocket.main_mass_flow * 1.0)
    sim = make_sim(rocket=rocket, world=ZERO_G_VACUUM, spawn=SpawnConfig(vy=0.0, theta=0.0))
    run(sim, MAIN, 1.0)
    v_burnout = sim.snapshot().vel[0, 1]
    run(sim, MAIN, 1.0)
    s = sim.snapshot()
    assert s.fuel[0] == 0.0
    assert s.vel[0, 1] == pytest.approx(v_burnout, rel=1e-3)


@pytest.mark.parametrize(("action", "sign"), [(LEFT, -1.0), (RIGHT, 1.0)])
def test_side_thrusters_rotate_in_opposite_directions(action, sign):
    # A huge Isp keeps the mass, and so the inertia, constant for a closed-form check.
    rocket = replace(SimConfig().rocket, side_isp=1e12)
    sim = make_sim(rocket=rocket, world=ZERO_G_VACUUM, spawn=SpawnConfig(vy=0.0, theta=0.0))
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
    spawn = SpawnConfig(**{"y": 25.5, "vy": -1.0, "theta": 0.0, **spawn_overrides})
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
        {"theta": math.radians(20.0), "y": 25.5},
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
    ref = [(s.pos[i], s.vel[i], s.theta[i], s.omega[i], s.fuel[i], FLYING) for i in range(num_envs)]
    for k in range(num_steps):
        sim.step(actions[k])
        ref = [
            reference.step(sim.config, *r[:5], actions[k, i]) if r[5] == FLYING else r
            for i, r in enumerate(ref)
        ]

    s = sim.snapshot()
    for i, (pos, vel, theta, omega, fuel, status) in enumerate(ref):
        assert s.status[i] == status
        np.testing.assert_allclose(s.pos[i], pos, rtol=1e-3, atol=0.05)
        np.testing.assert_allclose(s.vel[i], vel, rtol=1e-3, atol=0.01)
        assert s.theta[i] == pytest.approx(theta, abs=1e-3)
        assert s.omega[i] == pytest.approx(omega, abs=1e-4)
        assert s.fuel[i] == pytest.approx(fuel, rel=1e-4, abs=0.5)


def test_spawn_components_use_independent_draws():
    s = make_sim(num_envs=64, spawn=NOISY_SPAWN).snapshot()
    sp = NOISY_SPAWN
    x_unit = (s.pos[:, 0] - sp.x) / sp.x_noise
    vx_unit = (s.vel[:, 0] - sp.vx) / sp.vx_noise
    assert not np.allclose(x_unit, vx_unit)


def test_successive_resets_draw_new_spawns():
    sim = make_sim(num_envs=4, spawn=NOISY_SPAWN)
    first = sim.snapshot()
    sim.reset()
    second = sim.snapshot()
    assert not np.any(first.pos[:, 0] == second.pos[:, 0])


def touchdown_sim(auto_reset: bool) -> Simulator:
    spawn = SpawnConfig(y=25.5, vy=-1.0, theta=0.0)
    return Simulator(
        1, replace(SimConfig(), world=ZERO_G_VACUUM, spawn=spawn), "cpu", auto_reset=auto_reset
    )


def step_until_done(sim: Simulator, max_steps: int = 100) -> int:
    for k in range(max_steps):
        sim.step(0)
        if sim.terminal_snapshot().done[0]:
            return k + 1
    raise AssertionError("episode did not end")


def test_auto_reset_records_terminal_state_and_respawns():
    sim = touchdown_sim(auto_reset=True)
    n = step_until_done(sim)
    term = sim.terminal_snapshot()
    state = sim.snapshot()
    assert term.status[0] == LANDED
    assert term.steps[0] == n
    assert term.pos[0, 1] == pytest.approx(0.5 * sim.config.rocket.length, abs=0.02)
    assert state.status[0] == FLYING
    assert state.steps[0] == 0
    assert state.pos[0, 1] == pytest.approx(sim.config.spawn.y)

    sim.step(0)
    assert sim.terminal_snapshot().done[0] == 0
    assert sim.snapshot().steps[0] == 1


def test_done_is_set_only_on_the_terminating_step_without_auto_reset():
    sim = touchdown_sim(auto_reset=False)
    step_until_done(sim)
    assert sim.snapshot().status[0] == LANDED
    sim.step(0)
    assert sim.terminal_snapshot().done[0] == 0
    assert sim.snapshot().status[0] == LANDED


def test_auto_reset_runs_many_episodes_independently():
    # Falls of 100-300 m at ~50 m/s take 2-6 s, so 20 s fits at least three episodes per env.
    spawn = SpawnConfig(y=200.0, vy=-50.0, y_noise=100.0, theta_noise=0.5)
    sim = Simulator(32, replace(SimConfig(), spawn=spawn), "cpu", auto_reset=True)
    episodes = np.zeros(32, dtype=int)
    for _ in range(400):
        sim.step(0)
        episodes += sim.terminal_snapshot().done
    assert (sim.snapshot().status == FLYING).all()
    assert episodes.min() >= 3
    # Spawn noise desynchronizes the envs, so they complete different numbers of episodes.
    assert episodes.std() > 0


@pytest.mark.parametrize("fraction", [0.0, 0.3, 1.0])
def test_easy_spawn_fraction(fraction):
    sim = make_sim(num_envs=2000, easy_spawn_fraction=fraction)
    easy = np.abs(sim.snapshot().theta) < 1.0
    assert easy.mean() == pytest.approx(fraction, abs=0.04)


def test_mirrored_spawns_flip_half_the_rockets():
    spawn = SpawnConfig(x=200.0, vx=15.0, x_noise=50.0, vx_noise=5.0, theta_noise=0.3, mirror=True)
    s = make_sim(num_envs=2000, spawn=spawn).snapshot()
    nose_left = s.theta > 0
    assert nose_left.mean() == pytest.approx(0.5, abs=0.04)
    # Mirroring flips position, velocity and attitude together.
    assert np.all((s.pos[:, 0] > 0) == nose_left)
    assert np.all((s.vel[:, 0] > 0) == nose_left)


def test_mirrored_dynamics_are_symmetric():
    base = SpawnConfig(x=120.0, vx=8.0, y=2000.0, vy=-60.0, theta=1.3, omega=0.05)
    mirrored = replace(base, x=-120.0, vx=-8.0, theta=-1.3, omega=-0.05)
    a, b = make_sim(spawn=base), make_sim(spawn=mirrored)
    for k in range(60):
        action = [MAIN, LEFT, RIGHT, MAIN | LEFT][k % 4]
        swapped = action ^ (LEFT | RIGHT) if action & (LEFT | RIGHT) in (LEFT, RIGHT) else action
        a.step(action)
        b.step(swapped)
    sa, sb = a.snapshot(), b.snapshot()
    np.testing.assert_allclose(sa.pos[0] * [-1, 1], sb.pos[0], rtol=1e-4, atol=1e-3)
    assert sa.theta[0] == pytest.approx(-sb.theta[0], abs=1e-5)
