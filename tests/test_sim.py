from dataclasses import replace

import numpy as np
import pytest

from starship.sim import CRASHED, FLYING, LANDED, SimConfig, Simulator
from starship.sim.mass import make_mass_params, sample

DEVICE = "cpu"


def body_axis_z(rot: np.ndarray) -> np.ndarray:
    """World z component of the body axis b1 for (N, 4) quaternions."""
    x, y, z, w = rot.T
    return 2.0 * (x * z - w * y)


def config(**spawn) -> SimConfig:
    cfg = SimConfig()
    cfg.spawn = replace(cfg.spawn, **spawn)
    return cfg


def belly_flop_controller(sim: Simulator) -> np.ndarray:
    """PD on the pitch of the body axis: open the aft flaps when the nose rises."""
    s = sim.snapshot()
    u = np.clip(np.radians(60) * body_axis_z(s.rot) + np.radians(60) * s.omega[:, 1], -0.7, 0.7)
    aft, fwd = np.radians(52) - u, np.radians(20) + u
    return np.stack([aft, aft, fwd, fwd], axis=1).astype(np.float32)


def test_spawn_is_belly_down_and_horizontal():
    sim = Simulator(16, config(), device=DEVICE, seed=1)
    s = sim.snapshot()
    assert np.allclose(body_axis_z(s.rot), 0.0, atol=1e-5)
    x, y = s.rot[:, 0], s.rot[:, 1]
    belly_z = 1.0 - 2.0 * (x * x + y * y)  # world z of b3
    assert np.allclose(belly_z, -1.0, atol=1e-5)
    assert np.allclose(s.pos[:, 2], 3000.0)
    assert np.all(s.status == FLYING)


def test_fall_does_not_depend_on_heading():
    # Headings are random per env; the vertical motion must be identical.
    sim = Simulator(8, config(), device=DEVICE, seed=2)
    for _ in range(50):
        sim.step()
    s = sim.snapshot()
    assert np.ptp(s.pos[:, 2]) < 0.05
    assert np.ptp(s.vel[:, 2]) < 0.01
    assert np.ptp(body_axis_z(s.rot)) < 1e-3


def test_flaps_hold_the_belly_flop_at_flight_speeds():
    sim = Simulator(1, config(altitude=10_000.0, vertical_speed=-100.0), device=DEVICE)
    t_500 = speed_1km = None
    worst = 0.0
    for _ in range(1300):
        sim.step(flaps=belly_flop_controller(sim))
        s = sim.snapshot()
        if s.time[0] > 10.0:
            worst = max(worst, abs(body_axis_z(s.rot)[0]))
        if speed_1km is None and s.pos[0, 2] < 1000.0:
            speed_1km = -s.vel[0, 2]
        if s.pos[0, 2] < 500.0:
            t_500 = s.time[0]
            break
    # docs/starship.md: ~100 m/s near the ground at 183 t; SN15 fell 10 km -> 500 m in ~100 s at a
    # lighter ~120 t (scaled by sqrt(mass): ~80 s).
    assert worst < 0.15  # within ~9 deg of horizontal
    assert 95.0 < speed_1km < 125.0
    assert 60.0 < t_500 < 90.0


def test_full_thrust_upright_climbs_at_expected_acceleration():
    sim = Simulator(1, config(altitude=2000.0, vertical_speed=0.0, pitch=0.0), device=DEVICE)
    sim.set_commands(engine_on=(1, 1, 1), throttle=(1, 1, 1))
    for _ in range(25):  # past the 2 s spool-up
        sim.step()
    v0 = sim.snapshot().vel[0, 2]
    sim.step()
    s = sim.snapshot()
    accel = (s.vel[0, 2] - v0) / 0.1
    mass = 160_000.0 + s.tanks[0].sum()
    thrust = 3 * (2.39e6 - 79_500.0 * 1.33)  # ambient pressure at ~2 km
    drag_guess = 0.5 * 1.0 * s.vel[0, 2] ** 2 * 63.6 * 0.9  # tail-first axial drag
    expected = (thrust - drag_guess) / mass - 9.8
    assert accel == pytest.approx(expected, rel=0.05)


def test_burn_consumes_propellant_at_rated_flow():
    sim = Simulator(1, config(altitude=5000.0, vertical_speed=0.0, pitch=0.0), device=DEVICE)
    before = sim.snapshot().tanks[0].sum()
    sim.set_commands(engine_on=(1, 0, 0), throttle=(1, 1, 1))
    for _ in range(50):
        sim.step()
    used = before - sim.snapshot().tanks[0].sum()
    # Thrust fraction = spool (0 -> 1 over 2 s) x throttle (0.4 -> 1 at 0.4/s): integrated over
    # 5 s that is 3.8875 s at the rated 702.3 kg/s.
    assert used == pytest.approx(702.3 * 3.8875, rel=0.01)


def final_status(cfg: SimConfig, steps: int = 200) -> int:
    sim = Simulator(1, cfg, device=DEVICE)
    for _ in range(steps):
        sim.step()
        if sim.snapshot().status[0] != FLYING:
            break
    return int(sim.snapshot().status[0])


def test_gentle_upright_touchdown_lands_and_fast_one_crashes():
    sim = Simulator(1, config(), device=DEVICE)
    com = sample(make_mass_params(sim.config.vehicle), sim.snapshot().tanks, device=DEVICE)[0, 1]
    hover = float(com) + 0.05  # tail 5 cm above the ground
    assert final_status(config(altitude=hover, vertical_speed=-1.0, pitch=0.0)) == LANDED
    assert final_status(config(altitude=200.0, vertical_speed=-30.0, pitch=0.0)) == CRASHED
    assert final_status(config(altitude=hover, vertical_speed=-1.0, pitch=0.3)) == CRASHED


def test_auto_reset_respawns_and_keeps_terminal_state():
    cfg = config(altitude=300.0, vertical_speed=-60.0)
    sim = Simulator(4, cfg, device=DEVICE, auto_reset=True)
    for _ in range(100):
        sim.step()
        t = sim.terminal_snapshot()
        if t.done.any():
            break
    assert t.done.all()
    assert np.all(t.status == CRASHED)
    assert np.all(t.pos[:, 2] < 30.0)
    s = sim.snapshot()
    assert np.all(s.status == FLYING)
    assert np.allclose(s.pos[:, 2], 300.0)


def test_empty_tanks_give_no_thrust():
    cfg = config(altitude=5000.0, vertical_speed=0.0, pitch=0.0, header_fill=0.0, main_fill=0.0)
    dry = Simulator(1, cfg, device=DEVICE)
    coast = Simulator(1, cfg, device=DEVICE)
    dry.set_commands(engine_on=(1, 1, 1), throttle=(1, 1, 1))
    for _ in range(30):
        dry.step()
        coast.step()
    assert dry.snapshot().vel[0, 2] == pytest.approx(coast.snapshot().vel[0, 2], abs=1e-3)


def test_spinning_touchdown_crashes():
    sim = Simulator(1, config(), device=DEVICE)
    com = sample(make_mass_params(sim.config.vehicle), sim.snapshot().tanks, device=DEVICE)[0, 1]
    cfg = config(altitude=float(com) + 0.05, vertical_speed=-1.0, pitch=0.0)
    sim = Simulator(1, cfg, device=DEVICE)
    body = sim.state.body.numpy()
    body["omega"][0] = (0.5, 0.0, 0.0)  # 29 deg/s roll: above the landing limit, below tumbling
    sim.state.body.assign(body)
    for _ in range(20):
        sim.step()
        if sim.snapshot().status[0] != FLYING:
            break
    assert sim.snapshot().status[0] == CRASHED


@pytest.mark.parametrize("tilt_deg", [100.0, 110.0, 130.0, 160.0])
def test_nose_down_contact_finds_the_ogive_surface(tilt_deg):
    # Place the vehicle so the true lowest point of the ogive is 5 cm below ground: the first
    # step must end the episode.
    cfg = SimConfig()
    h = cfg.vehicle.hull
    r0, nose = h.diameter / 2, h.nose_length
    big_r = (nose**2 + r0**2) / (2 * r0)
    tilt = np.radians(tilt_deg)
    a_z, s = np.cos(tilt), np.sin(tilt)
    u = np.linspace(0.0, nose, 20001)
    r = np.sqrt(big_r**2 - u**2) + r0 - big_r
    lowest = np.min((h.length - nose + u) * a_z - np.maximum(r, 0) * s)
    sim = Simulator(1, config(pitch=tilt), device=DEVICE)
    com = sample(make_mass_params(cfg.vehicle), sim.snapshot().tanks, device=DEVICE)[0, 1]
    # Height of the CoM so that the lowest hull point sits at -5 cm, with all other points above.
    lowest_rel = min(lowest, -r0 * s, (h.length - nose) * a_z - r0 * s, h.length * a_z)
    altitude = -0.05 - (lowest_rel - com * a_z)
    sim = Simulator(
        1, config(pitch=tilt, altitude=float(altitude), vertical_speed=0.0), device=DEVICE
    )
    sim.step()
    assert sim.snapshot().status[0] == CRASHED


def test_copied_states_evolve_identically_to_the_original():
    source = Simulator(3, config(), device=DEVICE, seed=5)
    for _ in range(20):
        source.step(flaps=belly_flop_controller(source))
    copies = Simulator(4, config(), device=DEVICE, seed=9)
    copies.copy_from(source, [1, 1, 2, 0])
    assert np.allclose(copies.snapshot().pos, source.snapshot().pos[[1, 1, 2, 0]])
    for _ in range(20):
        source.step(flaps=belly_flop_controller(source))
        copies.step(flaps=belly_flop_controller(copies))
    s, c = source.snapshot(), copies.snapshot()
    assert np.allclose(c.pos, s.pos[[1, 1, 2, 0]], atol=1e-3)
    assert np.allclose(c.rot, s.rot[[1, 1, 2, 0]], atol=1e-5)
    assert np.allclose(c.time, s.time[[1, 1, 2, 0]])


def test_copy_within_the_same_simulator():
    sim = Simulator(4, config(), device=DEVICE, seed=3)
    for _ in range(5):
        sim.step()
    before = sim.snapshot()
    sim.copy_from(sim, [3, 2, 1, 0])
    after = sim.snapshot()
    assert np.allclose(after.pos, before.pos[::-1])
    assert np.allclose(after.rot, before.rot[::-1])


def test_tumbling_ends_the_episode():
    from starship.sim import TUMBLED

    # Passive belly-flop with aft flaps open pitches nose-down past the limit.
    sim = Simulator(1, config(altitude=5000.0), device=DEVICE)
    for _ in range(300):
        sim.step(flaps=(0.0, 0.0, 1.4, 1.4))
        if sim.snapshot().status[0] != FLYING:
            break
    assert sim.snapshot().status[0] == TUMBLED
    # Spinning faster than the rate limit also counts.
    sim = Simulator(1, config(altitude=5000.0, angular_rate_noise=1.5), device=DEVICE, seed=4)
    sim.step()
    assert sim.terminal_snapshot().status[0] == TUMBLED or sim.snapshot().status[0] == TUMBLED


def test_spawn_with_engines_lit():
    cfg = config(altitude=500.0, pitch=0.0, vertical_speed=-40.0, lit_throttle=0.7)
    sim = Simulator(1, cfg, device=DEVICE)
    assert np.allclose(sim.snapshot().spool, 1.0)
    sim.step(engine_on=(1, 1, 1), throttle=(0.7, 0.7, 0.7))
    assert sim.snapshot().vel[0, 2] > -40.0  # already decelerating, no 2 s spool-up


@pytest.mark.parametrize(
    ("omega_deg", "tumbles"),
    [((60, 0, 0), False), ((120, 0, 0), True), ((0, 50, 0), True), ((0, 0, 30), False)],
)
def test_roll_has_its_own_limit(omega_deg, tumbles):
    from starship.sim import TUMBLED

    sim = Simulator(1, config(altitude=3000.0, pitch=0.0, vertical_speed=0.0), device=DEVICE)
    body = sim.state.body.numpy()
    body["omega"][0] = tuple(np.radians(omega_deg))
    sim.state.body.assign(body)
    sim.step()
    assert (sim.snapshot().status[0] == TUMBLED) == tumbles
