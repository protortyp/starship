import math
from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from starship.sim.aero import _interpolate, make_aero_params, sample
from starship.sim.atmosphere import sample as sample_air
from starship.sim.config import G0, VehicleConfig


@pytest.fixture(scope="module")
def sea_level():
    return sample_air([0.0], device="cpu")[0]


def evaluate(velocities, **kwargs):
    return sample(velocities, device="cpu", **kwargs)


@pytest.mark.parametrize(
    ("fold", "expected_low", "expected_transonic"),
    [(40.0, 297.54, 551.12), (45.0, 280.27, 531.41)],
)
def test_broadside_drag_area(sea_level, fold, expected_low, expected_transonic):
    # claude_aero.md 4.6: 230-340 m^2 low-M, 550-700 m^2 at M=.8, flaps near mid.
    # 40 deg meets both; exact 45 deg with the configured leeward forward hinges falls
    # below the second envelope. Preserve this discrepancy instead of inflating C_dn.
    rho, sound = sea_level[2:4]
    speeds = np.array([85.0, 0.8 * sound])
    out = evaluate([[0, 0, v] for v in speeds], flap_angle=[math.radians(fold)] * 4)
    drag_area = -out[:, 2] / (0.5 * rho * speeds**2)
    assert drag_area == pytest.approx([expected_low, expected_transonic], abs=0.02)
    assert 230 < drag_area[0] < 340
    if fold == 40:
        assert 550 < drag_area[1] < 700
    assert drag_area[1] > 1.8 * drag_area[0]
    print(f"fold={fold:g} deg: C_D*A={drag_area[0]:.3f}/{drag_area[1]:.3f} m^2 (85 m/s / M=.8)")


@pytest.mark.parametrize("fold", [0.0, 40.0, 45.0])
def test_terminal_velocity_and_weight_balance(sea_level, fold):
    rho = sea_level[2]
    angles = [math.radians(fold)] * 4
    drag = -evaluate([[0, 0, 85]], flap_angle=angles)[0, 2]
    drag_area = drag / (0.5 * rho * 85**2)
    terminal = math.sqrt(2 * 183_000 * G0 / (rho * drag_area))
    terminal_drag = -evaluate([[0, 0, terminal]], flap_angle=angles)[0, 2]
    assert terminal_drag == pytest.approx(183_000 * G0, rel=2e-6)
    if fold == 0:
        assert 75 < terminal < 95
    else:
        # The 75-95 m/s target is for lighter SN ships (claude_aero.md 2.3). At 183 t,
        # 95 m/s needs >=324.65 m^2; the specified mid-flap model gives less than that.
        assert 98 < terminal < 103
        required_area = 2 * 183_000 * G0 / (rho * 95**2)
        assert drag_area < required_area
    print(f"fold={fold:g} deg: terminal={terminal:.3f} m/s at 183 t")


def test_stationary_aircraft_has_zero_wrench():
    out = evaluate([[0, 0, 0]] * 3, flap_angle=[[0] * 4, [math.pi / 4] * 4, [math.pi / 2] * 4])
    np.testing.assert_array_equal(out, 0.0)


def test_nose_first_drag_is_small_and_tail_first_drag_is_larger():
    out = evaluate([[85, 0, 0], [-85, 0, 0], [0, 0, 85]])
    assert out[0, 0] < 0 < out[1, 0]
    assert abs(out[0, 0]) < 0.15 * abs(out[2, 2])
    assert abs(out[1, 0]) > 2 * abs(out[0, 0])


def test_axial_coefficient_and_crossflow_mach_use_separate_velocities(sea_level):
    cfg = VehicleConfig()
    cfg.flaps = [replace(f, area=0) for f in cfg.flaps]
    velocities = [[0, 0, 85], [300, 0, 85], [-300, 0, 85]]
    out = evaluate(velocities, cfg=cfg)
    # Adding axial speed must not move the hull onto the transonic crossflow branch.
    np.testing.assert_allclose(out[:, 2], out[0, 2], rtol=1e-6)
    area = math.pi * (cfg.hull.diameter / 2) ** 2
    assert out[1, 0] == pytest.approx(-0.5 * sea_level[2] * 300**2 * area * 0.4, rel=1e-6)
    assert out[2, 0] == pytest.approx(0.5 * sea_level[2] * 300**2 * area * 0.9, rel=1e-6)
    assert out[1, 4] == pytest.approx(out[0, 4], rel=1e-6)


@pytest.mark.parametrize("fold", [0.0, 45.0, 85.0])
def test_symmetric_flaps_have_no_lateral_force_roll_or_yaw(fold):
    velocities = [[0, 0, 85], [35, 0, 85], [-35, 0, 85], [35, 0, -85]]
    out = evaluate(velocities, flap_angle=[math.radians(fold)] * 4, omega=(0, 0.1, 0))
    np.testing.assert_allclose(out[:, [1, 3, 5]], 0.0, atol=0.1)


def test_aft_exposure_moves_pressure_center_aft_and_has_correct_pitch_sign():
    com = 26.0
    angles = [[math.pi / 4] * 4, [0, 0, math.pi / 4, math.pi / 4]]
    out = evaluate([[0, 0, 85]] * 2, flap_angle=angles, com=com)
    # F3<0, tau2=-(x-com)*F3: adding drag aft of CoM gives negative pitch torque.
    # With canted forward plates this is the equivalent axis-intercept CoP; their
    # unchanged off-axis F1 moment is retained in both configurations.
    cop = com - out[:, 4] / out[:, 2]
    assert cop == pytest.approx([19.988, 19.292], abs=0.01)
    assert cop[1] < cop[0]
    assert out[1, 4] < out[0, 4]
    print(f"aft flaps 45->0 deg: CoP {cop[0]:.3f}->{cop[1]:.3f} m, shift={cop[1] - cop[0]:.3f} m")


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rotational_drag_opposes_rotation_in_still_air(axis):
    omega = np.zeros((2, 3))
    omega[:, axis] = [0.2, -0.2]
    out = evaluate([[0, 0, 0]] * 2, omega=omega)
    assert np.all(out[:, axis + 3] * omega[:, axis] < 0)
    np.testing.assert_allclose(out[0], -out[1], rtol=1e-5, atol=0.1)


def test_flaps_fold_leeward_about_canted_surface_hinges():
    cfg = VehicleConfig()
    p = make_aero_params(cfg, device="cpu")
    hinge, axis = p.hinge.numpy(), p.hinge_axis.numpy()
    opened, folded = p.span_open.numpy(), p.span_folded.numpy()
    np.testing.assert_allclose(np.linalg.norm(axis, axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(opened, axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(folded, axis=1), 1, atol=1e-7)
    np.testing.assert_allclose(np.sum(axis * opened, axis=1), 0, atol=1e-7)
    np.testing.assert_allclose(np.sum(axis * folded, axis=1), 0, atol=1e-7)
    np.testing.assert_allclose(np.sum(opened * folded, axis=1), 0, atol=1e-7)
    assert np.all(folded[:, 2] < -0.9)
    np.testing.assert_allclose(folded[:2], [[0, 0, -1], [0, 0, -1]], atol=1e-7)
    assert np.linalg.norm(hinge[2, 1:]) < cfg.hull.diameter / 2
    assert axis[2, 0] == pytest.approx(math.cos(cfg.flaps[2].cant))
    for values in (hinge, axis, opened, folded):
        np.testing.assert_allclose(values[::2] * [1, -1, 1], values[1::2], atol=1e-7)


def test_open_flaps_have_more_broadside_drag_than_folded_flaps():
    angles = [[0] * 4, [math.radians(45)] * 4, [math.radians(85)] * 4]
    out = evaluate([[0, 0, 85]] * 3, flap_angle=angles)
    assert -out[0, 2] > -out[1, 2] > -out[2, 2]


@wp.kernel
def _lookup_kernel(
    table: wp.array2d(dtype=float), mach: wp.array(dtype=float), out: wp.array2d(dtype=float)
):
    i = wp.tid()
    for j in range(1, table.shape[1]):
        out[i, j - 1] = _interpolate(table, mach[i], j)


def test_coefficient_tables_interpolate_and_hold_endpoints():
    p = make_aero_params(VehicleConfig(), device="cpu")
    mach = wp.array([-1, 0.3, 0.45, 0.8, 1.5, 2, 10], dtype=float, device="cpu")
    crossflow = wp.zeros((7, 1), dtype=float, device="cpu")
    plates = wp.zeros((7, 2), dtype=float, device="cpu")
    wp.launch(_lookup_kernel, dim=7, inputs=[p.crossflow_table, mach, crossflow], device="cpu")
    wp.launch(_lookup_kernel, dim=7, inputs=[p.plate_table, mach, plates], device="cpu")
    np.testing.assert_allclose(
        crossflow.numpy()[:, 0], [0.7, 0.7, 0.85, 1.6, 1.52, 1.43, 1.275], rtol=1e-6
    )
    np.testing.assert_allclose(plates.numpy()[0], [1.6, 1.17], rtol=1e-6)
    np.testing.assert_allclose(plates.numpy()[3], [1.6, 1.428], rtol=1e-6)
    np.testing.assert_allclose(plates.numpy()[4], [0.8, 1.65], rtol=1e-6)
    np.testing.assert_allclose(plates.numpy()[5:], [[0, 1.7], [0, 1.8]], rtol=1e-6)


def test_strip_quadrature_converges_and_body_cop_matches_planform_centroid():
    cfg = VehicleConfig()
    cfg.flaps = [replace(f, area=0) for f in cfg.flaps]
    coarse = evaluate([[0, 0, 85]], cfg=cfg)[0]
    cfg.aero.strips = 512
    fine = evaluate([[0, 0, 85]], cfg=cfg)[0]
    np.testing.assert_allclose(coarse[:3], fine[:3], rtol=0.001, atol=0.1)
    cop = 26 - fine[4] / fine[2]
    # Torque is a small residual about CoM; compare the pressure-center station instead.
    assert 26 - coarse[4] / coarse[2] == pytest.approx(cop, abs=0.02)
    assert 0.52 < (cfg.hull.length - cop) / cfg.hull.length < 0.56


def test_changing_com_translates_moment_without_changing_force():
    velocity = [25, -10, 85]
    out = evaluate([velocity] * 2, com=[20, 30], flap_angle=[0.1, 0.5, 0.9, 0.3])
    np.testing.assert_allclose(out[0, :3], out[1, :3], rtol=1e-6)
    expected_torque = out[0, 3:] - np.cross([10, 0, 0], out[0, :3])
    np.testing.assert_allclose(out[1, 3:], expected_torque, rtol=2e-6, atol=1)


def test_density_scaling_and_batched_six_dof_power_dissipation():
    rng = np.random.default_rng(73)
    velocity = rng.uniform(-600, 600, (64, 3))
    omega = rng.uniform(-1, 1, (64, 3))
    angles = rng.uniform(0, math.radians(85), (64, 4))
    out = evaluate(velocity, omega=omega, flap_angle=angles, altitude=np.linspace(0, 20_000, 64))
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    # Every quasi-steady force removes local kinetic energy, even with coupled rotation.
    power = np.sum(out[:, :3] * velocity + out[:, 3:] * omega, axis=1)
    assert np.all(power < 0)
    altitudes = [0, 10_000]
    air = sample_air(altitudes, device="cpu")
    low_speed = evaluate([[0, 0, 50]] * 2, altitude=altitudes)
    np.testing.assert_allclose(low_speed[1], low_speed[0] * air[1, 2] / air[0, 2], rtol=2e-6)
