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


ENTRY_ALTITUDE = 50_000.0  # near entry max-q; the Mach number sets the coefficients, not q
REFERENCE_AREA = math.pi * 4.5**2


def coefficients(mach, alphas, flap_angle=(0.0,) * 4, cfg=None, com=26.0):
    """Rows of (C_A, C_N, C_L, C_D, X_cp/L from the nose) at the given angles of attack (deg)
    with the belly windward; reference area is the cross-section as in Li et al. [12]."""
    rho, sound = sample_air([ENTRY_ALTITUDE], device="cpu")[0, 2:4]
    speed = mach * sound
    q = 0.5 * rho * speed**2
    radians = np.radians(alphas)
    velocities = speed * np.stack([np.cos(radians), np.zeros_like(radians), np.sin(radians)], 1)
    out = evaluate(
        velocities, flap_angle=list(flap_angle), altitude=ENTRY_ALTITUDE, cfg=cfg, com=com
    )
    fx, fz, pitch = out[:, 0], out[:, 2], out[:, 4]
    ca, cn = -fx / (q * REFERENCE_AREA), -fz / (q * REFERENCE_AREA)
    cd = -(fx * np.cos(radians) + fz * np.sin(radians)) / (q * REFERENCE_AREA)
    cl = (fx * np.sin(radians) - fz * np.cos(radians)) / (q * REFERENCE_AREA)
    length = (cfg or VehicleConfig()).hull.length
    with np.errstate(divide="ignore", invalid="ignore"):
        xcp = (length - (com - pitch / fz)) / length
    return np.stack([ca, cn, cl, cd, xcp], 1)


@pytest.mark.parametrize(
    ("fold", "expected_low", "expected_transonic"),
    [(40.0, 297.54, 551.12), (45.0, 280.27, 531.41)],
)
def test_broadside_drag_area(sea_level, fold, expected_low, expected_transonic):
    # docs/starship.md: 230-340 m^2 low-M, 550-700 m^2 at M=.8, flaps near mid.
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
        # The 75-95 m/s target is for lighter SN ships (docs/starship.md). At 183 t,
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
    velocities = [[0, 0, 85], [200, 0, 85], [-200, 0, 85]]
    out = evaluate(velocities, cfg=cfg)
    # Adding axial speed must not move the hull onto the transonic crossflow branch. Tail-first
    # flow has no slender-body term, so its normal force is the crossflow force alone.
    np.testing.assert_allclose(out[2, 2], out[0, 2], rtol=1e-6)
    assert abs(out[1, 2]) > abs(out[0, 2])
    area = math.pi * (cfg.hull.diameter / 2) ** 2
    assert out[1, 0] == pytest.approx(-0.5 * sea_level[2] * 200**2 * area * 0.4, rel=1e-6)
    assert out[2, 0] == pytest.approx(0.5 * sea_level[2] * 200**2 * area * 0.9, rel=1e-6)
    assert out[2, 4] == pytest.approx(out[0, 4], rel=1e-6)


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
    mach = wp.array([-1, 0.3, 0.45, 0.8, 1.5, 2, 10, 30], dtype=float, device="cpu")
    crossflow = wp.zeros((8, 2), dtype=float, device="cpu")
    plates = wp.zeros((8, 2), dtype=float, device="cpu")
    regimes = wp.zeros((8, 4), dtype=float, device="cpu")
    wp.launch(_lookup_kernel, dim=8, inputs=[p.crossflow_table, mach, crossflow], device="cpu")
    wp.launch(_lookup_kernel, dim=8, inputs=[p.plate_table, mach, plates], device="cpu")
    wp.launch(_lookup_kernel, dim=8, inputs=[p.mach_table, mach, regimes], device="cpu")
    np.testing.assert_allclose(
        crossflow.numpy()[:, 0], [0.7, 0.7, 0.85, 1.6, 1.52, 1.43, 1.2725, 1.27], rtol=1e-6
    )
    np.testing.assert_allclose(crossflow.numpy()[:, 1], [0, 0, 0, 0, 1, 1, 1, 1], atol=1e-6)
    np.testing.assert_allclose(plates.numpy()[0], [1.6, 1.17], rtol=1e-6)
    np.testing.assert_allclose(plates.numpy()[3], [1.6, 1.428], rtol=1e-6)
    np.testing.assert_allclose(plates.numpy()[4], [0.8, 1.65], rtol=1e-6)
    np.testing.assert_allclose(plates.numpy()[5:], [[0, 1.7], [0, 1.8312], [0, 1.84]], rtol=1e-4)
    # Subsonic rows hold the Mach-independent model; the Newtonian weight is complete by M 3.
    np.testing.assert_allclose(regimes.numpy()[:4], [[1.276, 0, 0.4, 0.9]] * 4, rtol=1e-6)
    np.testing.assert_allclose(regimes.numpy()[4], [1.532, 0.21, 0.51, 1.45], rtol=1e-6)
    np.testing.assert_allclose(regimes.numpy()[6:, 1], 1.0, rtol=1e-6)
    np.testing.assert_allclose(regimes.numpy()[7], [1.838, 1, 0.06, 1.67], rtol=1e-6)


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


def test_slender_body_term_gives_small_angle_normal_force(sea_level):
    # TN D-6996 eq. 1 at 5 deg nose-first, 85 m/s: q A_b sin(10 deg) cos(2.5 deg) ~ 49 kN on top
    # of a few kN of crossflow force.
    cfg = VehicleConfig()
    cfg.flaps = [replace(f, area=0) for f in cfg.flaps]
    a = math.radians(5.0)
    out = evaluate([[85 * math.cos(a), 0, 85 * math.sin(a)]], cfg=cfg)
    q = 0.5 * sea_level[2] * 85**2
    potential = q * math.pi * (cfg.hull.diameter / 2) ** 2 * math.sin(2 * a) * math.cos(a / 2)
    assert -out[0, 2] == pytest.approx(potential, rel=0.2)
    assert -out[0, 2] > potential


def test_m12_coefficients_match_li_et_al_cfd():
    # docs/starship.md [12], M 12 flaps flat: C_L 4.2 -> 2.9 and C_D 8.5 -> 10.4 from 60 to 70
    # deg, L/D 0.5 -> 0.3, max L/D 1.5-1.7 at 10-20 deg, X_cp 0.57-0.62 L at 60 deg, C_L max
    # near 50 deg and C_D max near 90 deg.
    alphas = [10, 20, 40, 50, 60, 70, 90]
    ca, _, cl, cd, xcp = coefficients(12, alphas).T
    at = dict(zip(alphas, range(len(alphas)), strict=True))
    assert cl[at[60]] == pytest.approx(4.2, rel=0.1)
    assert cd[at[60]] == pytest.approx(8.5, rel=0.1)
    assert cl[at[70]] == pytest.approx(2.9, rel=0.1)
    assert cd[at[70]] == pytest.approx(10.4, rel=0.1)
    assert cl[at[60]] / cd[at[60]] == pytest.approx(0.5, abs=0.05)
    assert cl[at[70]] / cd[at[70]] == pytest.approx(0.3, abs=0.05)
    assert 1.5 < max(cl[at[10]] / cd[at[10]], cl[at[20]] / cd[at[20]]) < 1.8
    assert xcp[at[60]] == pytest.approx(0.57, abs=0.02)
    assert 0.55 < xcp[at[70]] < 0.62
    assert cl[at[50]] > max(cl[at[40]], cl[at[60]])
    assert cd[at[90]] > cd[at[70]]
    assert 11 < cd[at[90]] < 12.5
    # The windward ogive loads the axial force most at 60-70 deg, not at 0 deg.
    assert 0.43 < ca[at[60]] < 0.53
    assert ca[at[60]] > max(ca[at[90]], coefficients(12, [0])[0, 0])
    print(
        f"M 12: C_L(60)={cl[at[60]]:.2f} C_D(70)={cd[at[70]]:.2f} L/D(60,70)="
        f"{cl[at[60]] / cd[at[60]]:.2f}/{cl[at[70]] / cd[at[70]]:.2f} X_cp(60)={xcp[at[60]]:.3f}"
    )


def test_m2_coefficients_approach_li_et_al_cfd():
    # docs/starship.md [12], M 2: C_L 5 -> 3.4 and C_D 10 -> 12.6 from 60 to 70 deg. Newtonian
    # pressures under-predict lee-side suction at low supersonic Mach (research note), so the
    # drag tolerance is wider than at M 12.
    cl60, cd60 = coefficients(2, [60])[0, 2:4]
    cl70, cd70 = coefficients(2, [70])[0, 2:4]
    assert cl60 == pytest.approx(5.0, rel=0.1)
    assert cd60 == pytest.approx(10.0, rel=0.15)
    assert cl70 == pytest.approx(3.4, rel=0.1)
    assert cd70 == pytest.approx(12.6, rel=0.15)
    assert cd70 > coefficients(12, [70])[0, 3]
    print(f"M 2: C_L(60)={cl60:.2f} C_D(60)={cd60:.2f} C_L(70)={cl70:.2f} C_D(70)={cd70:.2f}")


def test_supersonic_eta_raises_broadside_crossflow_drag():
    # TR R-474: the finite-length factor reaches 1 in supersonic crossflow. On the cylinder the
    # change is exactly (1 - 0.63) C_dn q A_cyl; the ogive strips are Newtonian at M 5.
    cfg = VehicleConfig()
    cfg.flaps = [replace(f, area=0) for f in cfg.flaps]
    rho, sound = sample_air([ENTRY_ALTITUDE], device="cpu")[0, 2:4]
    speed = 5 * sound
    unity = evaluate([[0, 0, speed]], altitude=ENTRY_ALTITUDE, cfg=cfg)[0, 2]
    cfg.aero.crossflow_scale_supersonic = cfg.aero.crossflow_scale
    held = evaluate([[0, 0, speed]], altitude=ENTRY_ALTITUDE, cfg=cfg)[0, 2]
    slope = make_aero_params(cfg, device="cpu").strip_slope.numpy()
    cylinder = cfg.hull.diameter * np.sum(slope == 0) * cfg.hull.length / cfg.aero.strips
    expected = 0.37 * 1.30 * 0.5 * rho * speed**2 * cylinder
    assert held - unity == pytest.approx(expected, rel=1e-3)
    # Flaps flat, the vehicle's hypersonic broadside drag rises from 8.5 (constant eta) to the
    # Newtonian / Li et al. level, while the subsonic belly-flop keeps its calibrated value.
    cd90 = coefficients(12, [90])[0, 3]
    assert 11 < cd90 < 12.5
    assert coefficients(0.8, [90])[0, 3] == pytest.approx(9.53, abs=0.05)


def test_axial_drag_tables_follow_mach():
    machs = [0.5, 0.8, 1.1, 2, 3, 5, 12, 25]
    nose = np.array([coefficients(m, [0])[0, 0] for m in machs])
    tail = np.array([-coefficients(m, [180])[0, 0] for m in machs])
    # Nose-first: 0.4 subsonic, transonic peak, then wave + base + friction falling to ~0.28
    # (research note section 3). Tail-first: flat face 0.9 subsonic, ~0.9 Cp_max supersonic.
    np.testing.assert_allclose(nose, [0.40, 0.40, 0.65, 0.47, 0.34, 0.32, 0.30, 0.28], atol=0.02)
    assert np.all(np.diff(nose[2:]) < 0)
    np.testing.assert_allclose(tail[:2], 0.9, rtol=1e-6)
    assert np.all((tail[4:] > 1.6) & (tail[4:] < 1.8))
    assert np.all(tail > nose)
    print("C_A0 nose-first:", np.round(nose, 3), "tail-first:", np.round(tail, 3))


def test_folded_flaps_shift_hypersonic_pressure_center_as_in_the_panel_model():
    # Research note section 1.4 (M 12): forward flaps at 80 deg move X_cp aft by ~0.03 L and
    # cut C_D(90) by ~8 %; aft flaps at 80 deg move it forward by ~0.07 L and cut C_D(90) by
    # ~20 %. Li et al. [12] see the same trend (trim 69 deg -> 28 deg with forward flaps up).
    fold = math.radians(80)
    flat = coefficients(12, [60, 90])
    forward = coefficients(12, [60, 90], flap_angle=(0, 0, fold, fold))
    aft = coefficients(12, [60, 90], flap_angle=(fold, fold, 0, 0))
    assert forward[0, 4] - flat[0, 4] == pytest.approx(0.03, abs=0.01)
    assert aft[0, 4] - flat[0, 4] == pytest.approx(-0.07, abs=0.015)
    assert 1 - forward[1, 3] / flat[1, 3] == pytest.approx(0.08, abs=0.03)
    assert 1 - aft[1, 3] / flat[1, 3] == pytest.approx(0.20, abs=0.04)


def test_hull_shadow_removes_the_leeward_flap_force_at_hypersonic_speed():
    # Sideslip towards +b2 with both aft flaps folded to the lee: the port plate hangs in the
    # hull's shadow. Subsonic flow keeps both plates loaded (wake pressure is in C_N90).
    cfg = VehicleConfig()
    cfg.flaps = [f if i < 2 else replace(f, area=0) for i, f in enumerate(cfg.flaps)]
    hull = VehicleConfig()
    hull.flaps = [replace(f, area=0) for f in hull.flaps]
    rho, sound = sample_air([ENTRY_ALTITUDE], device="cpu")[0, 2:4]
    angles = (math.pi / 2, math.pi / 2, 0, 0)
    for mach, plates, cn90 in [(12, 1, 1.8312), (0.5, 2, 1.17)]:
        speed = mach * sound
        for side in (1, -1):
            velocity = [[0, side * speed, 0]]
            kw = {"flap_angle": angles, "altitude": ENTRY_ALTITUDE}
            flaps = evaluate(velocity, cfg=cfg, **kw)[0] - evaluate(velocity, cfg=hull, **kw)[0]
            expected = plates * 0.5 * rho * speed**2 * 40 * cn90
            assert -side * flaps[1] == pytest.approx(expected, rel=1e-3)


def test_blends_are_continuous_through_the_transonic_and_supersonic_range():
    machs = np.arange(0.6, 4.0, 0.02)
    for alpha in (0, 40, 70, 90, 150, 180):
        angle = (math.radians(45),) * 4
        rows = np.array([coefficients(m, [alpha], flap_angle=angle)[0] for m in machs])
        assert np.isfinite(rows[:, :4]).all()
        drag = rows[:, 3]
        assert np.all(drag > 0)
        # The steepest gradient is the measured transonic C_dn rise below M_n 1; the eta,
        # Newtonian and axial-table blends above M 1.1 must stay gentler than that.
        steps = np.abs(np.diff(drag) / drag[:-1])
        assert np.all(steps < 0.06)
        assert np.all(steps[machs[:-1] >= 1.1] < 0.02)
