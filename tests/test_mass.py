import dataclasses

import pytest

from starship.sim.config import MassConfig, VehicleConfig
from starship.sim.mass import make_mass_params, sample

CFG = VehicleConfig()
PARAMS = make_mass_params(CFG)
LANDING = [0.0, 0.0, 18_000.0, 5_000.0]  # full header tanks, empty mains


def props(tanks, cfg_params=PARAMS):
    return sample(cfg_params, [tanks], device="cpu")[0]


def test_total_mass_is_dry_plus_tanks():
    tanks = [1000.0, 300.0, 9000.0, 2500.0]
    assert props(tanks)[0] == pytest.approx(CFG.mass.dry_mass + sum(tanks), rel=1e-6)


def test_engine_ring_alone_has_known_com_and_inertia():
    m, x, r = 20_000.0, 2.5, CFG.engine.pivot_radius
    mass_cfg = MassConfig(
        dry_mass=m, engines_mass=m, engines_station=x, flaps_mass=0.0, aft_mass=0.0
    )
    p = make_mass_params(dataclasses.replace(CFG, mass=mass_cfg))
    mass, com, i1, i2, i3 = props([0.0] * 4, p)
    assert mass == pytest.approx(m)
    assert com == pytest.approx(x, rel=1e-6)
    assert i1 == pytest.approx(m * r * r, rel=1e-4)
    assert i2 == pytest.approx(m * r * r / 2, rel=1e-2)
    assert i3 == pytest.approx(m * r * r / 2, rel=1e-2)


def test_full_header_tanks_move_com_forward():
    assert props(LANDING)[1] > props([0.0] * 4)[1] + 0.5


def test_liquid_settles_at_tank_bottom():
    # A little main LOX sits at the bottom of the tank, aft of the dry centre of mass.
    assert props([20_000.0, 0.0, 0.0, 0.0])[1] < props([0.0] * 4)[1]


def test_inertia_positive_and_slender():
    _, _, i1, i2, i3 = props(LANDING)
    assert min(i1, i2, i3) > 0.0
    assert i2 > 5 * i1
    assert i3 > 5 * i1
    assert i2 == pytest.approx(i3, rel=0.05)


def test_landing_values_match_reference():
    # docs/starship.md: landing CoM 0.40-0.50 L from the engine plane, pitch/yaw inertia
    # 2.5e7-7e7 kg m^2, roll 1.8e6-3e6 kg m^2, for dry 160 t + 23 t header.
    mass, com, i1, i2, i3 = props(LANDING)
    assert mass == pytest.approx(183_000.0)
    assert 0.40 < com / CFG.hull.length < 0.50
    assert 2.5e7 < i2 < 7e7
    assert 2.5e7 < i3 < 7e7
    assert 1.8e6 < i1 < 3e6
