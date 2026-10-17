import numpy as np
import pytest

from starship.sim.atmosphere import sample

# U.S. Standard Atmosphere 1976 tables (NASA-TM-X-74335), geometric altitude.
TABLE = [
    # z (m), T (K), p (Pa), rho (kg/m^3)
    (0.0, 288.150, 101325.0, 1.2250),
    (1000.0, 281.651, 89874.6, 1.1117),
    (5000.0, 255.676, 54048.3, 0.73643),
    (11000.0, 216.774, 22699.9, 0.36480),
    (20000.0, 216.650, 5529.3, 0.088910),
    (32000.0, 228.490, 889.06, 0.013555),
    (50000.0, 270.650, 79.779, 0.0010269),
    (80000.0, 198.639, 1.0524, 1.8458e-5),
]


@pytest.mark.parametrize(("z", "t", "p", "rho"), TABLE)
def test_matches_standard_tables(z, t, p, rho):
    temperature, pressure, density, _, _ = sample([z], device="cpu")[0]
    assert temperature == pytest.approx(t, rel=1e-4)
    assert pressure == pytest.approx(p, rel=1e-3)
    assert density == pytest.approx(rho, rel=1e-3)


def test_sea_level_speed_of_sound_and_viscosity():
    _, _, _, a, mu = sample([0.0], device="cpu")[0]
    assert a == pytest.approx(340.294, rel=1e-4)
    assert mu == pytest.approx(1.7894e-5, rel=1e-3)


def test_continuous_across_layer_boundaries():
    z = np.linspace(0.0, 85000.0, 20001)
    out = sample(z, device="cpu")
    step = np.abs(np.diff(out, axis=0)) / np.abs(out[:-1])
    assert step.max() < 2e-3
