import numpy as np
import pytest

from starship.train import ObsConfig, RocketEnv
from starship.train.actions import ACTION_DIM
from starship.train.catch_probe import measure
from starship.train.encoding import CATCH_OBS_DIM, OBS_DIM
from tests.test_catch import at_plane
from tests.test_env import mirror_pair
from tests.test_policy_catch import policy_config


def test_geometry_features_match_physical_pin_and_fin_coordinates():
    cfg = policy_config()
    env = RocketEnv(1, cfg, ObsConfig(catch_geometry=True, mirror=False), device="cpu")
    env.sim.copy_from(at_plane(cfg, clearance=10), [0])
    obs = env.observe().numpy()[0]
    raw = measure(env.sim)
    assert env.obs_dim == CATCH_OBS_DIM == 38
    np.testing.assert_allclose(
        obs[OBS_DIM:],
        [
            raw["pin_left_above_support_m"] / 50,
            raw["pin_right_above_support_m"] / 50,
            raw["aft_fin_clearance_m"] / 50,
            raw["pin_alignment_error_m"] / 10,
            raw["closure_time_s"] / 2,
            raw["pin_descent_eta_s"] / 10,
        ],
        atol=1e-6,
    )


def test_extra_observations_mirror_with_vehicle_and_preserve_terminal_state():
    cfg = policy_config()
    cfg.spawn.pitch = 0.1
    cfg.spawn.offset = 2
    env = RocketEnv(2, cfg, ObsConfig(catch_geometry=True), device="cpu", seed=4)
    mirror_pair(env, world_axis=1)
    np.testing.assert_allclose(*env.observe().numpy(), atol=2e-6)
    fixture = at_plane(cfg)
    env.sim.copy_from(fixture, [0, 0])
    result = env.step(np.zeros((2, ACTION_DIM), dtype=np.float32))
    assert env.final_status.numpy()[0] == 2
    np.testing.assert_allclose(result.final_obs.numpy()[:, 32:34], 0, atol=2e-6)
    assert (result.obs.numpy()[:, 32:34] > 0).all()
    # A later nonterminal step must clear all 38 final-observation channels.
    env.step(np.zeros((2, ACTION_DIM), dtype=np.float32))
    np.testing.assert_array_equal(env.final_obs.numpy(), 0)


def test_geometry_features_are_optional_and_ground_features_are_zero():
    plain = RocketEnv(1, device="cpu")
    extended = RocketEnv(1, obs_config=ObsConfig(catch_geometry=True), device="cpu")
    assert plain.obs.shape == (1, 32)
    assert extended.obs.shape == (1, 38)
    np.testing.assert_array_equal(extended.observe().numpy()[:, 32:], 0)


def test_ascending_pin_eta_uses_sentinel_and_distant_features_are_bounded():
    cfg = policy_config()
    cfg.spawn.vertical_speed = 10
    env = RocketEnv(1, cfg, ObsConfig(catch_geometry=True, mirror=False), device="cpu")
    bodies = env.sim.state.body.numpy().copy()
    bodies["pos"][:, 2] = 20000
    env.sim.state.body.assign(bodies)
    obs = env.observe().numpy()[0]
    assert obs[37] == pytest.approx(-0.1)
    np.testing.assert_allclose(obs[32:34], 10)
    assert obs[34] == -10
