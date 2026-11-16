import numpy as np
import pytest

from starship.train.learning_audit import channel_metrics, discounted_returns, value_metrics


def test_critic_audit_detects_bias_even_when_explained_variance_is_perfect():
    metrics = value_metrics(np.array([11.0, 12.0, 13.0]), np.array([1.0, 2.0, 3.0]))
    assert metrics["explained_variance"] == pytest.approx(1)
    assert metrics["bias"] == pytest.approx(10)
    assert metrics["rmse"] == pytest.approx(10)
    assert value_metrics([0.0, 1.0], [2.0, 2.0])["explained_variance"] is None


def test_monte_carlo_masks_autoreset_rewards_and_has_no_bootstrap():
    rewards = np.array([[1.0, 2.0], [3.0, 4.0], [999.0, 6.0]])
    valid = np.array([[True, True], [True, True], [False, True]])
    result = discounted_returns(rewards, valid, 0.5)
    np.testing.assert_allclose(result, [[2.5, 5.5], [3.0, 7.0], [0.0, 6.0]])


def test_channel_report_distinguishes_raw_scale_from_normalized_outlier():
    raw = np.array([[0.1, 1.0], [0.2, 2.0]])
    rows = channel_metrics(raw, raw * [100.0, 1.0])
    assert rows[0]["raw_max"] == 0.2
    assert rows[0]["normalized_max"] == 20
    assert rows[0]["normalized_abs_gt_10"] == 0.5
    assert rows[1]["normalized_abs_gt_10"] == 0


def test_nonfinite_input_channel_is_reported_without_crashing():
    rows = channel_metrics(np.array([[np.nan]]), np.array([[np.inf]]))
    assert rows[0]["raw_nonfinite"] == rows[0]["normalized_nonfinite"] == 1
    assert rows[0]["raw_max"] is None


def test_braking_estimate_is_an_explicit_constant_mass_comparison():
    from starship.train.learning_audit import braking_estimates

    result = braking_estimates(np.array([[10.0, -10.0, 1.0, 20.0, 10.0]]))
    assert result["engines"]["1"]["stopping_distance_m_median"] == pytest.approx(4.8)
    assert result["engines"]["1"]["height_margin_m_median"] == pytest.approx(5.2)
    assert result["engines"]["3"]["stopping_distance_m_median"] < 4.8
    assert "not a feasibility proof" in result["assumptions"]


@pytest.mark.parametrize("catch_enabled", [False, True])
@pytest.mark.parametrize(("pitch", "fill"), [(0.0, 1.0), (0.3, 0.4)])
def test_initial_braking_height_uses_the_active_landing_target(catch_enabled, pitch, fill):
    import warp as wp

    from starship.sim import SimConfig, Simulator
    from starship.sim.mass import make_mass_params, sample
    from starship.train.learning_audit import initial_physics

    cfg = SimConfig()
    cfg.catch.enabled = catch_enabled
    cfg.catch.height = 81.0
    cfg.spawn.pitch = pitch
    cfg.spawn.header_fill = fill
    sim = Simulator(1, cfg, device="cpu")
    com = sample(make_mass_params(cfg.vehicle), sim.snapshot().tanks, device="cpu")[0, 1]
    body = sim.state.body.numpy()[0]
    axis = np.asarray(wp.quat_rotate(wp.quat(*body["rot"]), wp.vec3(1.0, 0.0, 0.0)))
    base_height = body["pos"][2] - axis[2] * com
    expected = base_height
    if catch_enabled:
        expected += axis[2] * sim.params.catcher.station - cfg.catch.height
    output = wp.zeros((1, 5), dtype=float, device="cpu")
    wp.launch(initial_physics, dim=1, inputs=[sim.params, sim.state, output], device="cpu")
    assert output.numpy()[0, 0] == pytest.approx(expected, abs=5e-5)


@pytest.mark.parametrize("mass", [0.0, -1.0, np.nan, np.inf])
def test_braking_estimate_rejects_incomplete_or_invalid_physics(mass):
    from starship.train.learning_audit import braking_estimates

    with pytest.raises(ValueError, match="finite physics and positive mass"):
        braking_estimates(np.array([[10.0, -10.0, mass, 20.0, 10.0]]))


def test_audit_names_follow_encoding_width_and_geometry_order():
    from starship.train.encoding import CATCH_OBS_DIM, OBSERVATION_NAMES

    assert len(OBSERVATION_NAMES) == CATCH_OBS_DIM
    assert len(set(OBSERVATION_NAMES)) == CATCH_OBS_DIM
    assert (
        channel_metrics(np.zeros((1, 38)), np.zeros((1, 38)))[35]["name"] == "pin_horizontal_error"
    )


def test_tilt_audit_separates_failure_region_and_handles_endpoints():
    from starship.train.learning_audit import initial_tilt_metrics

    obs = np.zeros((4, 32))
    obs[:, 0] = [1.0000001, np.cos(0.12), np.cos(0.13), -1.0000001]
    result = initial_tilt_metrics(
        obs, np.array([1, 2, 2, 1]), np.array([1, 10, 20, 4]), np.array([1, 0, 0, 4])
    )
    bins = result["bins"]
    assert sum(row["episodes"] for row in bins) == 4
    assert bins[0]["catch_rate"] == bins[-1]["catch_rate"] == 1
    assert bins[2]["episodes"] == 2
    assert bins[2]["catch_rate"] == 0
    assert bins[2]["critic_initial_states"]["bias"] == 15
    assert bins[2]["critic_initial_states"]["explained_variance"] is None
    assert bins[1]["catch_rate"] is bins[1]["critic_initial_states"] is None


def test_action_audit_detects_saturation_despite_unchanged_gaussian_noise():
    from starship.train.actions import CONTINUOUS
    from starship.train.learning_audit import continuous_action_metrics

    means = np.zeros((3, len(CONTINUOUS)))
    means[:, -1] = [-12, 8, 13]
    rows = continuous_action_metrics(means, np.full(len(CONTINUOUS), 0.02))
    assert rows[0]["saturated_mean_fraction"] == 0
    assert rows[-1]["saturated_mean_fraction"] == 1
    assert rows[0]["pre_tanh_std"] == rows[-1]["pre_tanh_std"] == 0.02
    key = "sigma_interval_half_width_range_fraction_median"
    assert rows[0][key] == pytest.approx(np.tanh(0.02) / 2)
    assert rows[-1][key] < rows[0][key] * 1e-6
    assert rows[0]["sigma_interval_half_width_actuator_units_median"] == pytest.approx(
        rows[0][key] * 0.6
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("stochastic", [False, True])
def test_complete_audit_repeats_on_one_backend(tmp_path, device, stochastic):
    import torch
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf

    from starship.sim import TIMEOUT
    from starship.train.catch_pretrain import make_runner
    from starship.train.learning_audit import audit

    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose(
            "train",
            overrides=[
                "device=cpu",
                "num_envs=8",
                "eval_envs=8",
                "curriculum=none",
                "sim.spawn.altitude=100",
                "sim.spawn.pitch=0",
                "sim.spawn.vertical_speed=-5",
                "sim.max_time=0.4",
            ],
        )
    cfg.curriculum.stages = [{"name": "audit timeout", "overrides": []}]
    (tmp_path / ".hydra").mkdir()
    OmegaConf.save(cfg, tmp_path / ".hydra/config.yaml")
    runner = make_runner(cfg, tmp_path / "setup", 100)
    checkpoint = tmp_path / "model.pt"
    runner.save(checkpoint)
    runner.writer.close()
    arrays = []
    for label in ["first", "repeat"]:
        folder = tmp_path / label
        report = audit(
            checkpoint,
            8,
            200,
            folder,
            stochastic=stochastic,
            action_seed=300 if stochastic else None,
            **({"device": device} if device != "cpu" else {}),
        )
        assert report["audit_device"] == device
        assert report["episodes"] == 8
        with np.load(folder / "trajectories.npz") as data:
            arrays.append({key: data[key].copy() for key in ["obs", "valid", "status", "returns"]})
        assert np.all(arrays[-1]["status"] == TIMEOUT)
        assert np.isfinite(arrays[-1]["returns"]).all()
    for key in arrays[0]:
        np.testing.assert_array_equal(arrays[0][key], arrays[1][key])
