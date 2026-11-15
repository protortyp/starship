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
