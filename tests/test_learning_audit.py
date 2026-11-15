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
