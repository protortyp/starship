from pathlib import Path

import numpy as np
import pytest

from starship.train.learning_audit import audit
from starship.train.return_uncertainty import uncertainty_metrics


def test_noise_free_return_targets_still_expose_critic_bias():
    result = uncertainty_metrics([[0, 2, 4]] * 3, [5, 7, 9])
    assert result["within_start_return_variance"] == 0
    assert result["estimated_single_rollout_ev_ceiling"] == 1
    assert result["critic_vs_mean_return"]["bias"] == 5


def test_variance_components_remove_repeat_noise_from_between_start_variance():
    result = uncertainty_metrics([[1, 5], [-1, 3]], [0, 4])
    assert result["within_start_return_variance"] == 2
    assert result["estimated_between_start_return_variance"] == 7
    assert result["estimated_single_rollout_ev_ceiling"] == pytest.approx(7 / 9)
    assert result["critic_vs_individual_returns"]["explained_variance"] == pytest.approx(0.8)
    assert result["critic_vs_mean_return"]["explained_variance"] == 1


def test_constant_returns_and_pure_noise_do_not_claim_predictable_variation():
    assert (
        uncertainty_metrics([[1, 1], [1, 1]], [1, 1])["estimated_single_rollout_ev_ceiling"] is None
    )
    result = uncertainty_metrics([[1, 1], [-1, -1]], [0, 0])
    assert result["estimated_between_start_return_variance"] == 0
    assert result["estimated_single_rollout_ev_ceiling"] == 0


@pytest.mark.parametrize("returns", [[[1, 2]], [[1], [2]], [[1, np.nan], [1, 2]]])
def test_uncertainty_rejects_incomplete_or_nonfinite_samples(returns):
    with pytest.raises(ValueError, match=r"at least two|must be finite"):
        uncertainty_metrics(returns, [1, 2])


def test_explicit_action_seed_requires_stochastic_policy():
    with pytest.raises(ValueError, match="requires stochastic"):
        audit(Path("missing.pt"), 2, 1, Path("unused"), action_seed=2)
