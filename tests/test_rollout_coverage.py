import torch

from starship.train.metrics import RolloutCoverage


def observations(*angles):
    return torch.cos(torch.tensor(angles, dtype=torch.float64))[:, None]


def test_terminal_transition_keeps_old_spawn_bin_across_autoreset():
    coverage = RolloutCoverage(observations(0.2, 1.0), control_dt=0.1)
    coverage.step(torch.tensor([0, 1]), observations(0.2, 0.2), torch.tensor([0, 5]))
    coverage.step(torch.tensor([1, 0]), observations(1.0, 0.2), torch.tensor([1, 0]))
    result = coverage.summarize()
    easy = "train/coverage/tilt_0_to_0.7"
    hard = "train/coverage/tilt_0.9_to_1.1"
    assert result["train/coverage/transitions"] == 4
    assert result[f"{easy}/transitions"] == 3
    assert result[f"{hard}/transitions"] == 1
    assert result[f"{easy}/caught"] == 1
    assert result[f"{hard}/completed"] == 1
    assert result[f"{hard}/caught"] == 0
    assert result[f"{easy}/starts"] == result[f"{hard}/starts"] == 2
    assert coverage.summarize() == {}
    # Summarizing clears counters, not ongoing episode identity or age.
    coverage.step(torch.tensor([0, 0]), observations(0.5, 0.5), torch.tensor([0, 0]))
    result = coverage.summarize()
    assert result[f"{easy}/transitions"] == result[f"{hard}/transitions"] == 1
    assert result[f"{easy}/starts"] == result[f"{hard}/starts"] == 0


def test_unfinished_episode_phase_counts_reconcile_across_updates():
    coverage = RolloutCoverage(observations(0.85), control_dt=1.0)
    for _ in range(3):
        coverage.step(torch.tensor([0]), observations(0.1), torch.tensor([0]))
    first = coverage.summarize()
    prefix = "train/coverage/tilt_0.8_to_0.9"
    assert first[f"{prefix}/first_2s/transitions"] == 2
    assert first[f"{prefix}/2_to_6s/transitions"] == 1
    assert first[f"{prefix}/completed"] == 0
    for _ in range(4):
        coverage.step(torch.tensor([0]), observations(0.1), torch.tensor([0]))
    second = coverage.summarize()
    assert second[f"{prefix}/2_to_6s/transitions"] == 3
    assert second[f"{prefix}/after_6s/transitions"] == 1
    assert first["train/coverage/transitions"] + second["train/coverage/transitions"] == 7
    assert second[f"{prefix}/transition_share"] == 1
