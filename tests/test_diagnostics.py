import pytest
import torch
from torch import nn

from starship.train import RocketEnv
from starship.train.diagnostics import (
    activation_stats,
    effective_rank,
    layer_norms,
    lstm_gate_stats,
    model_diagnostics,
    snapshot,
    update_ratios,
)
from starship.train.rsl_env import RslRocketEnv
from starship.train.runner import standalone_copy
from tests.test_train import compose_cfg, make_runner


def test_activation_stats_flags_dormant_and_dead_units():
    torch.manual_seed(0)
    acts = torch.randn(512, 4)
    acts[:, 1] = 0.3  # constant: dead
    acts[:, 2] *= 1e-4  # tiny relative to the layer: dormant
    s = activation_stats(acts)
    assert s["dead"] == pytest.approx(0.25)
    assert s["dormant"] == pytest.approx(0.25)


def test_effective_rank_matches_constructed_rank():
    torch.manual_seed(0)
    low = torch.randn(256, 3) @ torch.randn(3, 64)
    assert effective_rank(low) <= 3
    assert effective_rank(torch.randn(256, 64)) > 40


def test_lstm_gate_saturation_detects_saturated_gates():
    lstm = nn.LSTM(4, 8)
    with torch.no_grad():
        lstm.bias_ih_l0.fill_(0.0)
        lstm.bias_hh_l0.fill_(0.0)
        lstm.bias_ih_l0[8:16] = 50.0  # forget gate pinned open
    s = lstm_gate_stats(lstm, torch.randn(32, 4) * 0.01, torch.zeros(32, 8))
    assert s["forget_saturated"] == 1.0
    assert s["input_saturated"] == 0.0


def test_norms_and_update_ratios_per_layer():
    model = nn.Sequential(nn.Linear(3, 4), nn.ELU(), nn.Linear(4, 2))
    before = snapshot(model)
    norms = layer_norms(model)
    assert set(norms) == {"0", "2"}
    with torch.no_grad():
        model[2].weight.mul_(1.1)
        model[2].bias.mul_(1.1)
    ratios = update_ratios(model, before)
    assert ratios["0"] == 0.0
    assert ratios["2"] == pytest.approx(0.1, rel=1e-4)


def test_model_diagnostics_on_the_lstm_policy(tmp_path):
    cfg = compose_cfg()
    runner = make_runner(cfg, tmp_path)
    runner.learn(1)
    env = RslRocketEnv(RocketEnv(64, device="cpu"))
    actor = standalone_copy(runner.alg.get_policy())
    d = model_diagnostics(actor, env.get_observations())
    assert any(k.endswith("/dormant") for k in d)
    assert 1 <= d["features/effective_rank"] <= d["features/width"]
    assert 0.0 <= d["lstm/forget_saturated"] <= 1.0
