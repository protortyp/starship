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


@pytest.mark.parametrize("entropy", [-100.0, 0.0, 100.0])
def test_diagnose_only_warns_on_collapsed_engine_entropy(tmp_path, entropy):
    from starship.train.diagnose import run_summary

    scalars = {"explore/entropy": [(0, entropy)]}
    text = "\n".join(run_summary(tmp_path, scalars, {}))
    assert "3 Bernoulli + 8 Gaussian heads; no fixed max" in text
    assert "WARN" not in text
    scalars["explore/entropy_head/engine0"] = [(0, 0.69)]
    scalars["explore/entropy_head/engine1"] = [(0, 0.01)]
    text = "\n".join(run_summary(tmp_path, scalars, {}))
    assert "WARN engine1 entropy" in text
    assert "WARN engine0 entropy" not in text


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
    assert {"hidden0/dormant", "hidden1/dormant"} <= d.keys()
    assert 1 <= d["features/effective_rank"] <= d["features/width"]
    assert 0.0 <= d["lstm/forget_saturated"] <= 1.0


def test_diagnose_report_on_a_short_run(tmp_path):
    from omegaconf import OmegaConf

    from starship.train.diagnose import (
        find_run,
        load_scalars,
        plasticity_section,
        probe_section,
        run_summary,
    )

    cfg = compose_cfg("runner.algorithm.diagnostics_interval=1")
    runner = make_runner(cfg, tmp_path)
    runner.learn(2)
    runner.writer.flush()
    (tmp_path / ".hydra").mkdir()
    OmegaConf.save(cfg, tmp_path / ".hydra" / "config.yaml")

    run = find_run(str(tmp_path))
    scalars = load_scalars(run)
    report: dict = {}
    lines = run_summary(run, scalars, report) + plasticity_section(scalars, report)
    lines += probe_section(run, report, episodes=16, device="cpu")
    text = "\n".join(lines)
    assert "## Plasticity" in text
    assert "## Probe" in text
    assert report["probe"]["episodes"] == 16
    assert "actor/hidden1/dormant" in " ".join(report["plasticity"])
