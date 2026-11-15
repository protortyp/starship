import pytest
import torch
from torch import nn

from starship.train.diagnostics import lstm_gate_stats
from starship.train.recurrent_audit import phase_metrics


def fixture_model(kind):
    torch.manual_seed(17)
    model = nn.Module()
    model.obs_normalizer = nn.Identity()
    model.rnn = nn.Module()
    model.rnn.rnn = nn.LSTM(2, 3)
    model.mlp = nn.Linear(3, 12 if kind == "actor" else 1)
    obs = torch.randn(4, 3, 2)
    valid = torch.arange(4)[:, None] < torch.tensor([4, 2, 1])[None, :]
    with torch.no_grad():
        hidden, _ = model.rnn.rnn(obs)
        output = model.mlp(hidden)
        recorded = output[..., 3:11] if kind == "actor" else output.squeeze(-1)
    return model, obs, valid, recorded


@pytest.mark.parametrize("kind", ["actor", "critic"])
def test_phase_audit_matches_stepwise_gates_and_excludes_autoreset(kind):
    model, obs, valid, recorded = fixture_model(kind)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    # Poison autoreset frames: these must never affect earlier valid gates or verification.
    obs[~valid] = 1e6
    recorded[~valid] = float("nan")
    result = phase_metrics(model, obs, valid, recorded, kind=kind, window_steps=2, batch=2)
    phases = result["phases"]
    assert {k: v["transitions"] for k, v in phases.items()} == {
        "spawn": 3,
        "initial_window": 5,
        "after_initial_window": 2,
        "terminal_window": 5,
        "all": 7,
    }
    totals = {}
    lstm = model.rnn.rnn
    state = (torch.zeros(1, 3, 3), torch.zeros(1, 3, 3))
    with torch.no_grad():
        for t in range(4):
            selected = valid[t]
            stats = lstm_gate_stats(lstm, obs[t, selected], state[0][0, selected])
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value * int(selected.sum())
            _, state = lstm(obs[t : t + 1], state)
    for key, total in totals.items():
        assert phases["all"][key] == pytest.approx(total / 7, abs=1e-7)
    assert result["replay_matches_saved_outputs"]
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize("kind", ["actor", "critic"])
def test_phase_audit_rejects_a_different_policy_or_critic(kind):
    model, obs, valid, recorded = fixture_model(kind)
    recorded[0] += 1
    with pytest.raises(ValueError, match="replay differs"):
        phase_metrics(model, obs, valid, recorded, kind=kind)


def test_phase_audit_rejects_noncontiguous_episode_mask():
    model, obs, valid, recorded = fixture_model("critic")
    valid[2, 1] = False
    valid[3, 1] = True
    with pytest.raises(ValueError, match="contiguous first episodes"):
        phase_metrics(model, obs, valid, recorded, kind="critic")
