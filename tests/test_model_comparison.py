"""Controls needed to attribute changes to training duration and model capacity."""

import pytest
import torch

from starship.train.actions import ACTION_DIM
from starship.train.model_comparison import checkpoint, make_cohort, restore
from starship.train.world_model import Replay, WorldModel, fit_step


def training_batch(generator):
    state = torch.randn(2, 8, 43, generator=generator) * 0.1
    state[:, :, 6:10] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    actions = torch.randn(2, 8, ACTION_DIM, generator=generator)
    target = state.clone()
    target[:, :, 3] += 0.2 * actions[:, :, 3]
    return {"state": state, "next_state": target, "action": actions}


def test_optimizer_and_sampler_resume_exactly(tmp_path):
    torch.manual_seed(7)
    model = WorldModel(16, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    generator = torch.Generator().manual_seed(19)
    fit_step(model, optimizer, training_batch(generator))
    metadata = {"dataset": "fixed"}
    path = tmp_path / "checkpoint.pt"
    checkpoint(path, model, optimizer, generator, 1, 0.1, metadata)
    next_batch = training_batch(generator)
    expected_loss = fit_step(model, optimizer, next_batch)
    restored = WorldModel(16, 3)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
    restored_generator = torch.Generator().manual_seed(999)
    assert restore(path, restored, restored_optimizer, restored_generator, metadata) == (1, 0.1)
    actual_batch = training_batch(restored_generator)
    for key in next_batch:
        torch.testing.assert_close(actual_batch[key], next_batch[key], rtol=0, atol=0)
    assert fit_step(restored, restored_optimizer, actual_batch) == expected_loss
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match="provenance"):
        restore(path, restored, restored_optimizer, restored_generator, {"dataset": "changed"})


def test_fixed_cohorts_exclude_training_envs_and_reset_boundaries():
    state = torch.zeros(50, 10, 83)
    state[:, :, 2] = 300
    state[:, :, 9] = 1
    done = torch.zeros(50, 10, dtype=torch.bool)
    done[15] = True
    replay = Replay(
        {
            "state": state,
            "next_state": state.clone(),
            "action": torch.zeros(50, 10, ACTION_DIM),
            "done": done,
        }
    )
    rng_before = torch.get_rng_state().clone()
    first = make_cohort(replay, prediction_count=8, per_phase=4)
    second = make_cohort(replay, prediction_count=8, per_phase=4)
    assert torch.equal(rng_before, torch.get_rng_state())
    for key, entry in first.items():
        for name in ("prediction_indices", "probe_indices"):
            indices = entry[name]
            assert (indices[:, 1] >= 8).all()
            for start, env in indices.tolist():
                assert not done[start : start + int(key), env].any()
            torch.testing.assert_close(indices, second[key][name], rtol=0, atol=0)
        assert entry["phases"] == ["low"] * 4
        assert set(entry["missing_phases"]) == {"flip", "belly", "spooling"}


def test_default_architecture_keeps_legacy_checkpoint_keys():
    model = WorldModel()
    assert list(model.net.state_dict()) == [
        "0.weight",
        "0.bias",
        "2.weight",
        "2.bias",
        "4.weight",
        "4.bias",
    ]
    large = WorldModel(32, 3)
    assert len([layer for layer in large.net if isinstance(layer, torch.nn.Linear)]) == 4
    with pytest.raises(ValueError, match="positive"):
        WorldModel(32, 0)
