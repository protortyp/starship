import copy

import pytest
import torch
from rsl_rl.models import RNNModel
from tensordict import TensorDict

from starship.train.catch_input_init import CHANNELS, INPUT_WEIGHT, prepare_checkpoint
from starship.train.catch_pretrain import sequence_output


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def source():
    obs = TensorDict({"policy": torch.zeros(4, 32)}, batch_size=[4])
    actor, critic = [
        RNNModel(
            obs,
            {"actor": ["policy"], "critic": ["policy"]},
            role,
            n,
            hidden_dims=[8],
            rnn_hidden_dim=8,
            obs_normalization=True,
        ).eval()
        for role, n in [("actor", 12), ("critic", 1)]
    ]
    for model in [actor, critic]:
        for name in ["_mean", "_var", "_std"]:
            getattr(model.obs_normalizer, name)[..., CHANNELS] = 0
    optimizer = torch.optim.Adam(
        [
            {"params": actor.parameters(), "lr": 1e-6},
            {"params": critic.parameters(), "lr": 1e-5},
        ]
    )
    histories = torch.randn(12, 4, 32)
    histories[..., CHANNELS] = 0
    sum(sequence_output(model, histories).square().sum() for model in [actor, critic]).backward()
    optimizer.step()
    optimizer.zero_grad()
    checkpoint = copy.deepcopy(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iter": 42,
            "infos": {"total_env_steps": 1234, "curriculum": {"stage": 1, "passes": 0}},
        }
    )
    return checkpoint, actor, critic, optimizer, histories


def test_initialization_preserves_ground_outputs_optimizer_and_all_unrelated_state():
    checkpoint, actor, critic, optimizer, histories = source()
    original = copy.deepcopy(checkpoint)
    prepared = prepare_checkpoint(checkpoint, actor, critic, optimizer)
    assert equal_tree(checkpoint, original)
    assert equal_tree(prepared["optimizer_state_dict"], original["optimizer_state_dict"])
    assert prepared["infos"] == original["infos"]
    assert prepared["iter"] == 42
    for role, model in [("actor", actor), ("critic", critic)]:
        assert equal_tree(model.state_dict(), original[f"{role}_state_dict"])
        new_model = copy.deepcopy(model)
        new_model.load_state_dict(prepared[f"{role}_state_dict"])
        with torch.no_grad():
            assert torch.equal(
                sequence_output(model, histories), sequence_output(new_model, histories)
            )
        for key, value in original[f"{role}_state_dict"].items():
            new = prepared[f"{role}_state_dict"][key]
            if key in [INPUT_WEIGHT, "obs_normalizer._var", "obs_normalizer._std"]:
                assert torch.equal(value[..., :28], new[..., :28])
            else:
                assert torch.equal(value, new)
        assert not new_model.rnn.rnn.weight_ih_l0[:, CHANNELS].any()
        assert (new_model.obs_normalizer._std[..., CHANNELS] == 1).all()
        # Newly active nonzero features can learn despite zero input columns.
        catch_inputs = histories.clone()
        catch_inputs[..., CHANNELS] = torch.randn_like(catch_inputs[..., CHANNELS])
        sequence_output(new_model, catch_inputs).square().sum().backward()
        grad = new_model.rnn.rnn.weight_ih_l0.grad[:, CHANNELS]
        assert torch.isfinite(grad).all()
        assert (grad.abs().sum(0) > 0).all()


@pytest.mark.parametrize("problem", ["used_stats", "used_adam", "wrong_model", "missing_group"])
def test_rejects_unsafe_or_mismatched_source_without_mutation(problem):
    checkpoint, actor, critic, optimizer, _ = source()
    if problem == "used_stats":
        actor.obs_normalizer._std[..., CHANNELS] = 1
        checkpoint["actor_state_dict"] = copy.deepcopy(actor.state_dict())
    elif problem == "used_adam":
        parameter = dict(actor.named_parameters())[INPUT_WEIGHT]
        optimizer.state[parameter]["exp_avg"][:, CHANNELS] = 0.01
        checkpoint["optimizer_state_dict"] = copy.deepcopy(optimizer.state_dict())
    elif problem == "wrong_model":
        with torch.no_grad():
            next(actor.parameters()).add_(1)
    else:
        checkpoint["optimizer_state_dict"]["param_groups"].pop()
    original = copy.deepcopy(checkpoint)
    live = copy.deepcopy(actor.state_dict())
    expected = {
        "used_stats": "unused zero-moment",
        "used_adam": "learned optimizer",
        "wrong_model": "match the source",
        "missing_group": "groups must match",
    }[problem]
    with pytest.raises(ValueError, match=expected):
        prepare_checkpoint(checkpoint, actor, critic, optimizer)
    assert equal_tree(checkpoint, original)
    assert equal_tree(actor.state_dict(), live)
