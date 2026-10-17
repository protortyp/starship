"""24-to-28 input migration, preserving the learned part of policies and Adam state."""

import torch
from torch import nn

from starship.train.encoding import LEGACY_OBS_DIM, OBS_DIM

_INPUT_WEIGHTS = {"rnn.rnn.weight_ih_l0", "mlp.0.weight"}
_MOMENTS = {"obs_normalizer._mean": 0, "obs_normalizer._var": 1, "obs_normalizer._std": 1}


def _pad(value: torch.Tensor, fill: float = 0) -> torch.Tensor:
    result = value.new_full((*value.shape[:-1], OBS_DIM), fill)
    result[..., :LEGACY_OBS_DIM] = value
    return result


def migrate_checkpoint(checkpoint: dict, actor: nn.Module, critic: nn.Module, optimizer) -> None:
    changed = set()
    for name, model in (("actor", actor), ("critic", critic)):
        state = checkpoint[f"{name}_state_dict"]
        target = model.state_dict()
        parameters = dict(model.named_parameters())
        for key in _INPUT_WEIGHTS | _MOMENTS.keys():
            if key not in state or key not in target:
                continue
            old, new = state[key], target[key]
            if (
                old.shape != new.shape
                and old.ndim == new.ndim
                and old.shape[:-1] == new.shape[:-1]
                and old.shape[-1] == LEGACY_OBS_DIM
                and new.shape[-1] == OBS_DIM
            ):
                state[key] = _pad(old, _MOMENTS.get(key, 0))
                if key in parameters:
                    changed.add(id(parameters[key]))
    if not changed or "optimizer_state_dict" not in checkpoint:
        return
    saved = checkpoint["optimizer_state_dict"]
    for old_group, group in zip(saved["param_groups"], optimizer.param_groups, strict=True):
        for index, parameter in zip(old_group["params"], group["params"], strict=True):
            if id(parameter) not in changed:
                continue
            for key, value in saved["state"].get(index, {}).items():
                if (
                    isinstance(value, torch.Tensor)
                    and value.ndim == 2
                    and value.shape[-1] == LEGACY_OBS_DIM
                ):
                    saved["state"][index][key] = _pad(value)


def load_policy(path, device="cpu") -> nn.Module:
    """Load a standalone export for current observations, including pre-entry LSTM actors."""
    policy = torch.load(path, map_location=device, weights_only=False).eval()
    if getattr(policy, "obs_dim", None) != LEGACY_OBS_DIM:
        return policy
    if getattr(policy, "is_recurrent", False):
        layer = policy.rnn.rnn
        layer.weight_ih_l0 = nn.Parameter(_pad(layer.weight_ih_l0.detach()))
        layer.input_size = OBS_DIM
        layer.flatten_parameters()
    else:
        layer = policy.mlp[0]
        layer.weight = nn.Parameter(_pad(layer.weight.detach()))
        layer.in_features = OBS_DIM
    for key, fill in _MOMENTS.items():
        name = key.split(".")[-1]
        if hasattr(policy.obs_normalizer, name):
            setattr(policy.obs_normalizer, name, _pad(getattr(policy.obs_normalizer, name), fill))
    policy.obs_dim = OBS_DIM
    policy.reset()
    return policy
