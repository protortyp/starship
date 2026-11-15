"""Observation and arm-action migration, preserving learned weights and Adam moments."""

import torch
from torch import nn

from starship.train.actions import ACTION_DIM, LEGACY_ACTION_DIM
from starship.train.encoding import LEGACY_OBS_DIM, OBS_DIM

_INPUT_WEIGHTS = {"rnn.rnn.weight_ih_l0", "mlp.0.weight"}
_MOMENTS = {"obs_normalizer._mean": 0, "obs_normalizer._var": 1, "obs_normalizer._std": 1}


def _pad(value: torch.Tensor, fill: float = 0) -> torch.Tensor:
    result = value.new_full((*value.shape[:-1], OBS_DIM), fill)
    result[..., : value.shape[-1]] = value
    return result


def _pad_actions(value: torch.Tensor) -> torch.Tensor:
    result = value.new_zeros((ACTION_DIM, *value.shape[1:]))
    result[:LEGACY_ACTION_DIM] = value
    return result


def _output_layer(model):
    return [
        (name, layer) for name, layer in model.mlp.named_modules() if isinstance(layer, nn.Linear)
    ][-1]


def migrate_checkpoint(checkpoint: dict, actor: nn.Module, critic: nn.Module, optimizer) -> None:
    changed = {}
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
                and old.shape[-1] in (LEGACY_OBS_DIM, 28, 31)
                and new.shape[-1] == OBS_DIM
            ):
                state[key] = _pad(old, _MOMENTS.get(key, 0))
                if key in parameters:
                    changed.setdefault(id(parameters[key]), []).append(_pad)
        if name == "actor":
            layer_name, _ = _output_layer(model)
            for suffix in ("weight", "bias"):
                key = f"mlp.{layer_name}.{suffix}"
                old, new = state[key], target[key]
                if old.shape[0] == LEGACY_ACTION_DIM and new.shape[0] == ACTION_DIM:
                    state[key] = _pad_actions(old)
                    changed.setdefault(id(parameters[key]), []).append(_pad_actions)
    if not changed or "optimizer_state_dict" not in checkpoint:
        return
    saved = checkpoint["optimizer_state_dict"]
    for old_group, group in zip(saved["param_groups"], optimizer.param_groups, strict=True):
        for index, parameter in zip(old_group["params"], group["params"], strict=True):
            if id(parameter) not in changed:
                continue
            for key, value in saved["state"].get(index, {}).items():
                if isinstance(value, torch.Tensor) and value.ndim > 0:
                    for pad in changed[id(parameter)]:
                        value = pad(value)
                    saved["state"][index][key] = value


def load_policy(path, device="cpu") -> nn.Module:
    """Load a standalone export for current observations, including pre-entry LSTM actors."""
    policy = torch.load(path, map_location=device, weights_only=False).eval()
    if getattr(policy, "obs_dim", None) in (LEGACY_OBS_DIM, 28, 31):
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
                setattr(
                    policy.obs_normalizer, name, _pad(getattr(policy.obs_normalizer, name), fill)
                )
        policy.obs_dim = OBS_DIM
    _, layer = _output_layer(policy)
    if layer.out_features == LEGACY_ACTION_DIM:
        layer.weight = nn.Parameter(_pad_actions(layer.weight.detach()))
        layer.bias = nn.Parameter(_pad_actions(layer.bias.detach()))
        layer.out_features = ACTION_DIM
        policy.distribution.output_dim = ACTION_DIM
        policy.distribution._distribution = None
    policy.reset()
    return policy
