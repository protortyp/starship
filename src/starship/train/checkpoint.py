"""Observation and arm-action migration, preserving learned weights and Adam moments."""

from functools import partial

import torch
from torch import nn

from starship.train.actions import ACTION_DIM, LEGACY_ACTION_DIM
from starship.train.encoding import CATCH_OBS_DIM, LEGACY_OBS_DIM, OBS_DIM

_INPUT_WEIGHTS = {"rnn.rnn.weight_ih_l0", "mlp.0.weight"}
_MOMENTS = {"obs_normalizer._mean": 0, "obs_normalizer._var": 1, "obs_normalizer._std": 1}


def _pad(value: torch.Tensor, fill: float = 0, dim: int = OBS_DIM) -> torch.Tensor:
    result = value.new_full((*value.shape[:-1], dim), fill)
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
                and old.shape[-1] in (LEGACY_OBS_DIM, 28, 31, OBS_DIM)
                and new.shape[-1] in (OBS_DIM, CATCH_OBS_DIM)
                and old.shape[-1] < new.shape[-1]
            ):
                pad = partial(_pad, dim=new.shape[-1])
                state[key] = pad(old, _MOMENTS.get(key, 0))
                if key in parameters:
                    changed.setdefault(id(parameters[key]), []).append(pad)
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


def load_policy(path, device="cpu", obs_dim: int | None = None) -> nn.Module:
    """Load a standalone export for current observations, including pre-entry LSTM actors."""
    policy = torch.load(path, map_location=device, weights_only=False).eval()
    old_dim = policy.obs_dim
    obs_dim = max(OBS_DIM, old_dim) if obs_dim is None else obs_dim
    if obs_dim not in (OBS_DIM, CATCH_OBS_DIM) or old_dim > obs_dim:
        raise ValueError(f"cannot load {old_dim}-input policy for {obs_dim} observations")
    if old_dim != obs_dim:
        if old_dim not in (LEGACY_OBS_DIM, 28, 31, OBS_DIM):
            raise ValueError(f"unsupported policy observation width {old_dim}")
        pad = partial(_pad, dim=obs_dim)
        if getattr(policy, "is_recurrent", False):
            layer = policy.rnn.rnn
            layer.weight_ih_l0 = nn.Parameter(pad(layer.weight_ih_l0.detach()))
            layer.input_size = obs_dim
            layer.flatten_parameters()
        else:
            layer = policy.mlp[0]
            layer.weight = nn.Parameter(pad(layer.weight.detach()))
            layer.in_features = obs_dim
        for key, fill in _MOMENTS.items():
            name = key.split(".")[-1]
            if hasattr(policy.obs_normalizer, name):
                setattr(
                    policy.obs_normalizer, name, pad(getattr(policy.obs_normalizer, name), fill)
                )
        policy.obs_dim = obs_dim
    _, layer = _output_layer(policy)
    if layer.out_features == LEGACY_ACTION_DIM:
        layer.weight = nn.Parameter(_pad_actions(layer.weight.detach()))
        layer.bias = nn.Parameter(_pad_actions(layer.bias.detach()))
        layer.out_features = ACTION_DIM
        policy.distribution.output_dim = ACTION_DIM
        policy.distribution._distribution = None
    policy.reset()
    return policy
