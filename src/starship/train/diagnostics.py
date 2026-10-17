"""Network health diagnostics: dormant and dead units, feature rank, LSTM gate saturation, and
per-layer weight, gradient and update norms. Used for TensorBoard logging during training and by
the `starship-diagnose` report.

Definitions:
    dormant unit  ReDo score s_i = E|a_i| / mean_j E|a_j| <= tau (Sokar et al., 2023); the unit
                  barely contributes relative to its layer.
    dead unit     output (nearly) constant over the batch, so no gradient signal varies through it.
    effective rank  smallest k whose top-k singular values hold (1 - delta) of the singular value
                  mass (Kumar et al., 2021); low rank means collapsed features.
    gate saturation  fraction of LSTM sigmoid gates beyond (0.01, 0.99), where their gradient
                  nearly vanishes.
"""

import copy

import torch
from torch import nn

DORMANT_TAU = 0.025
DEAD_STD = 1e-5
RANK_DELTA = 0.01
GATE_EPS = 0.01


def activation_stats(acts: torch.Tensor, tau: float = DORMANT_TAU) -> dict[str, float]:
    """Dormant and dead unit fractions for activations of shape (batch, units)."""
    acts = acts.detach().float().reshape(-1, acts.shape[-1])
    mean_abs = acts.abs().mean(dim=0)
    score = mean_abs / (mean_abs.mean() + 1e-12)
    return {
        "dormant": (score <= tau).float().mean().item(),
        "dead": (acts.std(dim=0) < DEAD_STD).float().mean().item(),
        "mean_abs": mean_abs.mean().item(),
    }


def effective_rank(features: torch.Tensor, delta: float = RANK_DELTA) -> int:
    """srank of a (batch, dim) feature matrix."""
    s = torch.linalg.svdvals(features.detach().float().reshape(-1, features.shape[-1]))
    cumulative = torch.cumsum(s, dim=0) / (s.sum() + 1e-12)
    return int((cumulative < 1.0 - delta).sum().item()) + 1


@torch.no_grad()
def lstm_gate_stats(lstm: nn.LSTM, x: torch.Tensor, h: torch.Tensor) -> dict[str, float]:
    """Saturation of the first layer's gates for inputs x (batch, in) and hidden state h (batch, H).

    PyTorch orders the stacked gates as input, forget, cell, output.
    """
    gates = x @ lstm.weight_ih_l0.T + lstm.bias_ih_l0 + h @ lstm.weight_hh_l0.T + lstm.bias_hh_l0
    i, f, g, o = gates.detach().chunk(4, dim=-1)
    out = {}
    for name, value in (("input", i.sigmoid()), ("forget", f.sigmoid()), ("output", o.sigmoid())):
        out[f"{name}_saturated"] = (
            ((value < GATE_EPS) | (value > 1.0 - GATE_EPS)).float().mean().item()
        )
    out["cell_saturated"] = (g.tanh().abs() > 1.0 - GATE_EPS).float().mean().item()
    out["forget_mean"] = f.sigmoid().mean().item()
    return out


def _layer_name(param_name: str) -> str:
    """Group a parameter name into its layer, e.g. 'mlp.2.weight' -> 'mlp.2'."""
    return param_name.rsplit(".", 1)[0]


def layer_norms(model: nn.Module, grads: bool = False) -> dict[str, float]:
    """L2 norm per layer of the parameters (or of their gradients)."""
    sums: dict[str, float] = {}
    for name, p in model.named_parameters():
        t = p.grad if grads else p
        if t is None:
            continue
        key = _layer_name(name)
        sums[key] = sums.get(key, 0.0) + t.detach().float().pow(2).sum().item()
    return {k: v**0.5 for k, v in sums.items()}


def snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def update_ratios(model: nn.Module, before: dict[str, torch.Tensor]) -> dict[str, float]:
    """||theta_after - theta_before|| / ||theta_before|| per layer."""
    delta: dict[str, float] = {}
    norm: dict[str, float] = {}
    for name, p in model.named_parameters():
        key = _layer_name(name)
        delta[key] = delta.get(key, 0.0) + (p.detach() - before[name]).float().pow(2).sum().item()
        norm[key] = norm.get(key, 0.0) + before[name].float().pow(2).sum().item()
    return {k: (delta[k] / (norm[k] + 1e-12)) ** 0.5 for k in delta}


@torch.no_grad()
def model_diagnostics(model: nn.Module, obs, hidden_state=None) -> dict[str, float]:
    """Activation, rank and gate diagnostics of an rsl_rl MLPModel or RNNModel on a batch of
    observations. Runs on a copy, so the model's own recurrent state is untouched."""
    probe = copy.deepcopy(model).eval()
    captured: list[tuple[str, torch.Tensor]] = []
    hooks = []
    for name, module in probe.mlp.named_children():
        if not isinstance(module, nn.Linear):
            hooks.append(
                module.register_forward_hook(
                    lambda _m, _i, out, name=name: captured.append((f"mlp.{name}", out))
                )
            )
    lstm_input: list[torch.Tensor] = []
    rnn = getattr(probe, "rnn", None)
    if rnn is not None:
        hooks.append(
            probe.obs_normalizer.register_forward_hook(lambda _m, _i, out: lstm_input.append(out))
        )
        rnn.hidden_state = hidden_state
    with torch.no_grad():
        probe(obs)
    for h in hooks:
        h.remove()

    out: dict[str, float] = {}
    for name, acts in captured:
        for k, v in activation_stats(acts).items():
            out[f"{name}/{k}"] = v
    if captured:
        last = captured[-1][1]
        out["features/effective_rank"] = float(effective_rank(last))
        out["features/width"] = float(last.shape[-1])
    if rnn is not None and lstm_input:
        lstm = rnn.rnn
        state = hidden_state[0] if isinstance(hidden_state, tuple) else hidden_state
        h0 = (
            state[0] if state is not None else torch.zeros(lstm_input[0].shape[0], lstm.hidden_size)
        )
        for k, v in lstm_gate_stats(lstm, lstm_input[0], h0.to(lstm_input[0].device)).items():
            out[f"lstm/{k}"] = v
    return out
