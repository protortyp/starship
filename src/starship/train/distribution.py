"""Policy head for rsl_rl's pluggable output distributions: independent Bernoulli engine switches
followed by a diagonal Gaussian over the continuous actuator commands."""

import math
from typing import NamedTuple

import torch
from rsl_rl.modules.distribution import Distribution
from torch import nn
from torch.distributions import Bernoulli, Normal
from torch.nn import functional as F

from starship.train.actions import ACTION_DIM, CONTINUOUS_HEADS, ENGINE_HEADS


class _Heads(NamedTuple):
    engine: Bernoulli
    continuous: Normal


class HybridDistribution(Distribution):
    """The MLP outputs ENGINE_HEADS logits followed by CONTINUOUS_HEADS Gaussian means; the
    Gaussian std is a learnable, state-independent parameter per continuous head. Actions are
    (..., ACTION_DIM): 0/1 engine flags, then unbounded pre-tanh samples (see actions.py)."""

    def __init__(
        self,
        output_dim: int = ACTION_DIM,
        init_std: float = 0.5,
        std_range: tuple[float, float] = (0.05, 2.0),
    ) -> None:
        if output_dim != ACTION_DIM:
            raise ValueError(f"HybridDistribution needs output_dim={ACTION_DIM}, got {output_dim}")
        super().__init__(output_dim)
        # Pre-tanh units: std 0.5 explores the middle of each actuator range without piling
        # samples onto the tanh limits; the clamp keeps the policy from freezing (std -> 0) or
        # from saturating every actuator (std >> 1), where log_prob gradients become useless.
        self.log_std = nn.Parameter(torch.full((CONTINUOUS_HEADS,), math.log(init_std)))
        self.log_std_range = (math.log(std_range[0]), math.log(std_range[1]))
        # Kept under this name because runner.standalone_copy drops `_distribution` before deep
        # copying the actor; the last heads may hold autograd-graph tensors.
        self._distribution: _Heads | None = None
        Bernoulli.set_default_validate_args(False)
        Normal.set_default_validate_args(False)

    @staticmethod
    def _split(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[..., :ENGINE_HEADS], x[..., ENGINE_HEADS:]

    def update(self, mlp_output: torch.Tensor) -> None:
        logits, mean = self._split(mlp_output)
        std = torch.exp(self.log_std.clamp(*self.log_std_range)).expand_as(mean)
        self._distribution = _Heads(Bernoulli(logits=logits), Normal(mean, std))

    def sample(self) -> torch.Tensor:
        engine, continuous = self._distribution
        return torch.cat([engine.sample(), continuous.sample()], dim=-1)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        logits, mean = self._split(mlp_output)
        return torch.cat([(logits > 0.0).to(mean.dtype), mean], dim=-1)

    def as_deterministic_output_module(self) -> nn.Module:
        return _DeterministicOutput(ENGINE_HEADS)

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        """Engine on-probabilities followed by the Gaussian means, shape (..., ACTION_DIM)."""
        engine, continuous = self._distribution
        return torch.cat([engine.probs, continuous.mean], dim=-1)

    @property
    def std(self) -> torch.Tensor:
        engine, continuous = self._distribution
        p = engine.probs
        return torch.cat([torch.sqrt(p * (1.0 - p)), continuous.stddev], dim=-1)

    @property
    def entropy(self) -> torch.Tensor:
        return self.entropy_per_head.sum(dim=-1)

    @property
    def entropy_per_head(self) -> torch.Tensor:
        """Entropy of each head's decision, shape (..., ACTION_DIM)."""
        engine, continuous = self._distribution
        return torch.cat([engine.entropy(), continuous.entropy()], dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        engine, continuous = self._distribution
        return (engine.logits, continuous.mean, continuous.stddev)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        engine, continuous = self._distribution
        on, cont = self._split(outputs)
        return engine.log_prob(on).sum(dim=-1) + continuous.log_prob(cont).sum(dim=-1)

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_logits, old_mean, old_std = old_params
        new_logits, new_mean, new_std = new_params
        # Bernoulli KL from logits: torch's version goes through probabilities and returns inf
        # once a probability saturates to exactly 0 or 1 in float32.
        p = torch.sigmoid(old_logits)
        kl_engine = p * (F.logsigmoid(old_logits) - F.logsigmoid(new_logits)) + (1.0 - p) * (
            F.logsigmoid(-old_logits) - F.logsigmoid(-new_logits)
        )
        kl_gauss = (
            torch.log(new_std / old_std)
            + (old_std.square() + (old_mean - new_mean).square()) / (2.0 * new_std.square())
            - 0.5
        )
        return kl_engine.sum(dim=-1) + kl_gauss.sum(dim=-1)


class _DeterministicOutput(nn.Module):
    """Exportable module: engine flags from the logits' sign, continuous means passed through."""

    def __init__(self, engine_heads: int) -> None:
        super().__init__()
        self.engine_heads = engine_heads

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        logits = mlp_output[..., : self.engine_heads]
        mean = mlp_output[..., self.engine_heads :]
        return torch.cat([(logits > 0.0).to(mean.dtype), mean], dim=-1)
