"""Policy head for rsl_rl's pluggable output distributions: independent Bernoulli engine switches
followed by a diagonal Gaussian over the continuous actuator commands."""

import math
from typing import NamedTuple

import torch
from rsl_rl.modules.distribution import Distribution
from torch import nn
from torch.distributions import Bernoulli, Categorical, Normal
from torch.nn import functional as F

from starship.train.actions import ACTION_DIM, CONTINUOUS_HEADS, ENGINE_HEADS


class _Heads(NamedTuple):
    engine: Bernoulli | Categorical
    continuous: Normal
    engine_logits: torch.Tensor
    engine_rank: torch.Tensor | None
    deterministic_engine_count: torch.Tensor | None


class HybridDistribution(Distribution):
    """The MLP outputs ENGINE_HEADS logits followed by CONTINUOUS_HEADS Gaussian means; the
    Gaussian std is a learnable, state-independent parameter per continuous head. Actions are
    (..., ACTION_DIM): 0/1 engine flags, then unbounded pre-tanh samples (see actions.py)."""

    def __init__(
        self,
        output_dim: int = ACTION_DIM,
        init_std: float = 0.5,
        std_range: tuple[float, float] = (0.05, 2.0),
        deterministic_engine_selection: str = "threshold",
        stochastic_engine_selection: str = "independent",
        ranked_count_temperature: float = 0.75,
        ranked_count_hold_steps: int = 1,
    ) -> None:
        if output_dim != ACTION_DIM:
            raise ValueError(f"HybridDistribution needs output_dim={ACTION_DIM}, got {output_dim}")
        if deterministic_engine_selection not in ("threshold", "floor_expected"):
            raise ValueError(
                "deterministic_engine_selection must be 'threshold' or 'floor_expected'"
            )
        if stochastic_engine_selection not in ("independent", "ranked_count"):
            raise ValueError("stochastic_engine_selection must be 'independent' or 'ranked_count'")
        if ranked_count_temperature <= 0:
            raise ValueError("ranked_count_temperature must be positive")
        if ranked_count_hold_steps < 1:
            raise ValueError("ranked_count_hold_steps must be positive")
        super().__init__(output_dim)
        self.deterministic_engine_selection = deterministic_engine_selection
        self.stochastic_engine_selection = stochastic_engine_selection
        self.ranked_count_temperature = ranked_count_temperature
        self.ranked_count_hold_steps = ranked_count_hold_steps
        self._held_engine_count: torch.Tensor | None = None
        self._engine_count_steps_left: torch.Tensor | None = None
        self._force_deterministic_engine_count: torch.Tensor | None = None
        self._engine_credit_mask: torch.Tensor | None = None
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
        rank = None
        deterministic_count = None
        engine_logits = logits
        if self.stochastic_engine_selection == "ranked_count":
            expected = torch.sigmoid(logits).sum(dim=-1, keepdim=True)
            deterministic_count = expected.squeeze(-1).floor().to(torch.long)
            counts = torch.arange(ENGINE_HEADS + 1, device=logits.device, dtype=logits.dtype)
            # Shift the nearest-integer boundaries onto 1, 2, ... so the categorical mode is
            # floor(expected), exactly matching deployment's conservative engine-count decoder.
            center = (expected - 0.5).clamp_min(0.0)
            engine_logits = -((counts - center) / self.ranked_count_temperature).square()
            engine = Categorical(logits=engine_logits)
            rank = logits.argsort(dim=-1, descending=True).argsort(dim=-1)
        else:
            engine = Bernoulli(logits=logits)
        self._distribution = _Heads(
            engine, Normal(mean, std), engine_logits, rank, deterministic_count
        )
        self._force_deterministic_engine_count = None
        self._engine_credit_mask = torch.ones(
            engine_logits.shape[:-1], dtype=mean.dtype, device=mean.device
        )

    def set_force_deterministic_engine_count(self, mask: torch.Tensor) -> None:
        """Use deployment's count for selected rollout environments before a phase decision."""
        self._force_deterministic_engine_count = mask.bool()

    def set_engine_credit_mask(self, mask: torch.Tensor) -> None:
        """Include count log-probability only where a fresh count was actually sampled."""
        # Rollout storage can be allocated inside inference mode. Clone so policy gradients do not
        # try to save an inference tensor for backward when this mask is reused during PPO updates.
        self._engine_credit_mask = (
            mask.clone()
            .to(
                dtype=self._distribution.continuous.mean.dtype,
                device=self._distribution.continuous.mean.device,
            )
            .squeeze(-1)
        )

    def sample(self) -> torch.Tensor:
        heads = self._distribution
        if heads.engine_rank is None:
            on = heads.engine.sample()
        else:
            count = heads.engine.sample()
            force = getattr(self, "_force_deterministic_engine_count", None)
            if force is not None and heads.deterministic_engine_count is not None:
                count = torch.where(force, heads.deterministic_engine_count, count)
            if self.ranked_count_hold_steps > 1 and not torch.is_grad_enabled() and count.ndim == 1:
                if (
                    self._held_engine_count is None
                    or self._held_engine_count.shape != count.shape
                    or self._held_engine_count.device != count.device
                ):
                    self._held_engine_count = count.clone()
                    self._engine_count_steps_left = torch.full_like(
                        count, self.ranked_count_hold_steps
                    )
                refresh = self._engine_count_steps_left <= 0
                self._held_engine_count = torch.where(refresh, count, self._held_engine_count)
                self._engine_count_steps_left = torch.where(
                    refresh,
                    torch.full_like(count, self.ranked_count_hold_steps),
                    self._engine_count_steps_left,
                )
                count = self._held_engine_count
                self._engine_count_steps_left -= 1
            on = (heads.engine_rank < count.unsqueeze(-1)).to(heads.continuous.mean.dtype)
        return torch.cat([on, heads.continuous.sample()], dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Clear held exploratory counts globally or for completed environments."""
        if dones is None or self._engine_count_steps_left is None:
            self._held_engine_count = None
            self._engine_count_steps_left = None
            return
        done = dones.bool()
        if done.shape == self._engine_count_steps_left.shape:
            self._engine_count_steps_left[done] = 0

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        logits, mean = self._split(mlp_output)
        selection = getattr(self, "deterministic_engine_selection", "threshold")
        on = _select_engines(logits, selection).to(mean.dtype)
        return torch.cat([on, mean], dim=-1)

    def as_deterministic_output_module(self) -> nn.Module:
        selection = getattr(self, "deterministic_engine_selection", "threshold")
        return _DeterministicOutput(ENGINE_HEADS, selection)

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        """Engine on-probabilities followed by the Gaussian means, shape (..., ACTION_DIM)."""
        heads = self._distribution
        if heads.engine_rank is None:
            on = heads.engine.probs
        else:
            counts = torch.arange(ENGINE_HEADS + 1, device=heads.engine.probs.device)
            tail = (counts > heads.engine_rank.unsqueeze(-1)).to(heads.engine.probs.dtype)
            on = (tail * heads.engine.probs.unsqueeze(-2)).sum(dim=-1)
        return torch.cat([on, heads.continuous.mean], dim=-1)

    @property
    def std(self) -> torch.Tensor:
        p = self.mean[..., :ENGINE_HEADS]
        return torch.cat([torch.sqrt(p * (1.0 - p)), self._distribution.continuous.stddev], dim=-1)

    @property
    def entropy(self) -> torch.Tensor:
        return self.entropy_per_head.sum(dim=-1)

    @property
    def entropy_per_head(self) -> torch.Tensor:
        """Entropy of each head's decision, shape (..., ACTION_DIM)."""
        heads = self._distribution
        engine = heads.engine.entropy()
        credit = getattr(self, "_engine_credit_mask", None)
        if credit is not None:
            if engine.ndim > credit.ndim:
                credit = credit.unsqueeze(-1)
            engine = engine * credit
        if heads.engine_rank is not None:
            engine = engine.unsqueeze(-1).expand(*engine.shape, ENGINE_HEADS) / ENGINE_HEADS
        return torch.cat([engine, heads.continuous.entropy()], dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        heads = self._distribution
        credit = getattr(self, "_engine_credit_mask", None)
        if credit is None:
            credit = torch.ones(
                heads.engine_logits.shape[:-1],
                dtype=heads.continuous.mean.dtype,
                device=heads.continuous.mean.device,
            )
        return (heads.engine_logits, heads.continuous.mean, heads.continuous.stddev, credit)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        heads = self._distribution
        on, cont = self._split(outputs)
        if heads.engine_rank is None:
            engine_log_prob = heads.engine.log_prob(on).sum(dim=-1)
        else:
            engine_log_prob = heads.engine.log_prob(on.sum(dim=-1).to(torch.long))
        credit = getattr(self, "_engine_credit_mask", None)
        if credit is not None:
            engine_log_prob = engine_log_prob * credit
        return engine_log_prob + heads.continuous.log_prob(cont).sum(dim=-1)

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_logits, old_mean, old_std = old_params[:3]
        new_logits, new_mean, new_std = new_params[:3]
        engine_credit = old_params[3] if len(old_params) > 3 else 1.0
        # Bernoulli KL from logits: torch's version goes through probabilities and returns inf
        # once a probability saturates to exactly 0 or 1 in float32.
        if self.stochastic_engine_selection == "ranked_count":
            old_log_p, new_log_p = old_logits.log_softmax(-1), new_logits.log_softmax(-1)
            kl_engine = (old_log_p.exp() * (old_log_p - new_log_p)).sum(-1)
        else:
            p = torch.sigmoid(old_logits)
            kl_engine = p * (F.logsigmoid(old_logits) - F.logsigmoid(new_logits)) + (1.0 - p) * (
                F.logsigmoid(-old_logits) - F.logsigmoid(-new_logits)
            )
        kl_gauss = (
            torch.log(new_std / old_std)
            + (old_std.square() + (old_mean - new_mean).square()) / (2.0 * new_std.square())
            - 0.5
        )
        if self.stochastic_engine_selection != "ranked_count":
            kl_engine = kl_engine.sum(dim=-1)
        return kl_engine * engine_credit + kl_gauss.sum(dim=-1)


class _DeterministicOutput(nn.Module):
    """Exportable module: engine flags from the logits' sign, continuous means passed through."""

    def __init__(self, engine_heads: int, engine_selection: str = "threshold") -> None:
        super().__init__()
        self.engine_heads = engine_heads
        self.engine_selection = engine_selection

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        logits = mlp_output[..., : self.engine_heads]
        mean = mlp_output[..., self.engine_heads :]
        return torch.cat([_select_engines(logits, self.engine_selection).to(mean.dtype), mean], -1)


def _select_engines(logits: torch.Tensor, selection: str) -> torch.Tensor:
    if selection == "threshold":
        return logits > 0.0
    # Independent Bernoulli heads can hover either side of 0.5 and make threshold decoding switch
    # discontinuously between engine subsets. Preserve their expected count conservatively, while
    # choosing the heads in which the policy has the most confidence.
    count = torch.sigmoid(logits).sum(dim=-1).floor().to(torch.long)
    rank = logits.argsort(dim=-1, descending=True).argsort(dim=-1)
    return rank < count.unsqueeze(-1)
