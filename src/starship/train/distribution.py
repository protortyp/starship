"""Independent Bernoulli per thruster, for rsl_rl's pluggable output distributions."""

import torch
from rsl_rl.modules.distribution import Distribution
from torch import nn
from torch.distributions import Bernoulli


class BernoulliDistribution(Distribution):
    """The MLP outputs one logit per thruster; actions are 0/1 floats of shape (N, output_dim)."""

    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim)
        self._distribution: Bernoulli | None = None
        Bernoulli.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        self._distribution = Bernoulli(logits=mlp_output)

    def sample(self) -> torch.Tensor:
        return self._distribution.sample()

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return (mlp_output > 0.0).float()

    def as_deterministic_output_module(self) -> nn.Module:
        return _ThresholdOutput()

    @property
    def input_dim(self) -> int:
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        return self._distribution.probs

    @property
    def std(self) -> torch.Tensor:
        p = self._distribution.probs
        return torch.sqrt(p * (1.0 - p))

    @property
    def entropy(self) -> torch.Tensor:
        return self._distribution.entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self._distribution.logits,)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self._distribution.log_prob(outputs).sum(dim=-1)

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old, new = Bernoulli(logits=old_params[0]), Bernoulli(logits=new_params[0])
        return torch.distributions.kl_divergence(old, new).sum(dim=-1)


class _ThresholdOutput(nn.Module):
    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return (mlp_output > 0.0).float()
