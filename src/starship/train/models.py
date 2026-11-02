"""Residual policy models that preserve learned behavior at fine-tuning initialization."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models.rnn_model import RNNModel
from rsl_rl.modules import MLP, HiddenState
from tensordict import TensorDict
from torch import nn

from starship.train.actions import CONTINUOUS_HEADS, ENGINE_HEADS


class ContinuousAdapterRNNModel(RNNModel):
    """RNN actor with zero-initialized nonlinear residuals over a frozen controller.

    The inherited MLP remains checkpoint-compatible and continues to produce every engine logit.
    The continuous adapter adds richer state-dependent corrections to throttle, gimbal, flaps, and
    RCS. An optional engine adapter can adjust the three engine logits when a later curriculum stage
    requires a different engine count; both see the frozen recurrent latent.
    """

    def __init__(
        self,
        *args,
        continuous_adapter_hidden_dim: int = 64,
        engine_adapter_hidden_dim: int = 0,
        **kwargs,
    ) -> None:
        if continuous_adapter_hidden_dim <= 0:
            raise ValueError("continuous_adapter_hidden_dim must be positive")
        if engine_adapter_hidden_dim < 0:
            raise ValueError("engine_adapter_hidden_dim must be nonnegative")
        super().__init__(*args, **kwargs)
        self.continuous_adapter = MLP(
            self.latent_dim,
            CONTINUOUS_HEADS,
            [continuous_adapter_hidden_dim],
            kwargs.get("activation", "elu"),
        )
        output = next(
            module
            for module in reversed(list(self.continuous_adapter.modules()))
            if isinstance(module, nn.Linear)
        )
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        self.engine_adapter = None
        if engine_adapter_hidden_dim:
            self.engine_adapter = MLP(
                self.latent_dim,
                ENGINE_HEADS,
                [engine_adapter_hidden_dim],
                kwargs.get("activation", "elu"),
            )
            engine_output = next(
                module
                for module in reversed(list(self.engine_adapter.modules()))
                if isinstance(module, nn.Linear)
            )
            nn.init.zeros_(engine_output.weight)
            nn.init.zeros_(engine_output.bias)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        base = self.mlp(latent)
        engine = base[..., :ENGINE_HEADS]
        if self.engine_adapter is not None:
            engine = engine + self.engine_adapter(latent)
        continuous = base[..., 3:] + self.continuous_adapter(latent)
        output = torch.cat((engine, continuous), dim=-1)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def as_jit(self) -> nn.Module:
        if isinstance(self.rnn.rnn, nn.LSTM):
            return _TorchLSTMContinuousAdapter(self)
        raise NotImplementedError("ContinuousAdapterRNNModel JIT export currently requires LSTM")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        raise NotImplementedError("ContinuousAdapterRNNModel ONNX export is not implemented")

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        super().reset(dones, hidden_state)
        reset_distribution = getattr(self.distribution, "reset", None)
        if reset_distribution is not None:
            reset_distribution(dones)


class _TorchLSTMContinuousAdapter(nn.Module):
    """TorchScript-friendly deterministic LSTM policy with the continuous adapter."""

    def __init__(self, model: ContinuousAdapterRNNModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        self.continuous_adapter = copy.deepcopy(model.continuous_adapter)
        self.engine_adapter = (
            copy.deepcopy(model.engine_adapter)
            if model.engine_adapter is not None
            else _ZeroEngineAdapter()
        )
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.register_buffer(
            "hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
        )
        self.register_buffer(
            "cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        latent, (hidden, cell) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = hidden
        self.cell_state[:] = cell
        latent = latent.squeeze(0)
        base = self.mlp(latent)
        engine = base[..., :3] + self.engine_adapter(latent)
        continuous = base[..., 3:] + self.continuous_adapter(latent)
        output = torch.cat((engine, continuous), dim=-1)
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0


class _ZeroEngineAdapter(nn.Module):
    """Keep the exported wrapper structurally uniform when engine adaptation is disabled."""

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(latent[..., :3])
