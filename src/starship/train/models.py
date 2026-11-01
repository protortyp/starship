"""Policy models that preserve a learned discrete engine schedule during fine-tuning."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models.rnn_model import RNNModel
from rsl_rl.modules import MLP, HiddenState
from tensordict import TensorDict
from torch import nn

from starship.train.actions import CONTINUOUS_HEADS


class ContinuousAdapterRNNModel(RNNModel):
    """RNN actor with a zero-initialized nonlinear residual on continuous outputs only.

    The inherited MLP remains checkpoint-compatible and continues to produce every engine logit.
    The adapter sees the frozen recurrent latent and can add richer state-dependent corrections to
    throttle, gimbal, flaps, and RCS without any path to the discrete engine outputs.
    """

    def __init__(self, *args, continuous_adapter_hidden_dim: int = 64, **kwargs) -> None:
        if continuous_adapter_hidden_dim <= 0:
            raise ValueError("continuous_adapter_hidden_dim must be positive")
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

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        latent = self.get_latent(obs, masks, hidden_state)
        base = self.mlp(latent)
        continuous = base[..., 3:] + self.continuous_adapter(latent)
        output = torch.cat((base[..., :3], continuous), dim=-1)
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


class _TorchLSTMContinuousAdapter(nn.Module):
    """TorchScript-friendly deterministic LSTM policy with the continuous adapter."""

    def __init__(self, model: ContinuousAdapterRNNModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)
        self.mlp = copy.deepcopy(model.mlp)
        self.continuous_adapter = copy.deepcopy(model.continuous_adapter)
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
        continuous = base[..., 3:] + self.continuous_adapter(latent)
        output = torch.cat((base[..., :3], continuous), dim=-1)
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0
