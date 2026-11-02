"""Residual policy models that preserve learned behavior at fine-tuning initialization."""

from __future__ import annotations

import copy

import torch
from rsl_rl.models.rnn_model import RNNModel
from rsl_rl.modules import MLP, HiddenState
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict
from torch import nn

from starship.train.actions import ACTION_DIM, CONTINUOUS_HEADS, ENGINE_HEADS


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
        post_loop_adapter_hidden_dim: int = 0,
        post_loop_progress_index: int = 27,
        post_loop_progress_threshold: float = 0.999999,
        **kwargs,
    ) -> None:
        if continuous_adapter_hidden_dim <= 0:
            raise ValueError("continuous_adapter_hidden_dim must be positive")
        if engine_adapter_hidden_dim < 0:
            raise ValueError("engine_adapter_hidden_dim must be nonnegative")
        if post_loop_adapter_hidden_dim < 0:
            raise ValueError("post_loop_adapter_hidden_dim must be nonnegative")
        super().__init__(*args, **kwargs)
        if not 0 <= post_loop_progress_index < self.obs_dim:
            raise ValueError("post_loop_progress_index must address an actor observation")
        if not 0.0 <= post_loop_progress_threshold <= 1.0:
            raise ValueError("post_loop_progress_threshold must be between zero and one")
        self.post_loop_progress_index = post_loop_progress_index
        self.post_loop_progress_threshold = post_loop_progress_threshold
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
        self.post_loop_adapter = None
        if post_loop_adapter_hidden_dim:
            # The frozen recurrent latent was optimized for the stunt and can discard landing
            # details. Give the post-loop controller a direct Markov-state path while gating the
            # complete residual off until the observed stunt progress is complete.
            self.post_loop_adapter = MLP(
                self.obs_dim,
                ACTION_DIM,
                [post_loop_adapter_hidden_dim, post_loop_adapter_hidden_dim],
                kwargs.get("activation", "elu"),
            )
            post_loop_output = next(
                module
                for module in reversed(list(self.post_loop_adapter.modules()))
                if isinstance(module, nn.Linear)
            )
            nn.init.zeros_(post_loop_output.weight)
            nn.init.zeros_(post_loop_output.bias)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
        normalized = self.obs_normalizer(raw)
        latent = self.rnn(normalized, masks, hidden_state).squeeze(0)
        base = self.mlp(latent)
        engine = base[..., :ENGINE_HEADS]
        engine_adapter = getattr(self, "engine_adapter", None)
        if engine_adapter is not None:
            engine = engine + engine_adapter(latent)
        continuous = base[..., 3:] + self.continuous_adapter(latent)
        output = torch.cat((engine, continuous), dim=-1)
        post_loop_adapter = getattr(self, "post_loop_adapter", None)
        if post_loop_adapter is not None:
            adapter_obs = unpad_trajectories(normalized, masks) if masks is not None else normalized
            progress_index = getattr(self, "post_loop_progress_index", 27)
            progress_threshold = getattr(self, "post_loop_progress_threshold", 0.999999)
            progress = raw[..., progress_index]
            if masks is not None:
                progress = unpad_trajectories(progress.unsqueeze(-1), masks).squeeze(-1)
            gate = (progress >= progress_threshold).to(output.dtype).unsqueeze(-1)
            output = output + gate * post_loop_adapter(adapter_obs)
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
        engine_adapter = getattr(model, "engine_adapter", None)
        self.engine_adapter = (
            copy.deepcopy(engine_adapter) if engine_adapter else _ZeroEngineAdapter()
        )
        post_loop_adapter = getattr(model, "post_loop_adapter", None)
        self.post_loop_adapter = (
            copy.deepcopy(post_loop_adapter) if post_loop_adapter else _ZeroPostLoopAdapter()
        )
        self.post_loop_progress_index = getattr(model, "post_loop_progress_index", 27)
        self.post_loop_progress_threshold = getattr(model, "post_loop_progress_threshold", 0.999999)
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
        raw = x
        x = self.obs_normalizer(raw)
        latent, (hidden, cell) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = hidden
        self.cell_state[:] = cell
        latent = latent.squeeze(0)
        base = self.mlp(latent)
        engine = base[..., :3] + self.engine_adapter(latent)
        continuous = base[..., 3:] + self.continuous_adapter(latent)
        output = torch.cat((engine, continuous), dim=-1)
        progress = raw[..., self.post_loop_progress_index]
        gate = (progress >= self.post_loop_progress_threshold).to(output.dtype).unsqueeze(-1)
        output = output + gate * self.post_loop_adapter(x)
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0


class _ZeroEngineAdapter(nn.Module):
    """Keep the exported wrapper structurally uniform when engine adaptation is disabled."""

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(latent[..., :3])


class _ZeroPostLoopAdapter(nn.Module):
    """Keep the exported wrapper uniform when the phase-gated residual is disabled."""

    def __init__(self) -> None:
        super().__init__()
        self.output_dim = ACTION_DIM

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(obs[..., : self.output_dim])
