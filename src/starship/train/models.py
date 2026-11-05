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
    requires a different engine count; both see the frozen recurrent latent. Direct-observation
    post-loop and near-ground adapters can specialize recovery and touchdown without changing the
    learned stunt.
    """

    def __init__(
        self,
        *args,
        continuous_adapter_hidden_dim: int = 64,
        engine_adapter_hidden_dim: int = 0,
        post_loop_adapter_hidden_dim: int = 0,
        post_loop_lateral_adapter_hidden_dim: int = 0,
        touchdown_adapter_hidden_dim: int = 0,
        landing_expert: bool = False,
        update_landing_normalization: bool = False,
        post_loop_progress_index: int = 27,
        post_loop_progress_threshold: float = 0.999999,
        touchdown_altitude_index: int = 12,
        touchdown_altitude_threshold: float = 1.0 / 6.0,
        resample_engine_count_post_loop: bool = False,
        resample_engine_count_touchdown: bool = False,
        **kwargs,
    ) -> None:
        if continuous_adapter_hidden_dim <= 0:
            raise ValueError("continuous_adapter_hidden_dim must be positive")
        if engine_adapter_hidden_dim < 0:
            raise ValueError("engine_adapter_hidden_dim must be nonnegative")
        if post_loop_adapter_hidden_dim < 0:
            raise ValueError("post_loop_adapter_hidden_dim must be nonnegative")
        if post_loop_lateral_adapter_hidden_dim < 0:
            raise ValueError("post_loop_lateral_adapter_hidden_dim must be nonnegative")
        if touchdown_adapter_hidden_dim < 0:
            raise ValueError("touchdown_adapter_hidden_dim must be nonnegative")
        super().__init__(*args, **kwargs)
        if not 0 <= post_loop_progress_index < self.obs_dim:
            raise ValueError("post_loop_progress_index must address an actor observation")
        if not 0.0 <= post_loop_progress_threshold <= 1.0:
            raise ValueError("post_loop_progress_threshold must be between zero and one")
        if not 0 <= touchdown_altitude_index < self.obs_dim:
            raise ValueError("touchdown_altitude_index must address an actor observation")
        if not touchdown_altitude_threshold >= 0.0:
            raise ValueError("touchdown_altitude_threshold must be nonnegative")
        self.post_loop_progress_index = post_loop_progress_index
        self.post_loop_progress_threshold = post_loop_progress_threshold
        self.touchdown_altitude_index = touchdown_altitude_index
        self.touchdown_altitude_threshold = touchdown_altitude_threshold
        self.resample_engine_count_post_loop = resample_engine_count_post_loop
        self.resample_engine_count_touchdown = resample_engine_count_touchdown
        self._post_loop_count_sampled: torch.Tensor | None = None
        self._touchdown_count_sampled: torch.Tensor | None = None
        self._deterministic_post_loop_count: torch.Tensor | None = None
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
        self.post_loop_lateral_adapter = None
        if post_loop_lateral_adapter_hidden_dim:
            # Give steering a private feature path so it can gain capacity without changing the
            # protected engine-count logits or throttle through shared hidden activations.
            self.post_loop_lateral_adapter = MLP(
                self.obs_dim,
                ACTION_DIM - ENGINE_HEADS - 1,
                [
                    post_loop_lateral_adapter_hidden_dim,
                    post_loop_lateral_adapter_hidden_dim,
                ],
                kwargs.get("activation", "elu"),
            )
            lateral_output = next(
                module
                for module in reversed(list(self.post_loop_lateral_adapter.modules()))
                if isinstance(module, nn.Linear)
            )
            nn.init.zeros_(lateral_output.weight)
            nn.init.zeros_(lateral_output.bias)
        self.touchdown_adapter = None
        if touchdown_adapter_hidden_dim:
            # Fine touchdown corrections should not perturb either the learned stunt or the
            # high-altitude landing trajectory. This residual sees the same direct Markov state as
            # the post-loop controller, but is additionally gated to the final 500 m by default.
            self.touchdown_adapter = MLP(
                self.obs_dim,
                ACTION_DIM,
                [touchdown_adapter_hidden_dim, touchdown_adapter_hidden_dim],
                kwargs.get("activation", "elu"),
            )
            touchdown_output = next(
                module
                for module in reversed(list(self.touchdown_adapter.modules()))
                if isinstance(module, nn.Linear)
            )
            nn.init.zeros_(touchdown_output.weight)
            nn.init.zeros_(touchdown_output.bias)

        # A residual landing controller still has to cancel every unwanted command produced by
        # the stunt controller.  The phase expert is instead a complete, independently trainable
        # copy of the output controller.  Checkpoint migration initializes each copy from its
        # source module, so enabling it leaves the deployed policy bit-exact at the handoff while
        # allowing PPO to change landing behavior without touching the salto.
        self.landing_mlp = copy.deepcopy(self.mlp) if landing_expert else None
        self.landing_continuous_adapter = (
            copy.deepcopy(self.continuous_adapter) if landing_expert else None
        )
        self.landing_engine_adapter = (
            copy.deepcopy(self.engine_adapter)
            if landing_expert and self.engine_adapter is not None
            else None
        )
        self.landing_post_loop_adapter = (
            copy.deepcopy(self.post_loop_adapter)
            if landing_expert and self.post_loop_adapter is not None
            else None
        )
        self.landing_post_loop_lateral_adapter = (
            copy.deepcopy(self.post_loop_lateral_adapter)
            if landing_expert and self.post_loop_lateral_adapter is not None
            else None
        )
        self.landing_touchdown_adapter = (
            copy.deepcopy(self.touchdown_adapter)
            if landing_expert and self.touchdown_adapter is not None
            else None
        )
        # Landing transfer sees a deliberately wider state distribution. Its private running
        # statistics can adapt without changing the frozen normalization used by the salto path.
        self.landing_obs_normalizer = copy.deepcopy(self.obs_normalizer) if landing_expert else None
        self.update_landing_normalization_enabled = update_landing_normalization

    def update_landing_normalization(self, obs: TensorDict) -> None:
        normalizer = getattr(self, "landing_obs_normalizer", None)
        if (
            normalizer is not None
            and self.obs_normalization
            and self.update_landing_normalization_enabled
        ):
            raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
            normalizer.update(raw)

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
        post_loop_lateral_adapter = getattr(self, "post_loop_lateral_adapter", None)
        if post_loop_lateral_adapter is not None:
            adapter_obs = unpad_trajectories(normalized, masks) if masks is not None else normalized
            progress_index = getattr(self, "post_loop_progress_index", 27)
            progress_threshold = getattr(self, "post_loop_progress_threshold", 0.999999)
            progress = raw[..., progress_index]
            if masks is not None:
                progress = unpad_trajectories(progress.unsqueeze(-1), masks).squeeze(-1)
            gate = (progress >= progress_threshold).to(output.dtype).unsqueeze(-1)
            lateral = output[..., ENGINE_HEADS + 1 :] + gate * post_loop_lateral_adapter(
                adapter_obs
            )
            output = torch.cat((output[..., : ENGINE_HEADS + 1], lateral), dim=-1)
        touchdown_adapter = getattr(self, "touchdown_adapter", None)
        if touchdown_adapter is not None:
            adapter_obs = unpad_trajectories(normalized, masks) if masks is not None else normalized
            progress_index = getattr(self, "post_loop_progress_index", 27)
            progress_threshold = getattr(self, "post_loop_progress_threshold", 0.999999)
            altitude_index = getattr(self, "touchdown_altitude_index", 12)
            altitude_threshold = getattr(self, "touchdown_altitude_threshold", 1.0 / 6.0)
            progress = raw[..., progress_index]
            altitude = raw[..., altitude_index]
            if masks is not None:
                progress = unpad_trajectories(progress.unsqueeze(-1), masks).squeeze(-1)
                altitude = unpad_trajectories(altitude.unsqueeze(-1), masks).squeeze(-1)
            gate = ((progress >= progress_threshold) & (altitude <= altitude_threshold)).to(
                output.dtype
            )
            output = output + gate.unsqueeze(-1) * touchdown_adapter(adapter_obs)
        landing_mlp = getattr(self, "landing_mlp", None)
        if landing_mlp is not None:
            landing_normalized = self.landing_obs_normalizer(raw)
            landing_base = landing_mlp(latent)
            landing_engine = landing_base[..., :ENGINE_HEADS]
            landing_engine_adapter = getattr(self, "landing_engine_adapter", None)
            if landing_engine_adapter is not None:
                landing_engine = landing_engine + landing_engine_adapter(latent)
            landing_continuous = landing_base[..., ENGINE_HEADS:] + (
                self.landing_continuous_adapter(latent)
            )
            landing_output = torch.cat((landing_engine, landing_continuous), dim=-1)
            adapter_obs = (
                unpad_trajectories(landing_normalized, masks)
                if masks is not None
                else landing_normalized
            )
            landing_post_loop_adapter = getattr(self, "landing_post_loop_adapter", None)
            if landing_post_loop_adapter is not None:
                landing_output = landing_output + landing_post_loop_adapter(adapter_obs)
            landing_post_loop_lateral_adapter = getattr(
                self, "landing_post_loop_lateral_adapter", None
            )
            if landing_post_loop_lateral_adapter is not None:
                landing_lateral = landing_output[..., ENGINE_HEADS + 1 :] + (
                    landing_post_loop_lateral_adapter(adapter_obs)
                )
                landing_output = torch.cat(
                    (landing_output[..., : ENGINE_HEADS + 1], landing_lateral), dim=-1
                )
            landing_touchdown_adapter = getattr(self, "landing_touchdown_adapter", None)
            if landing_touchdown_adapter is not None:
                altitude_index = getattr(self, "touchdown_altitude_index", 12)
                altitude_threshold = getattr(self, "touchdown_altitude_threshold", 1.0 / 6.0)
                altitude = raw[..., altitude_index]
                if masks is not None:
                    altitude = unpad_trajectories(altitude.unsqueeze(-1), masks).squeeze(-1)
                touchdown_gate = (altitude <= altitude_threshold).to(output.dtype).unsqueeze(-1)
                landing_output = landing_output + touchdown_gate * landing_touchdown_adapter(
                    adapter_obs
                )
            progress_index = getattr(self, "post_loop_progress_index", 27)
            progress_threshold = getattr(self, "post_loop_progress_threshold", 0.999999)
            progress = raw[..., progress_index]
            if masks is not None:
                progress = unpad_trajectories(progress.unsqueeze(-1), masks).squeeze(-1)
            landing_gate = (progress >= progress_threshold).unsqueeze(-1)
            output = torch.where(landing_gate, landing_output, output)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                self._resample_count_at_post_loop_handoff(raw, masks)
                return self.distribution.sample()
            actions = self.distribution.deterministic_output(output)
            return self._hold_deterministic_count_after_post_loop(actions, output, raw, masks)
        return output

    def _hold_deterministic_count_after_post_loop(
        self,
        actions: torch.Tensor,
        output: torch.Tensor,
        raw: torch.Tensor,
        masks: torch.Tensor | None,
    ) -> torch.Tensor:
        """Keep deployment's handoff count coherent with the persistent training decision."""
        if (
            not getattr(self, "resample_engine_count_post_loop", False)
            or masks is not None
            or raw.ndim != 2
        ):
            return actions
        progress_index = getattr(self, "post_loop_progress_index", 27)
        progress_threshold = getattr(self, "post_loop_progress_threshold", 0.999999)
        complete = raw[:, progress_index] >= progress_threshold
        held = getattr(self, "_deterministic_post_loop_count", None)
        if held is None or held.shape != complete.shape or held.device != complete.device:
            held = torch.full_like(complete, -1, dtype=torch.long)
        handoff = complete & (held < 0)
        touchdown_sampled = getattr(self, "_touchdown_count_sampled", None)
        if (
            touchdown_sampled is None
            or touchdown_sampled.shape != complete.shape
            or touchdown_sampled.device != complete.device
        ):
            touchdown_sampled = torch.zeros_like(complete)
        if getattr(self, "resample_engine_count_touchdown", False):
            altitude_index = getattr(self, "touchdown_altitude_index", 12)
            altitude_threshold = getattr(self, "touchdown_altitude_threshold", 1.0 / 6.0)
            touchdown_ready = complete & (raw[:, altitude_index] <= altitude_threshold)
            handoff = handoff | (touchdown_ready & ~touchdown_sampled)
            touchdown_sampled = touchdown_sampled | touchdown_ready
        current_count = actions[:, :ENGINE_HEADS].sum(dim=-1).to(torch.long)
        held[handoff] = current_count[handoff]
        rank = output[:, :ENGINE_HEADS].argsort(dim=-1, descending=True).argsort(dim=-1)
        held_on = (rank < held.clamp_min(0).unsqueeze(-1)).to(actions.dtype)
        engine = torch.where(complete.unsqueeze(-1), held_on, actions[:, :ENGINE_HEADS])
        self._deterministic_post_loop_count = held
        self._touchdown_count_sampled = touchdown_sampled
        return torch.cat((engine, actions[:, ENGINE_HEADS:]), dim=-1)

    def _resample_count_at_post_loop_handoff(
        self, raw: torch.Tensor, masks: torch.Tensor | None
    ) -> None:
        """Choose one coherent engine count per phase instead of switching during ignition."""
        if (
            not getattr(self, "resample_engine_count_post_loop", False)
            or masks is not None
            or torch.is_grad_enabled()
            or raw.ndim != 2
        ):
            return
        progress_index = getattr(self, "post_loop_progress_index", 27)
        progress_threshold = getattr(self, "post_loop_progress_threshold", 0.999999)
        complete = raw[:, progress_index] >= progress_threshold
        sampled = getattr(self, "_post_loop_count_sampled", None)
        if sampled is None or sampled.shape != complete.shape or sampled.device != complete.device:
            sampled = torch.zeros_like(complete)
        handoff = complete & ~sampled
        touchdown_sampled = getattr(self, "_touchdown_count_sampled", None)
        if (
            touchdown_sampled is None
            or touchdown_sampled.shape != complete.shape
            or touchdown_sampled.device != complete.device
        ):
            touchdown_sampled = torch.zeros_like(complete)
        if getattr(self, "resample_engine_count_touchdown", False):
            altitude_index = getattr(self, "touchdown_altitude_index", 12)
            altitude_threshold = getattr(self, "touchdown_altitude_threshold", 1.0 / 6.0)
            touchdown_ready = complete & (raw[:, altitude_index] <= altitude_threshold)
            handoff = handoff | (touchdown_ready & ~touchdown_sampled)
            touchdown_sampled = touchdown_sampled | touchdown_ready
        set_deterministic = getattr(self.distribution, "set_force_deterministic_engine_count", None)
        if set_deterministic is not None:
            set_deterministic(~complete)
        set_credit = getattr(self.distribution, "set_engine_credit_mask", None)
        if set_credit is not None:
            set_credit(handoff)
        if handoff.any():
            reset_distribution = getattr(self.distribution, "reset", None)
            if reset_distribution is not None:
                reset_distribution(handoff)
        self._post_loop_count_sampled = sampled | complete
        self._touchdown_count_sampled = touchdown_sampled

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
        sampled = getattr(self, "_post_loop_count_sampled", None)
        if dones is None or sampled is None:
            self._post_loop_count_sampled = None
        elif dones.shape == sampled.shape:
            sampled[dones.bool()] = False
        touchdown_sampled = getattr(self, "_touchdown_count_sampled", None)
        if dones is None or touchdown_sampled is None:
            self._touchdown_count_sampled = None
        elif dones.shape == touchdown_sampled.shape:
            touchdown_sampled[dones.bool()] = False
        held = getattr(self, "_deterministic_post_loop_count", None)
        if held is None or (dones is None and hidden_state is None):
            self._deterministic_post_loop_count = None
        elif dones is not None and dones.shape == held.shape:
            held[dones.bool()] = -1


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
        post_loop_lateral_adapter = getattr(model, "post_loop_lateral_adapter", None)
        self.post_loop_lateral_adapter = (
            copy.deepcopy(post_loop_lateral_adapter)
            if post_loop_lateral_adapter
            else _ZeroPostLoopLateralAdapter()
        )
        touchdown_adapter = getattr(model, "touchdown_adapter", None)
        self.touchdown_adapter = (
            copy.deepcopy(touchdown_adapter) if touchdown_adapter else _ZeroPostLoopAdapter()
        )
        landing_mlp = getattr(model, "landing_mlp", None)
        self.landing_expert_enabled = landing_mlp is not None
        landing_obs_normalizer = getattr(model, "landing_obs_normalizer", None)
        self.landing_obs_normalizer = copy.deepcopy(
            landing_obs_normalizer if landing_obs_normalizer is not None else model.obs_normalizer
        )
        self.landing_mlp = copy.deepcopy(landing_mlp if landing_mlp is not None else model.mlp)
        landing_continuous_adapter = getattr(model, "landing_continuous_adapter", None)
        self.landing_continuous_adapter = copy.deepcopy(
            landing_continuous_adapter
            if landing_continuous_adapter is not None
            else model.continuous_adapter
        )
        landing_engine_adapter = getattr(model, "landing_engine_adapter", None)
        self.landing_engine_adapter = (
            copy.deepcopy(landing_engine_adapter)
            if landing_engine_adapter is not None
            else _ZeroEngineAdapter()
        )
        landing_post_loop_adapter = getattr(model, "landing_post_loop_adapter", None)
        self.landing_post_loop_adapter = (
            copy.deepcopy(landing_post_loop_adapter)
            if landing_post_loop_adapter is not None
            else _ZeroPostLoopAdapter()
        )
        landing_post_loop_lateral_adapter = getattr(
            model, "landing_post_loop_lateral_adapter", None
        )
        self.landing_post_loop_lateral_adapter = (
            copy.deepcopy(landing_post_loop_lateral_adapter)
            if landing_post_loop_lateral_adapter is not None
            else _ZeroPostLoopLateralAdapter()
        )
        landing_touchdown_adapter = getattr(model, "landing_touchdown_adapter", None)
        self.landing_touchdown_adapter = (
            copy.deepcopy(landing_touchdown_adapter)
            if landing_touchdown_adapter is not None
            else _ZeroPostLoopAdapter()
        )
        self.post_loop_progress_index = getattr(model, "post_loop_progress_index", 27)
        self.post_loop_progress_threshold = getattr(model, "post_loop_progress_threshold", 0.999999)
        self.touchdown_altitude_index = getattr(model, "touchdown_altitude_index", 12)
        self.touchdown_altitude_threshold = getattr(
            model, "touchdown_altitude_threshold", 1.0 / 6.0
        )
        self.resample_engine_count_post_loop = getattr(
            model, "resample_engine_count_post_loop", False
        )
        self.resample_engine_count_touchdown = getattr(
            model, "resample_engine_count_touchdown", False
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
        self.register_buffer("held_engine_count", torch.full((1,), -1, dtype=torch.long))
        self.register_buffer("touchdown_count_sampled", torch.zeros(1, dtype=torch.bool))

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
        lateral = output[..., 4:] + gate * self.post_loop_lateral_adapter(x)
        output = torch.cat((output[..., :4], lateral), dim=-1)
        altitude = raw[..., self.touchdown_altitude_index]
        touchdown_gate = (progress >= self.post_loop_progress_threshold) & (
            altitude <= self.touchdown_altitude_threshold
        )
        output = output + touchdown_gate.to(output.dtype).unsqueeze(-1) * self.touchdown_adapter(x)
        if self.landing_expert_enabled:
            landing_x = self.landing_obs_normalizer(raw)
            landing_base = self.landing_mlp(latent)
            landing_engine = landing_base[..., :3] + self.landing_engine_adapter(latent)
            landing_continuous = landing_base[..., 3:] + self.landing_continuous_adapter(latent)
            landing_output = torch.cat((landing_engine, landing_continuous), dim=-1)
            landing_output = landing_output + self.landing_post_loop_adapter(landing_x)
            landing_lateral = landing_output[..., 4:] + self.landing_post_loop_lateral_adapter(
                landing_x
            )
            landing_output = torch.cat((landing_output[..., :4], landing_lateral), dim=-1)
            landing_output = landing_output + touchdown_gate.to(landing_output.dtype).unsqueeze(
                -1
            ) * self.landing_touchdown_adapter(landing_x)
            output = torch.where(gate > 0.5, landing_output, output)
        actions = self.deterministic_output(output)
        if self.resample_engine_count_post_loop:
            complete = progress >= self.post_loop_progress_threshold
            handoff = complete & (self.held_engine_count < 0)
            if self.resample_engine_count_touchdown:
                touchdown_ready = complete & (altitude <= self.touchdown_altitude_threshold)
                handoff = handoff | (touchdown_ready & ~self.touchdown_count_sampled)
                self.touchdown_count_sampled[:] = self.touchdown_count_sampled | touchdown_ready
            current_count = actions[..., :3].sum(dim=-1).to(torch.long)
            self.held_engine_count[:] = torch.where(handoff, current_count, self.held_engine_count)
            rank = output[..., :3].argsort(dim=-1, descending=True).argsort(dim=-1)
            held_on = (rank < self.held_engine_count.clamp_min(0).unsqueeze(-1)).to(actions.dtype)
            engine = torch.where(complete.unsqueeze(-1), held_on, actions[..., :3])
            actions = torch.cat((engine, actions[..., 3:]), dim=-1)
        return actions

    @torch.jit.export
    def reset(self) -> None:
        self.hidden_state[:] = 0.0
        self.cell_state[:] = 0.0
        self.held_engine_count[:] = -1
        self.touchdown_count_sampled[:] = False


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


class _ZeroPostLoopLateralAdapter(nn.Module):
    """Keep the exported wrapper uniform when the private steering residual is disabled."""

    def __init__(self) -> None:
        super().__init__()
        self.output_dim = ACTION_DIM - ENGINE_HEADS - 1

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(obs[..., : self.output_dim])
