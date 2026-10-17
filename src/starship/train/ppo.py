"""rsl_rl PPO with the update loop reimplemented to measure policy health, exploration and
optimizer statistics, plus optional value normalization. Otherwise the optimization matches
rsl_rl's update (without RND, symmetry augmentation and multi-GPU, which are rejected at
construction)."""

import torch
from rsl_rl.algorithms import PPO
from torch import nn

from starship.train.rsl_env import HEAD_NAMES


def _norm(tensors) -> float:
    return torch.linalg.vector_norm(torch.stack([t.detach().norm() for t in tensors])).item()


class ValueNormalizer(nn.Module):
    """Running mean and variance of returns. The critic predicts normalized values, so its targets
    stay at unit scale while the return distribution shifts during training."""

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(()))
        self.register_buffer("var", torch.ones(()))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.detach().float().flatten()
        batch_mean, batch_var, n = x.mean(), x.var(unbiased=False), float(x.numel())
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / total
        self.var = (
            self.var * self.count + batch_var * n + delta**2 * self.count * n / total
        ) / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sqrt(self.var + self.eps) + self.mean


class InstrumentedPPO(PPO):
    def __init__(self, *args, value_normalization: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        if self.rnd or self.symmetry or self.is_multi_gpu:
            raise ValueError("InstrumentedPPO does not support RND, symmetry or multi-GPU")
        # Lives on the critic so it is saved and restored with the critic's state dict.
        self.value_norm = None
        if value_normalization:
            self.critic.value_norm = ValueNormalizer().to(self.device)
            self.value_norm = self.critic.value_norm

    def _values(self, critic_output: torch.Tensor) -> torch.Tensor:
        """Critic output in return units."""
        if self.value_norm is None:
            return critic_output
        return self.value_norm.denormalize(critic_output)

    def act(self, obs):
        actions = super().act(obs)
        self.transition.values = self._values(self.transition.values)
        return actions

    def compute_returns(self, obs) -> None:
        if self.value_norm is None:
            return super().compute_returns(obs)
        # Same as rsl_rl's compute_returns, with the bootstrap value in return units.
        st = self.storage
        hidden = self.critic.get_hidden_state()
        last_values = self._values(self.critic(obs).detach())
        self.critic.reset(hidden_state=hidden)
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = (
                last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            )
            not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        sums: dict[str, float] = {}

        def add(key: str, value: float) -> None:
            sums[key] = sums.get(key, 0.0) + value

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        if self.value_norm is not None:
            self.value_norm.update(self.storage.returns)

        num_updates = 0
        for batch in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (
                        batch.advantages.std() + 1e-8
                    )

            with torch.amp.autocast(
                device_type=torch.device(self.device).type,
                enabled=self.use_mixed_precision,
                dtype=torch.bfloat16,
            ):
                self.actor(
                    batch.observations,
                    masks=batch.masks,
                    hidden_state=batch.hidden_states[0],
                    stochastic_output=True,
                )
                log_prob = self.actor.get_output_log_prob(batch.actions)
                values = self.critic(
                    batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
                )
                dist_params = self.actor.output_distribution_params
                entropy = self.actor.output_entropy

                with torch.inference_mode():
                    kl_mean = self.actor.get_kl_divergence(
                        batch.old_distribution_params, dist_params
                    ).mean()
                if self.desired_kl is not None and self.schedule == "adaptive":
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

                log_ratio = log_prob - torch.squeeze(batch.old_actions_log_prob)
                ratio = torch.exp(log_ratio)
                advantages = torch.squeeze(batch.advantages)
                surrogate = -advantages * ratio
                surrogate_clipped = -advantages * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                policy_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value loss in the critic's own units (normalized when value normalization is on).
                old_values, returns = batch.values, batch.returns
                if self.value_norm is not None:
                    old_values = self.value_norm.normalize(old_values)
                    returns = self.value_norm.normalize(returns)
                if self.use_clipped_value_loss:
                    value_clipped = old_values + (values - old_values).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_loss = torch.max(
                        (values - returns).pow(2), (value_clipped - returns).pow(2)
                    ).mean()
                else:
                    value_loss = (returns - values).pow(2).mean()

                entropy_bonus = entropy.mean()
                loss = (
                    policy_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_bonus
                )

            self.optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_ returns the norm before clipping.
            actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            with torch.no_grad():
                w = ratio.detach().float()
                add("loss/policy", policy_loss.item())
                add("loss/critic", value_loss.item())
                add("loss/entropy_bonus", (-self.entropy_coef * entropy_bonus).item())
                add("loss/total", loss.item())
                add("policy/clip_ratio", ((w - 1.0).abs() > self.clip_param).float().mean().item())
                # k3 estimator of KL(old || new) from the sampled actions.
                add("policy/approx_kl", ((w - 1.0) - log_ratio.detach()).mean().item())
                add("policy/kl_analytic", kl_mean.item())
                # Effective sample size of the importance weights, as a fraction of the batch.
                add("policy/ess", ((w.sum() ** 2) / (w.pow(2).sum() * w.numel())).item())
                add("policy/ratio_mean", w.mean().item())
                add("policy/ratio_max", w.max().item())
                add("explore/entropy", entropy_bonus.item())
                distribution = self.actor.distribution
                if hasattr(distribution, "entropy_per_head"):
                    per_head = distribution.entropy_per_head.reshape(-1, len(HEAD_NAMES))
                    probs = distribution.mean.reshape(-1, len(HEAD_NAMES))
                    for i, head in enumerate(HEAD_NAMES):
                        add(f"explore/entropy_head/{head}", per_head[:, i].mean().item())
                        add(f"explore/fire_prob/{head}", probs[:, i].mean().item())
                add("optim/grad_norm/policy", actor_grad.item())
                add("optim/grad_norm/critic", critic_grad.item())
            num_updates += 1

        metrics = {k: v / num_updates for k, v in sums.items()}
        metrics["explore/entropy_coef"] = self.entropy_coef
        if self.value_norm is not None:
            metrics["value/norm_mean"] = self.value_norm.mean.item()
            metrics["value/norm_std"] = self.value_norm.var.sqrt().item()
        metrics["optim/updates"] = float(num_updates)
        metrics["optim/param_norm/policy"] = _norm(self.actor.parameters())
        metrics["optim/param_norm/critic"] = _norm(self.critic.parameters())
        for i, group in enumerate(self.optimizer.param_groups):
            metrics[f"optim/lr/group_{i}"] = group["lr"]

        st = self.storage
        with torch.no_grad():
            returns, values = st.returns.flatten(), st.values.flatten()
            residual_var = (returns - values).var()
            metrics["value/explained_variance"] = (1.0 - residual_var / returns.var()).item()
            metrics["value/mean"] = values.mean().item()
            metrics["value/return_mean"] = returns.mean().item()
            metrics["value/advantage_std"] = st.advantages.std().item()

        obs = st.observations.flatten(0, 1)
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        st.clear()
        return metrics
