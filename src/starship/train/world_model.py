"""Residual physical dynamics and episode-safe replay windows (experimental)."""

import torch
from torch import nn

from starship.train.actions import ACTION_DIM
from starship.train.model_state import PHYSICAL_DIM

# Position, velocity, quaternion, angular velocity, engine states, flap states, tanks.
STATE_SCALE = [3000.0] * 3 + [100.0] * 3 + [1.0] * 7 + [1.0] * 26 + [100000.0] * 4


class Replay:
    """Time-major transitions; each environment belongs to exactly one data split."""

    def __init__(self, data: dict[str, torch.Tensor]):
        self.data = data
        shape = data["done"].shape
        if len(shape) != 2 or any(v.shape[:2] != shape for v in data.values()):
            raise ValueError("replay tensors must share [time, environment] dimensions")
        if shape[1] < 2:
            raise ValueError("at least two environments are needed for disjoint splits")

    def windows(self, horizon: int, validation: bool = False) -> torch.Tensor:
        """Return (time, env) starts with no terminal step, including the last step.

        Terminal states are recorded for later termination-head work, but a constant-dt
        physical model is evaluated on uninterrupted flight only.
        """
        done = self.data["done"]
        if not 1 <= horizon <= len(done):
            raise ValueError("horizon must fit inside replay")
        split = max(1, min(done.shape[1] - 1, int(done.shape[1] * 0.8)))
        valid = ~done.unfold(0, horizon, 1).any(-1)
        if validation:
            valid[:, :split] = False
        else:
            valid[:, split:] = False
        indices = valid.nonzero()
        if not len(indices):
            raise ValueError(f"no nonterminal {horizon}-step windows in requested split")
        return indices

    def sample(self, indices: torch.Tensor, count: int, horizon: int) -> dict[str, torch.Tensor]:
        selected = indices[torch.randint(len(indices), (count,), device=indices.device)]
        time = selected[:, 0][None] + torch.arange(horizon, device=indices.device)[:, None]
        env = selected[:, 1][None].expand_as(time)
        return {key: value[time, env] for key, value in self.data.items()}


class WorldModel(nn.Module):
    """Predict 43 physical channels at the control rate, with unit quaternion projection.

    This first model is deliberately restricted to calm physics. Wind/context and
    reward/termination heads must be added before imagined policy training.
    """

    def __init__(self, width: int = 256):
        super().__init__()
        self.register_buffer("scale", torch.tensor(STATE_SCALE))
        self.register_buffer("center", torch.zeros(PHYSICAL_DIM))
        self.register_buffer("delta_scale", torch.ones(PHYSICAL_DIM))
        self.net = nn.Sequential(
            nn.Linear(PHYSICAL_DIM + ACTION_DIM, width),
            nn.ELU(),
            nn.Linear(width, width),
            nn.ELU(),
            nn.Linear(width, PHYSICAL_DIM),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @torch.no_grad()
    def calibrate(self, state: torch.Tensor, next_state: torch.Tensor) -> None:
        self.center.copy_(state.mean(0))
        delta = next_state - state
        self.delta_scale.copy_(delta.std(0, unbiased=False).clamp_min(self.scale * 1e-5))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # The model sees exactly the continuous pre-tanh commands recorded by the env.
        x = torch.cat(((state - self.center) / self.scale, action), dim=-1)
        next_state = state + self.net(x) * self.delta_scale
        quat = next_state[..., 6:10]
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.cat((next_state[..., :6], quat, next_state[..., 10:]), dim=-1)

    def rollout(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        states = []
        for action in actions:
            state = self(state, action)
            states.append(state)
        return torch.stack(states)


def fit_step(model: WorldModel, optimizer, batch: dict[str, torch.Tensor]) -> float:
    states = batch["state"][..., :PHYSICAL_DIM]
    targets = batch["next_state"][..., :PHYSICAL_DIM]
    predicted = model.rollout(states[0], batch["action"])
    loss = ((predicted - targets) / model.delta_scale).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite world-model loss")
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return loss.item()


def physical_errors(predicted: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    difference = predicted - target
    q1 = nn.functional.normalize(predicted[..., 6:10], dim=-1)
    q2 = nn.functional.normalize(target[..., 6:10], dim=-1)
    # q and -q represent the same orientation.
    difference_norm = (q1 - q2).norm(dim=-1)
    sum_norm = (q1 + q2).norm(dim=-1)
    angle = 4 * torch.atan2(
        torch.minimum(difference_norm, sum_norm), torch.maximum(difference_norm, sum_norm)
    )
    return {
        "position_rmse_m": difference[..., :3].square().mean().sqrt().item(),
        "velocity_rmse_m_s": difference[..., 3:6].square().mean().sqrt().item(),
        "attitude_rmse_deg": torch.rad2deg(angle.square().mean().sqrt()).item(),
        "angular_velocity_rmse_rad_s": difference[..., 10:13].square().mean().sqrt().item(),
        "tank_rmse_kg": difference[..., 39:43].square().mean().sqrt().item(),
    }
