"""Per-episode environment metrics, accumulated on the env's device."""

import math

import torch

from starship.sim import CRASHED, LANDED, OUT_OF_BOUNDS, TIMEOUT
from starship.train.reward import REWARD_TERMS
from starship.train.rsl_env import HEAD_NAMES, THRUSTER_BITS, RslRocketEnv

OUTCOMES = {
    "landed": LANDED,
    "crashed": CRASHED,
    "out_of_bounds": OUT_OF_BOUNDS,
    "timeout": TIMEOUT,
}


class EpisodeTracker:
    """Tracks running sums per env and collects stats of every episode that finishes."""

    def __init__(self, env: RslRocketEnv):
        self.env = env
        n, device = env.num_envs, env.device
        self.policy_dt = env.env.config.policy_dt
        self._return = torch.zeros(n, device=device)
        self._terminal = torch.zeros(n, device=device)
        self._terms = torch.zeros(n, len(REWARD_TERMS), device=device)
        self._thruster_steps = torch.zeros(n, len(THRUSTER_BITS), device=device)
        self._bits = torch.tensor(THRUSTER_BITS, dtype=torch.int32, device=device)
        self._finished: list[dict[str, torch.Tensor]] = []
        self.total_episodes = 0

    def step(self, dones: torch.Tensor, record: torch.Tensor | None = None) -> None:
        """Account for the latest env step; keep finished episodes where `record` is set."""
        env = self.env
        self._return += env.reward
        self._terminal += env.terminal_reward
        self._terms += env.reward_terms
        self._thruster_steps += ((env.actions[:, None] & self._bits) != 0).float()
        done = dones.bool()
        keep = done if record is None else done & record
        if keep.any():
            steps = env.final_steps[keep].float()
            vel = env.final_vel[keep]
            self._finished.append(
                {
                    "return": self._return[keep],
                    "terminal": self._terminal[keep],
                    "terms": self._terms[keep],
                    "length_s": steps * self.policy_dt,
                    "status": env.final_status[keep],
                    "speed": vel.norm(dim=-1),
                    "vx": vel[:, 0],
                    "vy": vel[:, 1],
                    "tilt_deg": env.final_theta[keep].abs() * (180.0 / math.pi),
                    "spin_dps": env.final_omega[keep].abs() * (180.0 / math.pi),
                    "fuel_t": env.final_fuel[keep] / 1000.0,
                    "duty": self._thruster_steps[keep] / steps[:, None],
                }
            )
            self.total_episodes += int(keep.sum())
        if done.any():
            self._return[done] = 0.0
            self._terminal[done] = 0.0
            self._terms[done] = 0.0
            self._thruster_steps[done] = 0.0

    @property
    def num_finished(self) -> int:
        return sum(len(e["status"]) for e in self._finished)

    def summarize(self, prefix: str) -> dict[str, float]:
        """Means over the episodes finished since the last call; empty if none finished."""
        if not self._finished:
            return {}
        ep = {k: torch.cat([e[k] for e in self._finished]) for k in self._finished[0]}
        self._finished.clear()
        out = {
            "episodes": float(len(ep["status"])),
            "return": ep["return"].mean().item(),
            "return_shaping": (ep["return"] - ep["terminal"]).mean().item(),
            "return_terminal": ep["terminal"].mean().item(),
            "length_s": ep["length_s"].mean().item(),
        }
        # Episode sums of each reward term; they add up to `return`.
        for i, term in enumerate(REWARD_TERMS):
            out[f"reward/{term}"] = ep["terms"][:, i].mean().item()
        out["reward/terminal"] = ep["terminal"].mean().item()
        for name, code in OUTCOMES.items():
            out[f"rate/{name}"] = (ep["status"] == code).float().mean().item()
        for i, head in enumerate(HEAD_NAMES):
            out[f"duty/{head}"] = ep["duty"][:, i].mean().item()
        landed, crashed = ep["status"] == LANDED, ep["status"] == CRASHED
        if landed.any():
            for key in ("speed", "vx", "vy", "tilt_deg", "spin_dps", "fuel_t"):
                out[f"landed/{key}"] = ep[key][landed].mean().item()
        if crashed.any():
            out["crashed/impact_speed"] = ep["speed"][crashed].mean().item()
            out["crashed/fuel_t"] = ep["fuel_t"][crashed].mean().item()
        return {f"{prefix}/{k}": v for k, v in out.items()}
