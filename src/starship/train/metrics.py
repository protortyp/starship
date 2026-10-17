"""Per-episode environment metrics, accumulated on the env's device."""

import math

import torch

from starship.sim import CRASHED, LANDED, OUT_OF_BOUNDS, TIMEOUT, TUMBLED
from starship.train.actions import CONTINUOUS, ENGINE_NAMES
from starship.train.reward import REWARD_TERMS

OUTCOMES = {
    "landed": LANDED,
    "crashed": CRASHED,
    "tumbled": TUMBLED,
    "out_of_bounds": OUT_OF_BOUNDS,
    "timeout": TIMEOUT,
}

ATTITUDE_METRICS = {
    "roll_travel_deg": ("roll_travel", 180.0 / math.pi),
    "horizontal_travel_deg": ("horizontal_travel", 180.0 / math.pi),
    "peak_roll_rate_dps": ("peak_roll_rate", 180.0 / math.pi),
    "peak_horizontal_rate_dps": ("peak_horizontal_rate", 180.0 / math.pi),
    "flips": ("flips", 1.0),
    "reflips": ("reflips", 1.0),
}


def body_axis_z(rot: torch.Tensor) -> torch.Tensor:
    """World z component of the body axis b1 for (N, 4) quaternions (x, y, z, w)."""
    x, y, z, w = rot.unbind(-1)
    return 2.0 * (x * z - w * y)


class EpisodeTracker:
    """Tracks running sums per env and collects stats of every episode that finishes."""

    def __init__(self, env):
        self.env = env
        n, device = env.num_envs, env.device
        cfg = env.env.sim.config
        self.policy_dt = cfg.dt * cfg.substeps
        self.landing = cfg.landing
        lows = torch.tensor([lo for _, lo, _ in CONTINUOUS], device=device)
        highs = torch.tensor([hi for _, _, hi in CONTINUOUS], device=device)
        # Within 5 % of a range end counts as saturated.
        self._margin = 0.05 * (highs - lows)
        self._lows, self._highs = lows, highs
        self._return = torch.zeros(n, device=device)
        self._terminal = torch.zeros(n, device=device)
        self._terms = torch.zeros(n, len(REWARD_TERMS), device=device)
        self._engine_steps = torch.zeros(n, len(ENGINE_NAMES), device=device)
        self._action_sum = torch.zeros(n, len(CONTINUOUS), device=device)
        self._saturated = torch.zeros(n, len(CONTINUOUS), device=device)
        self._finished: list[dict[str, torch.Tensor]] = []
        self.total_episodes = 0

    def step(self, dones: torch.Tensor, record: torch.Tensor | None = None) -> None:
        """Account for the latest env step; keep finished episodes where `record` is set."""
        env = self.env
        self._return += env.reward
        self._terminal += env.terminal_reward
        self._terms += env.reward_terms
        self._engine_steps += env.engine_on
        a = env.squashed_actions
        self._action_sum += a
        near_end = (a < self._lows + self._margin) | (a > self._highs - self._margin)
        self._saturated += near_end.float()
        done = dones.bool()
        keep = done if record is None else done & record
        if keep.any():
            steps = env.final_steps[keep].float().clamp(min=1.0)
            vel = env.final_vel[keep]
            pos = env.final_pos[keep]
            self._finished.append(
                {
                    "return": self._return[keep],
                    "terminal": self._terminal[keep],
                    "terms": self._terms[keep],
                    "length_s": steps * self.policy_dt,
                    "status": env.final_status[keep],
                    "vz": -vel[:, 2],
                    "vh": vel[:, :2].norm(dim=-1),
                    "tilt_deg": torch.rad2deg(
                        torch.arccos(body_axis_z(env.final_rot[keep]).clamp(-1.0, 1.0))
                    ),
                    "rate_dps": torch.rad2deg(env.final_omega[keep].norm(dim=-1)),
                    "miss": pos[:, :2].norm(dim=-1),
                    "fuel_t": env.final_tanks[keep].sum(dim=-1) / 1000.0,
                    "engine_duty": self._engine_steps[keep] / steps[:, None],
                    "action_mean": self._action_sum[keep] / steps[:, None],
                    "saturated": self._saturated[keep] / steps[:, None],
                    **{
                        f"attitude/{key}": getattr(env, f"final_{field}")[keep] * scale
                        for key, (field, scale) in ATTITUDE_METRICS.items()
                    },
                    "wind/surface_speed": env.final_wind_speed[keep],
                    "wind/turbulence": env.final_turbulence[keep],
                }
            )
            self.total_episodes += int(keep.sum())
        if done.any():
            for buf in (
                self._return,
                self._terminal,
                self._terms,
                self._engine_steps,
                self._action_sum,
                self._saturated,
            ):
                buf[done] = 0.0

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
        # Episode sums of each reward term; together with the terminal reward they add up to
        # `return`.
        for i, term in enumerate(REWARD_TERMS):
            out[f"reward/{term}"] = ep["terms"][:, i].mean().item()
        out["reward/terminal"] = ep["terminal"].mean().item()
        for name, code in OUTCOMES.items():
            out[f"rate/{name}"] = (ep["status"] == code).float().mean().item()
        for i, name in enumerate(ENGINE_NAMES):
            out[f"actuators/{name}_duty"] = ep["engine_duty"][:, i].mean().item()
        for i, (name, _, _) in enumerate(CONTINUOUS):
            out[f"actuators/{name}_mean"] = ep["action_mean"][:, i].mean().item()
            out[f"actuators/{name}_saturated"] = ep["saturated"][:, i].mean().item()
        landed, crashed = ep["status"] == LANDED, ep["status"] == CRASHED
        for key in ATTITUDE_METRICS:
            values = ep[f"attitude/{key}"]
            out[f"attitude/{key}"] = values.mean().item()
            for subset, mask in (("landed", landed), ("crashed", crashed)):
                if mask.any():
                    out[f"{subset}/attitude/{key}"] = values[mask].mean().item()
        wind_cfg = self.env.env.config.wind
        if wind_cfg.surface_speed > 0 or wind_cfg.turbulence > 0:
            for key in ("wind/surface_speed", "wind/turbulence"):
                out[key] = ep[key].mean().item()
                for subset, mask in (("landed", landed), ("crashed", crashed)):
                    if mask.any():
                        out[f"{subset}/{key}"] = ep[key][mask].mean().item()
            # Landing rate in the calmer and the windier half of the configured range.
            windy = ep["wind/surface_speed"] > 0.5 * wind_cfg.surface_speed
            for name, mask in (("calm", ~windy), ("windy", windy)):
                if mask.any():
                    out[f"rate/landed_{name}"] = landed[mask].float().mean().item()
        touchdown = ("vz", "vh", "tilt_deg", "rate_dps", "miss", "fuel_t")
        if landed.any():
            for key in touchdown:
                out[f"landed/{key}"] = ep[key][landed].mean().item()
        if crashed.any():
            for key in touchdown:
                out[f"crashed/{key}"] = ep[key][crashed].mean().item()
            lim = self.landing
            # Share of crashes breaking each touchdown limit (a crash can break several).
            broken = {
                "vertical_speed": ep["vz"] > lim.max_vertical_speed,
                "horizontal_speed": ep["vh"] > lim.max_horizontal_speed,
                "tilt": ep["tilt_deg"] > math.degrees(lim.max_tilt),
                "rate": ep["rate_dps"] > math.degrees(lim.max_angular_rate),
                "miss": ep["miss"] > lim.pad_radius,
            }
            for name, hit in broken.items():
                out[f"crashed/breaks_{name}"] = hit[crashed].float().mean().item()
        return {f"{prefix}/{k}": v for k, v in out.items()}
