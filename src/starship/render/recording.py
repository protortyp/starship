"""Rerun recordings of evaluation episodes, written during training."""

from pathlib import Path

import numpy as np
import rerun as rr

from starship.render.rerun_viewer import blueprint, log_static_scene, log_step
from starship.sim import StateSnapshot
from starship.train.rsl_env import RslRocketEnv

FIELDS = ("pos", "vel", "theta", "omega", "status", "steps", "fuel")


class EvalRecorder:
    """Records the first `num_envs` envs of an eval env until each finished one episode.

    Eval envs auto-reset, so finished rockets are frozen at their terminal state instead of
    showing their respawn.
    """

    def __init__(self, env: RslRocketEnv, path: str | Path, num_envs: int = 16):
        self.sim = env.env.sim
        self.env = env
        self.n = min(num_envs, env.num_envs)
        self.cfg = self.sim.config
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rec = rr.RecordingStream("starship")
        self.rec.save(path, default_blueprint=blueprint(self.cfg))
        # Run dirs are logs/<experiment>/<time>/rerun/<file>; the name tells recordings apart
        # when several are open in one viewer.
        run = path.parent.parent
        self.rec.send_recording_name(f"{run.parent.name} {path.stem}")
        self.frozen = np.zeros(self.n, dtype=bool)
        self.final = {f: getattr(self.sim.snapshot(), f)[: self.n].copy() for f in FIELDS}
        self.trail: list = []
        self.t = 0.0
        # The recording stays active from here until close(): re-entering a RecordingStream's
        # context a second time silently drops everything logged afterwards.
        self.rec.__enter__()
        log_static_scene(self.cfg)

    def step(self, dones) -> None:
        """Log the state after an env step; `dones` are that step's episode ends."""
        live = self.sim.snapshot()
        term = self.sim.terminal_snapshot()
        newly = np.asarray(dones.cpu())[: self.n].astype(bool) & ~self.frozen
        for f in FIELDS:
            self.final[f][newly] = getattr(term, f)[: self.n][newly]
        self.frozen |= newly
        view = {}
        for f in FIELDS:
            current = getattr(live, f)[: self.n].copy()
            current[self.frozen] = self.final[f][self.frozen]
            view[f] = current
        actions = self.env.actions[: self.n].cpu().numpy().copy()
        actions[self.frozen] = 0
        self.t += self.cfg.policy_dt
        rr.set_time("sim_time", duration=self.t)
        log_step(self.cfg, StateSnapshot(**view), actions, self.trail)

    @property
    def done(self) -> bool:
        return bool(self.frozen.all())

    def close(self) -> None:
        self.rec.__exit__(None, None, None)
        self.rec.flush()
        self.rec.disconnect()
