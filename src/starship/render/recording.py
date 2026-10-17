"""Rerun recordings of evaluation episodes, written during training."""

from pathlib import Path

import numpy as np
import rerun as rr

from starship.render.rerun_viewer import blueprint, flap_geometry, log_scene, log_step
from starship.sim.mass import make_mass_params


class EvalRecorder:
    """Records the first `num_envs` envs of an eval env until each has finished one episode.

    Eval envs auto-reset, so a finished rocket stops being logged instead of showing its respawn.
    """

    def __init__(self, env, path: str | Path, num_envs: int = 16):
        self.sim = env.env.sim
        self.n = min(num_envs, self.sim.num_envs)
        self.cfg = self.sim.config
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.rec = rr.RecordingStream("starship")
        layout = blueprint()
        self.rec.save(path, default_blueprint=layout)
        # The recording stays active from here until close(): re-entering a RecordingStream's
        # context a second time silently drops everything logged afterwards.
        self.rec.__enter__()
        rr.send_blueprint(layout)
        log_scene(self.cfg, self.n)
        self.flaps = flap_geometry(self.cfg)
        self.mass_params = make_mass_params(self.cfg.vehicle)
        self.trails: list[list | None] = [[] for _ in range(self.n)]
        self.t = 0.0
        self._log()

    def _log(self) -> None:
        rr.set_time("sim_time", duration=self.t)
        log_step(self.cfg, self.sim.snapshot(), self.flaps, self.trails, self.mass_params)

    def step(self, dones) -> None:
        """Call after each env step with that step's episode ends."""
        done = np.asarray(dones.cpu())[: self.n].astype(bool)
        for i in np.nonzero(done)[0]:
            self.trails[i] = None
        self.t += self.cfg.dt * self.cfg.substeps
        if any(t is not None for t in self.trails):
            self._log()

    def close(self) -> None:
        self.rec.__exit__(None, None, None)
        self.rec.flush()
