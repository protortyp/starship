"""Training loop around rsl_rl's PPO with timing, episode metrics, periodic evaluation and policy
snapshots, all logged to TensorBoard."""

import copy
import time
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner
from torch.utils.tensorboard import SummaryWriter

from starship.train.metrics import EpisodeTracker
from starship.train.rsl_env import RslRocketEnv


def standalone_copy(policy: torch.nn.Module) -> torch.nn.Module:
    """Deep copy of an actor without its transient rollout state (recurrent hidden state and last
    distribution), which may hold autograd graph tensors that cannot be copied."""
    rnn = getattr(policy, "rnn", None)
    dist = getattr(policy, "distribution", None)
    hidden = rnn.hidden_state if rnn is not None else None
    last = getattr(dist, "_distribution", None)
    try:
        if rnn is not None:
            rnn.hidden_state = None
        if dist is not None:
            dist._distribution = None
        return copy.deepcopy(policy)
    finally:
        if rnn is not None:
            rnn.hidden_state = hidden
        if dist is not None:
            dist._distribution = last


def export_policy(policy: torch.nn.Module, path: str | Path) -> Path:
    """Save a standalone copy of the actor (normalizer, LSTM, head, distribution) as a module."""
    path = Path(path)
    torch.save(standalone_copy(policy).cpu().eval(), path)
    return path


class StarshipRunner(OnPolicyRunner):
    def __init__(
        self,
        env: RslRocketEnv,
        train_cfg: dict,
        log_dir: str | Path,
        device: str,
        eval_env: RslRocketEnv | None = None,
        eval_interval: int = 25,
        verbose: bool = True,
    ):
        # rsl_rl's own logger stays disabled; this runner writes its own TensorBoard metrics.
        super().__init__(env, train_cfg, log_dir=None, device=device)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.eval_env = eval_env
        self.eval_interval = eval_interval
        self.tracker = EpisodeTracker(env)
        self.total_env_steps = 0
        self.next_iteration = 0
        self.last_metrics: dict[str, float] = {}
        self.verbose = verbose

    def _sync(self) -> None:
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        env, alg, cfg = self.env, self.alg, self.cfg
        steps_per_env = cfg["num_steps_per_env"]
        obs = env.get_observations().to(self.device)
        alg.train_mode()
        start_it = self.next_iteration
        for it in range(start_it, start_it + num_learning_iterations):
            self._sync()
            t0 = time.perf_counter()
            dones_in_batch = 0
            with torch.inference_mode():
                for _ in range(steps_per_env):
                    actions = alg.act(obs)
                    obs, rewards, dones, extras = env.step(actions)
                    alg.process_env_step(obs, rewards, dones, extras)
                    self.tracker.step(dones)
                    dones_in_batch += int(dones.sum())
                self._sync()
                t1 = time.perf_counter()
                alg.compute_returns(obs)
                self._sync()
                t2 = time.perf_counter()
            metrics = alg.update()
            self._sync()
            t3 = time.perf_counter()

            batch = steps_per_env * env.num_envs
            self.total_env_steps += batch
            self.current_learning_iteration = it
            self.next_iteration = it + 1
            metrics.update(self.tracker.summarize("train/env"))
            metrics["train/iteration"] = float(it)
            metrics["train/env_steps"] = float(self.total_env_steps)
            metrics["train/episodes"] = float(self.tracker.total_episodes)
            metrics["train/dones_in_batch"] = float(dones_in_batch)
            metrics["train/batch_size"] = float(batch)

            t_eval = 0.0
            if self.eval_env is not None and (it + 1) % self.eval_interval == 0:
                t_eval = time.perf_counter()
                metrics.update(self.evaluate())
                t_eval = time.perf_counter() - t_eval
            if (it + 1) % cfg["save_interval"] == 0:
                self.save_all(it)

            iteration_s = time.perf_counter() - t0
            metrics["time/collect_s"] = t1 - t0
            metrics["time/returns_s"] = t2 - t1
            metrics["time/update_s"] = t3 - t2
            metrics["time/eval_s"] = t_eval
            metrics["time/iteration_s"] = iteration_s
            metrics["time/collect_fps"] = batch / (t1 - t0)
            metrics["time/train_fps"] = batch / iteration_s

            for key, value in metrics.items():
                self.writer.add_scalar(key, value, self.total_env_steps)
            self.last_metrics = metrics
            if self.verbose:
                self._print(it, start_it + num_learning_iterations, metrics)

        self.save_all(self.current_learning_iteration)
        self.writer.flush()

    def load(self, path: str, load_cfg=None, strict: bool = True, map_location=None) -> dict:
        infos = super().load(path, load_cfg, strict, map_location or self.device)
        # The optimizer state carries the old learning rate; the configured one should win.
        for group in self.alg.optimizer.param_groups:
            group["lr"] = self.alg.learning_rate
        self.next_iteration = self.current_learning_iteration + 1
        # Keeps TensorBoard's env-step axis continuous, assuming the same batch size as before.
        self.total_env_steps = (
            self.next_iteration * self.cfg["num_steps_per_env"] * self.env.num_envs
        )
        return infos

    def evaluate(self) -> dict[str, float]:
        """Run every eval env for one full episode with the deterministic policy."""
        env = self.eval_env
        policy = standalone_copy(self.alg.get_policy()).eval()
        policy.reset()
        tracker = EpisodeTracker(env)
        env.env.reset()
        obs = env.get_observations().to(self.device)
        pending = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        with torch.inference_mode():
            for _ in range(env.max_episode_length):
                actions = policy(obs, stochastic_output=False)
                obs, _, dones, _ = env.step(actions)
                tracker.step(dones, record=pending)
                policy.reset(dones)
                pending &= ~dones.bool()
                if not pending.any():
                    break
        return tracker.summarize("eval/env")

    def save_all(self, it: int) -> None:
        self.save(str(self.log_dir / f"model_{it}.pt"))
        export_policy(self.alg.get_policy(), self.log_dir / f"policy_{it}.pt")
        export_policy(self.alg.get_policy(), self.log_dir / "policy.pt")

    def _print(self, it: int, total: int, m: dict[str, float]) -> None:
        parts = [f"it {it + 1}/{total}", f"steps {m['train/env_steps'] / 1e6:.1f}M"]
        if "train/env/return" in m:
            parts.append(f"return {m['train/env/return']:8.1f}")
            parts.append(f"landed {m['train/env/rate/landed']:.2f}")
            parts.append(f"len {m['train/env/length_s']:5.1f}s")
        if "eval/env/return" in m:
            parts.append(f"eval landed {m['eval/env/rate/landed']:.2f}")
        parts.append(f"fps {m['time/train_fps']:,.0f}")
        print("  ".join(parts), flush=True)
