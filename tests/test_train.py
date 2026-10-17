import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from starship.train import RocketEnv
from starship.train.actions import ACTION_DIM
from starship.train.metrics import EpisodeTracker
from starship.train.rsl_env import RslRocketEnv
from starship.train.runner import StarshipRunner

# Upright and close to the ground: episodes end within a few seconds.
SHORT_EPISODES = [
    "sim.spawn.altitude=60.0",
    "sim.spawn.vertical_speed=-30.0",
    "sim.spawn.pitch=0.0",
    "sim.max_time=20.0",
]


def compose_cfg(*overrides):
    with initialize_config_module("starship.train.conf", version_base=None):
        return compose(
            "train",
            overrides=["num_envs=64", "runner.num_steps_per_env=8", "curriculum=none", *overrides],
        )


def make_env(cfg, n, seed) -> RslRocketEnv:
    return RslRocketEnv(
        RocketEnv(
            n,
            OmegaConf.to_object(cfg.sim),
            OmegaConf.to_object(cfg.obs),
            OmegaConf.to_object(cfg.reward),
            device="cpu",
            seed=seed,
        )
    )


def make_runner(cfg, log_dir, eval_envs=16) -> StarshipRunner:
    runner_cfg = OmegaConf.to_container(cfg.runner, resolve=True)
    runner_cfg["save_interval"] = 2
    return StarshipRunner(
        make_env(cfg, cfg.num_envs, 0),
        runner_cfg,
        log_dir,
        "cpu",
        eval_env=make_env(cfg, eval_envs, 1),
        eval_interval=2,
    )


def test_observations_are_not_aliased_across_steps():
    env = make_env(compose_cfg(), 4, 0)
    before = env.get_observations()["policy"]
    snapshot = before.clone()
    obs, reward, dones, extras = env.step(torch.zeros(4, ACTION_DIM))
    assert torch.equal(before, snapshot)
    assert not torch.equal(obs["policy"], snapshot)
    assert reward.shape == dones.shape == extras["time_outs"].shape == (4,)


METRIC_GROUPS = [
    "loss/policy",
    "loss/critic",
    "policy/clip_ratio",
    "policy/ess",
    "policy/approx_kl",
    "explore/entropy",
    "explore/entropy_head/engine0",
    "explore/entropy_head/flap_aft_starboard",
    "explore/on_prob/engine0",
    "explore/std/throttle",
    "explore/entropy_coef",
    "optim/lr/group_0",
    "optim/grad_norm/policy",
    "optim/param_norm/critic",
    "value/explained_variance",
    "train/env_steps",
    "train/episodes",
    "train/dones_in_batch",
    "time/collect_s",
    "time/update_s",
    "time/train_fps",
    "train/env/return",
    "train/env/rate/crashed",
    "train/env/rate/tumbled",
    "train/env/actuators/engine0_duty",
    "train/env/actuators/gimbal2_saturated",
    "train/reward/step_mean",
    "plasticity/actor/features/effective_rank",
    "plasticity/actor/lstm/forget_saturated",
    "plasticity/critic/hidden1/dormant",
    "eval/env/return",
    "eval/env/rate/landed",
]


def test_training_logs_metrics_and_exports_usable_policies(tmp_path):
    cfg = compose_cfg(*SHORT_EPISODES, "runner.algorithm.diagnostics_interval=1")
    runner = make_runner(cfg, tmp_path)
    runner.learn(2)
    missing = [k for k in METRIC_GROUPS if k not in runner.last_metrics]
    assert not missing
    assert 0.0 < runner.last_metrics["policy/ess"] <= 1.0
    assert runner.last_metrics["eval/env/episodes"] == 16
    assert (tmp_path / "policy_1.pt").exists()
    assert (tmp_path / "policy_best.pt").exists()
    assert "eval/best_landed" in runner.last_metrics
    assert list(tmp_path.glob("events.out.tfevents.*"))

    policy = torch.load(tmp_path / "policy.pt", weights_only=False).eval()
    policy.reset()
    env = runner.env
    with torch.inference_mode():
        actions = policy(env.get_observations(), stochastic_output=False)
    assert actions.shape == (env.num_envs, ACTION_DIM)
    assert set(actions[:, :3].unique().tolist()) <= {0.0, 1.0}


def test_episode_tracker_summarizes_finished_episodes():
    cfg = compose_cfg(*SHORT_EPISODES)
    env = make_env(cfg, 8, 0)
    tracker = EpisodeTracker(env)
    returns = torch.zeros(8)
    done_once = torch.zeros(8, dtype=torch.bool)
    # Engines off, flaps and gimbal centred: every rocket falls and crashes.
    for _ in range(200):
        _, _, dones, _ = env.step(torch.zeros(8, ACTION_DIM))
        returns += torch.where(done_once, 0.0, env.reward)
        tracker.step(dones, record=~done_once)
        done_once |= dones.bool()
        if done_once.all():
            break
    summary = tracker.summarize("train/env")
    assert summary["train/env/episodes"] == 8
    assert summary["train/env/rate/crashed"] == 1.0
    assert summary["train/env/actuators/engine0_duty"] == 0.0
    assert summary["train/env/crashed/breaks_vertical_speed"] == 1.0
    assert summary["train/env/return"] == pytest.approx(returns.mean().item(), rel=1e-4)
    assert tracker.summarize("train/env") == {}


def test_bench_measures_throughput():
    from starship.train.bench import measure

    cfg = compose_cfg()
    result = measure(cfg, 32, warmup=1, iters=1)
    assert result["num_envs"] == 32
    assert result["batch_size"] == 32 * 8
    assert result["train_fps"] > 0
    assert result["collect_fps"] > result["train_fps"]


def test_training_resumes_from_checkpoint(tmp_path):
    cfg = compose_cfg(*SHORT_EPISODES)
    first = make_runner(cfg, tmp_path / "a")
    first.learn(2)
    cfg.runner.algorithm.learning_rate = 1.234e-4
    cfg.runner.algorithm.schedule = "fixed"
    resumed = make_runner(cfg, tmp_path / "b")
    resumed.load(str(tmp_path / "a" / "model_1.pt"))
    assert resumed.best_eval == first.best_eval
    assert resumed.alg.optimizer.param_groups[0]["lr"] == pytest.approx(1.234e-4)
    for a, b in zip(
        first.alg.get_policy().parameters(), resumed.alg.get_policy().parameters(), strict=True
    ):
        assert torch.equal(a, b)
    resumed.learn(1)
    assert resumed.current_learning_iteration == 2


class CoastingPolicy(torch.nn.Module):
    def reset(self, dones=None):
        pass

    def forward(self, obs, stochastic_output=False):
        return torch.zeros(len(obs), ACTION_DIM)


@pytest.mark.parametrize("deadline", [60.0, 90.0, 180.0, 240.0])
def test_evaluation_counts_every_timeout(tmp_path, monkeypatch, deadline):
    cfg = compose_cfg(
        "num_envs=8",
        "sim.spawn.altitude=1e7",
        "sim.spawn.vertical_speed=0",
        "sim.spawn.pitch=0",
        "sim.max_altitude=2e7",
        f"sim.max_time={deadline}",
    )
    runner = make_runner(cfg, tmp_path, eval_envs=4)
    monkeypatch.setattr(runner.alg, "get_policy", CoastingPolicy)
    metrics = runner.evaluate()
    assert metrics["eval/env/episodes"] == 4
    assert metrics["eval/env/rate/timeout"] == 1
    assert metrics["eval/env/length_s"] == pytest.approx(deadline)


def collect(runner, steps=None):
    alg, env = runner.alg, runner.env
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(steps or runner.cfg["num_steps_per_env"]):
            actions = alg.act(obs)
            obs, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs, rewards, dones, extras)
            runner.tracker.step(dones)
        alg.compute_returns(obs)


@pytest.mark.parametrize(
    ("max_kl", "allowed_updates", "expected_updates"),
    [(None, 1, 4), (0.01, 1, 1), (0.01, 0, 0)],
)
def test_kl_stops_remaining_minibatches(
    tmp_path, monkeypatch, max_kl, allowed_updates, expected_updates
):
    cfg = compose_cfg(
        "num_envs=8",
        "runner.algorithm.num_learning_epochs=2",
        "runner.algorithm.num_mini_batches=2",
        "runner.algorithm.diagnostics_interval=0",
    )
    runner = make_runner(cfg, tmp_path)
    alg = runner.alg
    assert alg.max_kl == 0.01  # the default config enables the KL stop
    assert not alg.use_clipped_value_loss
    alg.max_kl = max_kl
    collect(runner)
    log_prob = alg.actor.get_output_log_prob
    calls = 0

    def shifted_log_prob(actions):
        nonlocal calls
        calls += 1
        return log_prob(actions) - (1.0 if calls > allowed_updates else 0.0)

    monkeypatch.setattr(alg.actor, "get_output_log_prob", shifted_log_prob)
    metrics = alg.update()
    assert metrics["optim/updates"] == expected_updates
    assert metrics["optim/kl_early_stop"] == (max_kl is not None)
    assert metrics["policy/last_approx_kl"] > 0.01
    assert calls == (expected_updates + 1 if max_kl is not None else expected_updates)
    assert alg.storage.step == 0


def test_resume_evaluates_before_learning(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from starship.train import train

    cfg = compose_cfg(*SHORT_EPISODES, "num_envs=8", "eval_envs=4", "device=cpu")
    first = make_runner(cfg, tmp_path / "first")
    checkpoint = tmp_path / "first" / "model.pt"
    first.best_eval = (0, 0.8, 12.0)
    first.save(str(checkpoint))
    cfg.checkpoint = str(checkpoint)
    out = tmp_path / "resumed"
    monkeypatch.setattr(
        train.HydraConfig, "get", lambda: SimpleNamespace(runtime=SimpleNamespace(output_dir=out))
    )
    events = []
    evaluate = StarshipRunner.evaluate

    def evaluated(self, *args, **kwargs):
        events.append("evaluate")
        assert self.best_eval == first.best_eval
        return evaluate(self, *args, **kwargs)

    def learn(self, iterations):
        events.append("learn")
        assert self.best_eval == first.best_eval
        saved = torch.load(out / "model_best.pt", weights_only=False)
        assert saved["infos"]["best_eval"] == first.best_eval
        for a, b in zip(first.alg.actor.parameters(), self.alg.actor.parameters(), strict=True):
            assert torch.equal(a, b)

    monkeypatch.setattr(StarshipRunner, "evaluate", evaluated)
    monkeypatch.setattr(StarshipRunner, "learn", learn)
    train.main.__wrapped__(cfg)
    assert events == ["evaluate", "learn"]
