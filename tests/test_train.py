import math

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from starship.control.policy import PolicyController
from starship.sim import LEFT, MAIN, RIGHT
from starship.train import RocketEnv
from starship.train.distribution import BernoulliDistribution
from starship.train.metrics import EpisodeTracker
from starship.train.rsl_env import RslRocketEnv, actions_to_bitmask
from starship.train.runner import StarshipRunner


def test_bernoulli_distribution_math():
    dist = BernoulliDistribution(3)
    dist.update(torch.zeros(2, 3))
    assert dist.entropy == pytest.approx(torch.full((2,), 3 * math.log(2)))
    assert dist.log_prob(torch.ones(2, 3)) == pytest.approx(torch.full((2,), 3 * math.log(0.5)))
    params = dist.params
    assert dist.kl_divergence(params, params) == pytest.approx(torch.zeros(2))
    logits = torch.tensor([[2.0, -1.0, 0.5]])
    assert dist.deterministic_output(logits).tolist() == [[1.0, 0.0, 1.0]]


def test_actions_map_to_thruster_bits():
    actions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
    assert actions_to_bitmask(actions).tolist() == [MAIN, LEFT | RIGHT, MAIN | LEFT | RIGHT, 0]


def test_observations_are_not_aliased_across_steps():
    env = RslRocketEnv(RocketEnv(4, device="cpu"))
    before = env.get_observations()["policy"]
    snapshot = before.clone()
    obs, reward, dones, extras = env.step(torch.ones(4, 3))
    assert torch.equal(before, snapshot)
    assert not torch.equal(obs["policy"], snapshot)
    assert reward.shape == dones.shape == extras["time_outs"].shape == (4,)


SHORT_EPISODES = ["sim.spawn.y=30.0", "sim.spawn.vy=-40.0"]


def compose_cfg(*overrides):
    with initialize_config_module("starship.train.conf", version_base=None):
        return compose("train", overrides=["num_envs=64", "runner.num_steps_per_env=8", *overrides])


def make_runner(cfg, log_dir, eval_envs=16) -> StarshipRunner:
    def env(n, seed):
        return RslRocketEnv(RocketEnv(n, OmegaConf.to_object(cfg.sim), device="cpu", seed=seed))

    runner_cfg = OmegaConf.to_container(cfg.runner, resolve=True)
    runner_cfg["save_interval"] = 2
    return StarshipRunner(
        env(cfg.num_envs, 0),
        runner_cfg,
        log_dir,
        "cpu",
        eval_env=env(eval_envs, 1),
        eval_interval=2,
    )


METRIC_GROUPS = [
    "loss/policy",
    "loss/critic",
    "policy/clip_ratio",
    "policy/ess",
    "policy/approx_kl",
    "explore/entropy",
    "explore/entropy_head/main",
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
    "train/env/duty/main",
    "eval/env/return",
    "eval/env/rate/landed",
]


@pytest.mark.parametrize("model", ["lstm", "mlp"])
def test_training_logs_metrics_and_exports_usable_policies(tmp_path, model):
    cfg = compose_cfg(f"model={model}", *SHORT_EPISODES)
    runner = make_runner(cfg, tmp_path)
    runner.learn(2)
    missing = [k for k in METRIC_GROUPS if k not in runner.last_metrics]
    assert not missing
    assert 0.0 < runner.last_metrics["policy/ess"] <= 1.0
    assert runner.last_metrics["eval/env/episodes"] == 16
    assert (tmp_path / "policy_1.pt").exists()
    assert list(tmp_path.glob("events.out.tfevents.*"))

    sim = runner.env.env.sim
    controller = PolicyController(sim, tmp_path / "policy.pt")
    for _ in range(3):
        actions = controller(sim.snapshot())
        sim.step(actions)
    assert actions.shape == (64,)
    assert actions.dtype == np.int32
    assert ((actions >= 0) & (actions < 8)).all()


def test_episode_tracker_summarizes_finished_episodes():
    cfg = compose_cfg(*SHORT_EPISODES)
    env = RslRocketEnv(RocketEnv(8, OmegaConf.to_object(cfg.sim), device="cpu"))
    tracker = EpisodeTracker(env)
    returns = torch.zeros(8)
    done_once = torch.zeros(8, dtype=torch.bool)
    for _ in range(200):
        _, reward, dones, _ = env.step(torch.zeros(8, 3))
        returns += torch.where(done_once, 0.0, reward)
        tracker.step(dones, record=~done_once)
        done_once |= dones.bool()
        if done_once.all():
            break
    summary = tracker.summarize("train/env")
    assert summary["train/env/episodes"] == 8
    assert summary["train/env/rate/crashed"] == 1.0
    assert summary["train/env/duty/main"] == 0.0
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
