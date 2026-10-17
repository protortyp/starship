import math
import subprocess

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
    "train/env/reward/speed",
    "train/env/reward/fuel",
    "train/reward/step_mean",
    "eval/env/return",
    "eval/env/rate/landed",
]


def test_training_logs_metrics_and_exports_usable_policies(tmp_path):
    cfg = compose_cfg(*SHORT_EPISODES)
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
        _, _, dones, _ = env.step(torch.zeros(8, 3))
        returns += torch.where(done_once, 0.0, env.reward)
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


def test_bernoulli_kl_is_finite_for_saturated_logits():
    dist = BernoulliDistribution(3)
    old = torch.tensor([[40.0, -40.0, 0.0]])
    new = torch.tensor([[-40.0, 40.0, 0.5]])
    kl = dist.kl_divergence((old,), (new,))
    assert torch.isfinite(kl).all()
    assert kl.item() == pytest.approx(40.0 + 40.0 + 0.0310, rel=1e-3)
    reference = torch.distributions.kl_divergence(
        torch.distributions.Bernoulli(logits=torch.tensor([0.3, -1.2])),
        torch.distributions.Bernoulli(logits=torch.tensor([-0.4, 2.0])),
    ).sum()
    stable = dist.kl_divergence((torch.tensor([[0.3, -1.2]]),), (torch.tensor([[-0.4, 2.0]]),))
    assert stable.item() == pytest.approx(reference.item(), rel=1e-5)


def test_training_resumes_from_checkpoint(tmp_path):
    cfg = compose_cfg(*SHORT_EPISODES)
    first = make_runner(cfg, tmp_path / "a")
    first.learn(2)
    cfg.runner.algorithm.learning_rate = 1.234e-4
    cfg.runner.algorithm.schedule = "fixed"
    resumed = make_runner(cfg, tmp_path / "b")
    resumed.load(str(tmp_path / "a" / "model_1.pt"))
    assert resumed.alg.optimizer.param_groups[0]["lr"] == pytest.approx(1.234e-4)
    for a, b in zip(
        first.alg.get_policy().parameters(), resumed.alg.get_policy().parameters(), strict=True
    ):
        assert torch.equal(a, b)
    resumed.learn(1)
    assert resumed.current_learning_iteration == 2


def test_learner_sees_scaled_rewards_while_metrics_stay_unscaled():
    from starship.train import RewardConfig

    env = RslRocketEnv(RocketEnv(4, reward_config=RewardConfig(learner_scale=0.05), device="cpu"))
    _, rewards, _, _ = env.step(torch.zeros(4, 3))
    scale = env.env.reward_config.learner_scale
    assert torch.allclose(rewards, env.reward * scale)
    assert not torch.allclose(rewards, env.reward)


def test_mirrored_episodes_look_identical_to_the_policy():
    from dataclasses import replace

    from starship.sim import SimConfig, SpawnConfig

    spawn = SpawnConfig(x=150.0, vx=6.0, theta=1.4, omega=0.03, mirror=True)
    env = RslRocketEnv(RocketEnv(64, replace(SimConfig(), spawn=spawn), device="cpu"))
    mirrored = env._mirrored.clone()
    assert 0 < mirrored.sum() < 64
    # The raw spawns really are mirror images of each other.
    start = env.env.sim.snapshot()
    assert np.all(np.sign(start.pos[:, 0]) == np.where(mirrored.numpy() == 1, -1, 1))
    torch.manual_seed(0)
    obs = env.get_observations()["policy"]
    for _ in range(80):
        assert torch.allclose(obs, obs[:1].expand_as(obs), atol=2e-4)
        action = (torch.rand(1, 3) < 0.5).float().expand(64, 3)
        td, _, dones, _ = env.step(action)
        obs = td["policy"]
        assert not dones.any()


def test_bitmask_swaps_side_thrusters_when_mirrored():
    actions = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    mirrored = torch.tensor([1, 1, 1])
    assert actions_to_bitmask(actions, mirrored).tolist() == [
        MAIN | RIGHT,
        LEFT,
        MAIN | LEFT | RIGHT,
    ]


def test_value_normalizer_tracks_running_moments():
    from starship.train.ppo import ValueNormalizer

    norm = ValueNormalizer()
    data = [torch.randn(1000) * 50 + 20, torch.randn(3000) * 80 - 10]
    for x in data:
        norm.update(x)
    all_x = torch.cat(data)
    assert norm.mean.item() == pytest.approx(all_x.mean().item(), rel=1e-4)
    assert norm.var.item() == pytest.approx(all_x.var(unbiased=False).item(), rel=1e-4)
    x = torch.tensor([3.0, -40.0])
    assert torch.allclose(norm.denormalize(norm.normalize(x)), x, atol=1e-4)


def test_value_normalization_is_saved_with_checkpoints(tmp_path):
    cfg = compose_cfg(*SHORT_EPISODES, "runner.algorithm.value_normalization=true")
    runner = make_runner(cfg, tmp_path / "a")
    runner.learn(2)
    norm = runner.alg.value_norm
    assert norm is not None
    assert norm.count.item() > 0
    assert "value/norm_std" in runner.last_metrics
    resumed = make_runner(cfg, tmp_path / "b")
    resumed.load(str(tmp_path / "a" / "model_1.pt"))
    assert resumed.alg.value_norm.mean.item() == pytest.approx(norm.mean.item())
    assert resumed.alg.value_norm.var.item() == pytest.approx(norm.var.item())


def test_rollback_restores_the_best_policy(tmp_path):
    cfg = compose_cfg(*SHORT_EPISODES)
    runner = make_runner(cfg, tmp_path)
    runner.rollback_drop, runner.rollback_patience = 0.5, 2
    runner.learn(2)
    best = torch.load(tmp_path / "model_best.pt", weights_only=False)["actor_state_dict"]
    runner.learn(1)
    runner.best_eval = (0, 1.0, 1e9)  # pretend the best eval landed everything
    steps = runner.total_env_steps
    runner._maybe_roll_back(3, {"eval/env/rate/landed": 0.0})
    assert runner.rollbacks == 0
    runner._maybe_roll_back(4, {"eval/env/rate/landed": 0.0})
    assert runner.rollbacks == 1
    restored = runner.alg.get_policy().state_dict()
    assert all(torch.equal(restored[k], v) for k, v in best.items())
    assert runner.total_env_steps == steps
    runner._maybe_roll_back(5, {"eval/env/rate/landed": 0.9})
    assert runner._below_best == 0


def test_reward_terms_add_up_to_the_episode_return():
    from starship.train.reward import REWARD_TERMS

    cfg = compose_cfg(*SHORT_EPISODES, "reward.time_cost=0.5")
    env = RslRocketEnv(
        RocketEnv(
            16,
            OmegaConf.to_object(cfg.sim),
            reward_config=OmegaConf.to_object(cfg.reward),
            device="cpu",
        )
    )
    tracker = EpisodeTracker(env)
    done_once = torch.zeros(16, dtype=torch.bool)
    torch.manual_seed(0)
    for _ in range(300):
        _, _, dones, _ = env.step((torch.rand(16, 3) < 0.5).float())
        assert torch.allclose(env.reward, env.reward_terms.sum(-1) + env.terminal_reward, atol=1e-5)
        tracker.step(dones, record=~done_once)
        done_once |= dones.bool()
        if done_once.all():
            break
    m = tracker.summarize("e")
    parts = sum(m[f"e/reward/{t}"] for t in REWARD_TERMS) + m["e/reward/terminal"]
    assert parts == pytest.approx(m["e/return"], rel=1e-4, abs=1e-3)
    assert m["e/reward/time"] < 0.0


def test_eval_episodes_are_recorded_for_rerun(tmp_path):
    cfg = compose_cfg(*SHORT_EPISODES)
    runner = make_runner(cfg, tmp_path)
    runner.record_every, runner.record_envs, runner.eval_interval = 1, 4, 1
    runner.learn(2)
    recordings = sorted((tmp_path / "rerun").glob("eval_*.rrd"))
    assert [p.name for p in recordings] == ["eval_0.rrd", "eval_1.rrd"]
    stats = subprocess.run(
        ["rerun", "rrd", "stats", str(recordings[-1])], capture_output=True, text=True
    ).stdout
    # Per-step data, not only the static scene, must reach the file.
    assert "/world/rockets" in stats
    assert "/plots/altitude" in stats
