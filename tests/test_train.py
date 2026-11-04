import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from starship.train import RocketEnv
from starship.train.actions import ACTION_DIM, ENGINE_HEADS
from starship.train.checkpoint import load_policy
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
    assert summary["train/env/actuators/engine_count_mean"] == 0.0
    assert summary["train/env/crashed/actuators/engine_count_mean"] == 0.0
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


def test_in_process_checkpoint_restore_keeps_post_warmup_learning_rate(tmp_path):
    cfg = compose_cfg(
        *SHORT_EPISODES,
        "runner.algorithm.learning_rate=3e-5",
        "+runner.algorithm.critic_warmup_updates=1",
        "+runner.algorithm.post_warmup_learning_rate=1e-6",
    )
    runner = make_runner(cfg, tmp_path)
    checkpoint = tmp_path / "model.pt"
    runner.save(str(checkpoint))
    runner.alg._updates_done = 2

    runner.load(str(checkpoint))

    assert runner.alg.learning_rate == pytest.approx(1e-6)
    assert all(group["lr"] == pytest.approx(1e-6) for group in runner.alg.optimizer.param_groups)


def test_reset_critic_preserves_actor_and_its_optimizer_state(tmp_path):
    cfg = compose_cfg(*SHORT_EPISODES)
    first = make_runner(cfg, tmp_path / "a")
    first.learn(2)
    resumed = make_runner(cfg, tmp_path / "b")
    resumed.load(str(tmp_path / "a" / "model_1.pt"))
    actor_before = [p.detach().clone() for p in resumed.alg.actor.parameters()]
    critic_before = [p.detach().clone() for p in resumed.alg.critic.parameters()]
    actor_state = {
        p: {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in state.items()
        }
        for p, state in resumed.alg.optimizer.state.items()
        if p in set(resumed.alg.actor.parameters())
    }

    resumed.reset_critic()

    assert all(
        torch.equal(a, p) for a, p in zip(actor_before, resumed.alg.actor.parameters(), strict=True)
    )
    assert any(
        not torch.equal(a, p)
        for a, p in zip(critic_before, resumed.alg.critic.parameters(), strict=True)
    )
    assert not set(resumed.alg.critic.parameters()) & set(resumed.alg.optimizer.state)
    for parameter, state in actor_state.items():
        for key, value in state.items():
            actual = resumed.alg.optimizer.state[parameter][key]
            assert torch.equal(actual, value) if torch.is_tensor(value) else actual == value


def test_reset_actor_optimizer_preserves_weights_and_critic_state(tmp_path):
    cfg = compose_cfg(*SHORT_EPISODES)
    runner = make_runner(cfg, tmp_path)
    runner.learn(2)
    actor_before = [p.detach().clone() for p in runner.alg.actor.parameters()]
    critic_state = {
        p: {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in state.items()
        }
        for p, state in runner.alg.optimizer.state.items()
        if p in set(runner.alg.critic.parameters())
    }

    runner.reset_actor_optimizer()

    assert all(
        torch.equal(a, p) for a, p in zip(actor_before, runner.alg.actor.parameters(), strict=True)
    )
    assert not set(runner.alg.actor.parameters()) & set(runner.alg.optimizer.state)
    for parameter, state in critic_state.items():
        for key, value in state.items():
            actual = runner.alg.optimizer.state[parameter][key]
            assert torch.equal(actual, value) if torch.is_tensor(value) else actual == value


def test_post_loop_throttle_bias_warm_start_changes_only_trainable_residual(tmp_path):
    cfg = compose_cfg(
        *SHORT_EPISODES,
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.post_loop_adapter_hidden_dim=16",
    )
    runner = make_runner(cfg, tmp_path)
    actor = runner.alg.actor
    output = next(
        module
        for module in reversed(list(actor.post_loop_adapter.modules()))
        if isinstance(module, torch.nn.Linear)
    )
    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    runner.alg.optimizer.state[output.bias] = {"stale": torch.tensor(1.0)}

    runner.offset_post_loop_throttle(0.8)

    for name, parameter in actor.named_parameters():
        expected = before[name].clone()
        if parameter is output.bias:
            expected[3] += 0.8
        assert torch.equal(parameter, expected)
    assert output.bias not in runner.alg.optimizer.state


def test_post_loop_throttle_bias_rejects_incompatible_actor(tmp_path):
    runner = make_runner(compose_cfg(), tmp_path)
    with pytest.raises(ValueError, match="post-loop adapter"):
        runner.offset_post_loop_throttle(0.5)
    with pytest.raises(ValueError, match="finite"):
        runner.offset_post_loop_throttle(float("nan"))


def test_post_loop_engine_logit_warm_start_changes_only_trainable_residual(tmp_path):
    cfg = compose_cfg(
        *SHORT_EPISODES,
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.post_loop_adapter_hidden_dim=16",
    )
    runner = make_runner(cfg, tmp_path)
    actor = runner.alg.actor
    output = next(
        module
        for module in reversed(list(actor.post_loop_adapter.modules()))
        if isinstance(module, torch.nn.Linear)
    )
    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    runner.alg.optimizer.state[output.bias] = {"stale": torch.tensor(1.0)}

    runner.offset_post_loop_engine_logits(0.75)

    for name, parameter in actor.named_parameters():
        expected = before[name].clone()
        if parameter is output.bias:
            expected[:3] += 0.75
        assert torch.equal(parameter, expected)
    assert output.bias not in runner.alg.optimizer.state


def test_post_loop_engine_logit_bias_rejects_incompatible_actor(tmp_path):
    runner = make_runner(compose_cfg(), tmp_path)
    with pytest.raises(ValueError, match="post-loop adapter"):
        runner.offset_post_loop_engine_logits(0.5)
    with pytest.raises(ValueError, match="finite"):
        runner.offset_post_loop_engine_logits(float("nan"))


def test_touchdown_action_bias_warm_start_changes_only_gated_residual(tmp_path):
    cfg = compose_cfg(
        *SHORT_EPISODES,
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.touchdown_adapter_hidden_dim=16",
    )
    runner = make_runner(cfg, tmp_path)
    actor = runner.alg.actor
    output = next(
        module
        for module in reversed(list(actor.touchdown_adapter.modules()))
        if isinstance(module, torch.nn.Linear)
    )
    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    runner.alg.optimizer.state[output.bias] = {"stale": torch.tensor(1.0)}

    runner.offset_touchdown_engine_logits(0.64)
    runner.offset_touchdown_throttle(-2.0)

    for name, parameter in actor.named_parameters():
        expected = before[name].clone()
        if parameter is output.bias:
            expected[:3] += 0.64
            expected[3] -= 2.0
        assert torch.equal(parameter, expected)
    assert output.bias not in runner.alg.optimizer.state


def test_touchdown_action_bias_rejects_incompatible_actor(tmp_path):
    runner = make_runner(compose_cfg(), tmp_path)
    with pytest.raises(ValueError, match="touchdown adapter"):
        runner.offset_touchdown_engine_logits(0.5)
    with pytest.raises(ValueError, match="touchdown adapter"):
        runner.offset_touchdown_throttle(-1.0)
    with pytest.raises(ValueError, match="finite"):
        runner.offset_touchdown_engine_logits(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        runner.offset_touchdown_throttle(float("nan"))


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
def test_kl_stop_freezes_policy_but_keeps_training_critic(
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
    actor_before = [p.detach().clone() for p in alg.actor.parameters()]
    critic_before = [p.detach().clone() for p in alg.critic.parameters()]
    metrics = alg.update()
    assert metrics["optim/updates"] == expected_updates
    assert metrics["optim/kl_early_stop"] == (max_kl is not None)
    assert metrics["optim/critic_only_updates"] == 4 - expected_updates
    assert metrics["policy/last_approx_kl"] > 0.01
    assert calls == 4
    actor_moved = any(
        not torch.equal(a, p) for a, p in zip(actor_before, alg.actor.parameters(), strict=True)
    )
    critic_moved = any(
        not torch.equal(a, p) for a, p in zip(critic_before, alg.critic.parameters(), strict=True)
    )
    assert actor_moved == (expected_updates > 0)
    assert critic_moved
    assert alg.storage.step == 0


def test_critic_warmup_freezes_actor_and_trains_critic(tmp_path):
    cfg = compose_cfg(
        "num_envs=8",
        "runner.algorithm.num_learning_epochs=2",
        "runner.algorithm.num_mini_batches=2",
        "+runner.algorithm.critic_warmup_updates=1",
        "+runner.algorithm.post_warmup_learning_rate=1e-7",
    )
    runner = make_runner(cfg, tmp_path)
    alg = runner.alg
    collect(runner)
    actor_before = [p.detach().clone() for p in alg.actor.parameters()]
    critic_before = [p.detach().clone() for p in alg.critic.parameters()]

    metrics = alg.update()

    assert metrics["optim/critic_warmup"] == 1
    assert metrics["optim/updates"] == 0
    assert metrics["optim/critic_only_updates"] == 4
    assert all(torch.equal(a, p) for a, p in zip(actor_before, alg.actor.parameters(), strict=True))
    assert any(
        not torch.equal(a, p) for a, p in zip(critic_before, alg.critic.parameters(), strict=True)
    )
    collect(runner)
    metrics = alg.update()
    assert metrics["optim/critic_warmup"] == 0
    assert metrics["optim/lr/group_0"] == pytest.approx(1e-7)


def test_continuous_head_scope_preserves_engine_schedule_and_backbone(tmp_path):
    cfg = compose_cfg(
        "num_envs=8",
        "runner.algorithm.num_learning_epochs=2",
        "runner.algorithm.num_mini_batches=2",
        "+runner.algorithm.actor_update_scope=continuous_head",
    )
    runner = make_runner(cfg, tmp_path)
    alg = runner.alg
    before = {name: p.detach().clone() for name, p in alg.actor.named_parameters()}
    normalizer = {
        name: value.clone() for name, value in alg.actor.obs_normalizer.state_dict().items()
    }

    collect(runner)
    alg.update()

    after = dict(alg.actor.named_parameters())
    for name, value in before.items():
        if name in ("mlp.4.weight", "mlp.4.bias"):
            assert torch.equal(value[:3], after[name][:3])
            assert not torch.equal(value[3:], after[name][3:])
        elif name == "distribution.log_std":
            assert not torch.equal(value, after[name])
        else:
            assert torch.equal(value, after[name]), name
    for name, value in normalizer.items():
        assert torch.equal(value, alg.actor.obs_normalizer.state_dict()[name])


def test_continuous_adapter_loads_old_policy_exactly_and_trains_only_residual(tmp_path):
    base = make_runner(compose_cfg("num_envs=8"), tmp_path / "base")
    checkpoint = tmp_path / "base" / "model.pt"
    base.save(str(checkpoint))
    obs = base.env.get_observations()
    base.alg.actor.reset()
    expected = base.alg.actor(obs, stochastic_output=False).detach()

    cfg = compose_cfg(
        "num_envs=8",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.algorithm.actor_update_scope=continuous_adapter",
    )
    runner = make_runner(cfg, tmp_path / "adapter")
    runner.load(str(checkpoint))
    actor = runner.alg.actor
    actor.reset()
    actual = actor(obs, stochastic_output=False).detach()
    assert torch.equal(expected, actual)

    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    collect(runner)
    runner.alg.update()
    after = dict(actor.named_parameters())
    for name, value in before.items():
        if name.startswith("continuous_adapter.") or name == "distribution.log_std":
            assert not torch.equal(value, after[name]), name
        else:
            assert torch.equal(value, after[name]), name


def test_continuous_adapter_jit_matches_policy(tmp_path):
    cfg = compose_cfg(
        "num_envs=1",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
    )
    runner = make_runner(cfg, tmp_path)
    actor = runner.alg.actor.eval()
    obs = runner.env.get_observations()
    actor.reset()
    expected = actor(obs, stochastic_output=False)
    with pytest.warns(FutureWarning, match="torch.jit.script"):
        exported = torch.jit.script(actor.as_jit().eval())
    actual = exported(obs["policy"])
    torch.testing.assert_close(actual, expected)


def test_hybrid_adapter_loads_continuous_policy_exactly_and_trains_both_residuals(tmp_path):
    base_cfg = compose_cfg(
        "num_envs=8",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
    )
    base = make_runner(base_cfg, tmp_path / "base")
    checkpoint = tmp_path / "base" / "model.pt"
    base.save(str(checkpoint))
    obs = base.env.get_observations()
    base.alg.actor.reset()
    expected = base.alg.actor(obs, stochastic_output=False).detach()

    cfg = compose_cfg(
        "num_envs=8",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.engine_adapter_hidden_dim=8",
        "+runner.algorithm.actor_update_scope=hybrid_adapter",
    )
    runner = make_runner(cfg, tmp_path / "hybrid")
    runner.load(str(checkpoint))
    actor = runner.alg.actor
    actor.reset()
    actual = actor(obs, stochastic_output=False).detach()
    assert torch.equal(expected, actual)

    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    collect(runner)
    runner.alg.update()
    after = dict(actor.named_parameters())
    for name, value in before.items():
        trainable = name.startswith(("continuous_adapter.", "engine_adapter."))
        if trainable or name == "distribution.log_std":
            assert not torch.equal(value, after[name]), name
        else:
            assert torch.equal(value, after[name]), name


def test_hybrid_adapter_jit_matches_policy(tmp_path):
    cfg = compose_cfg(
        "num_envs=1",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.engine_adapter_hidden_dim=8",
    )
    runner = make_runner(cfg, tmp_path)
    actor = runner.alg.actor.eval()
    obs = runner.env.get_observations()
    actor.reset()
    expected = actor(obs, stochastic_output=False)
    with pytest.warns(FutureWarning, match="torch.jit.script"):
        exported = torch.jit.script(actor.as_jit().eval())
    actual = exported(obs["policy"])
    torch.testing.assert_close(actual, expected)


def test_post_loop_adapter_loads_exactly_and_trains_only_phase_residual(tmp_path):
    base_cfg = compose_cfg(
        "num_envs=8",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.engine_adapter_hidden_dim=8",
    )
    base = make_runner(base_cfg, tmp_path / "base")
    checkpoint = tmp_path / "base" / "model.pt"
    base.save(str(checkpoint))
    obs = base.env.get_observations()
    base.alg.actor.reset()
    expected = base.alg.actor(obs, stochastic_output=False).detach()

    cfg = compose_cfg(
        "num_envs=8",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.engine_adapter_hidden_dim=8",
        "+runner.actor.post_loop_adapter_hidden_dim=16",
        "+runner.actor.post_loop_progress_threshold=0.0",
        "+runner.algorithm.actor_update_scope=post_loop_adapter",
    )
    runner = make_runner(cfg, tmp_path / "post-loop")
    runner.load(str(checkpoint))
    actor = runner.alg.actor
    actor.reset()
    actual = actor(obs, stochastic_output=False).detach()
    assert torch.equal(expected, actual)

    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    collect(runner)
    runner.alg.update()
    after = dict(actor.named_parameters())
    for name, value in before.items():
        trainable = name.startswith("post_loop_adapter.")
        if trainable:
            assert not torch.equal(value, after[name]), name
        else:
            assert torch.equal(value, after[name]), name


def test_post_loop_lateral_head_preserves_engine_throttle_and_shared_features(tmp_path):
    cfg = compose_cfg(
        "num_envs=8",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.engine_adapter_hidden_dim=8",
        "+runner.actor.post_loop_adapter_hidden_dim=16",
        "+runner.actor.post_loop_progress_threshold=0.0",
        "+runner.algorithm.actor_update_scope=post_loop_lateral_head",
    )
    runner = make_runner(cfg, tmp_path)
    actor = runner.alg.actor
    output = next(
        module
        for module in reversed(list(actor.post_loop_adapter.modules()))
        if isinstance(module, torch.nn.Linear)
    )
    output_names = {
        name
        for name, parameter in actor.named_parameters()
        if parameter is output.weight or parameter is output.bias
    }
    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}

    collect(runner)
    runner.alg.update()

    after = dict(actor.named_parameters())
    for name, value in before.items():
        if name in output_names:
            assert torch.equal(value[: ENGINE_HEADS + 1], after[name][: ENGINE_HEADS + 1]), name
            assert not torch.equal(value[ENGINE_HEADS + 1 :], after[name][ENGINE_HEADS + 1 :]), name
        else:
            assert torch.equal(value, after[name]), name


def test_private_post_loop_lateral_adapter_loads_exactly_and_preserves_protected_actions(
    tmp_path,
):
    base_cfg = compose_cfg(
        "num_envs=8",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.engine_adapter_hidden_dim=8",
        "+runner.actor.post_loop_adapter_hidden_dim=16",
        "+runner.actor.post_loop_progress_threshold=0.0",
    )
    base = make_runner(base_cfg, tmp_path / "base")
    checkpoint = tmp_path / "base" / "model.pt"
    base.save(str(checkpoint))
    obs = base.env.get_observations()
    base.alg.actor.reset()
    expected = base.alg.actor(obs, stochastic_output=False).detach()

    cfg = compose_cfg(
        "num_envs=8",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.engine_adapter_hidden_dim=8",
        "+runner.actor.post_loop_adapter_hidden_dim=16",
        "+runner.actor.post_loop_lateral_adapter_hidden_dim=16",
        "+runner.actor.post_loop_progress_threshold=0.0",
        "+runner.algorithm.actor_update_scope=post_loop_lateral_adapter",
    )
    runner = make_runner(cfg, tmp_path / "lateral")
    runner.load(str(checkpoint))
    actor = runner.alg.actor
    for layer in ("0", "2"):
        for field in ("weight", "bias"):
            source = actor.post_loop_adapter.state_dict()[f"{layer}.{field}"]
            lateral = actor.post_loop_lateral_adapter.state_dict()[f"{layer}.{field}"]
            assert torch.equal(source, lateral)
    assert not actor.post_loop_lateral_adapter[4].weight.count_nonzero()
    assert not actor.post_loop_lateral_adapter[4].bias.count_nonzero()
    actor.reset()
    actual = actor(obs, stochastic_output=False).detach()
    assert torch.equal(expected, actual)

    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    collect(runner)
    runner.alg.update()
    after = dict(actor.named_parameters())
    changed = []
    for name, value in before.items():
        if name.startswith("post_loop_lateral_adapter."):
            changed.append(not torch.equal(value, after[name]))
        else:
            assert torch.equal(value, after[name]), name
    assert any(changed)

    actor.reset()
    updated = actor(obs, stochastic_output=False).detach()
    assert torch.equal(expected[..., : ENGINE_HEADS + 1], updated[..., : ENGINE_HEADS + 1])
    assert not torch.equal(expected[..., ENGINE_HEADS + 1 :], updated[..., ENGINE_HEADS + 1 :])
    actor.reset()
    expected_export = actor(obs[:1], stochastic_output=False)
    actor.reset()
    with pytest.warns(FutureWarning, match="torch.jit.script"):
        exported = torch.jit.script(actor.as_jit().eval())
    torch.testing.assert_close(exported(obs["policy"][:1]), expected_export)


def test_post_loop_adapter_is_gated_and_jit_matches_policy(tmp_path):
    cfg = compose_cfg(
        "num_envs=1",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.engine_adapter_hidden_dim=8",
        "+runner.actor.post_loop_adapter_hidden_dim=16",
    )
    runner = make_runner(cfg, tmp_path)
    actor = runner.alg.actor.eval()
    obs = runner.env.get_observations()
    obs["policy"][:, 27] = 0.0
    actor.reset()
    gated_off_before = actor(obs, stochastic_output=False).detach()
    with torch.no_grad():
        output = next(
            module
            for module in reversed(list(actor.post_loop_adapter.modules()))
            if isinstance(module, torch.nn.Linear)
        )
        output.bias.fill_(0.25)
    actor.reset()
    gated_off_after = actor(obs, stochastic_output=False).detach()
    assert torch.equal(gated_off_before, gated_off_after)

    obs["policy"][:, 27] = 1.0
    actor.reset()
    expected = actor(obs, stochastic_output=False)
    actor.reset()
    with pytest.warns(FutureWarning, match="torch.jit.script"):
        exported = torch.jit.script(actor.as_jit().eval())
    actual = exported(obs["policy"])
    torch.testing.assert_close(actual, expected)


def test_touchdown_adapter_loads_exactly_and_trains_only_near_ground_residual(tmp_path):
    base_cfg = compose_cfg(
        "num_envs=8",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.engine_adapter_hidden_dim=8",
    )
    base = make_runner(base_cfg, tmp_path / "base")
    checkpoint = tmp_path / "base" / "model.pt"
    base.save(str(checkpoint))
    obs = base.env.get_observations()
    base.alg.actor.reset()
    expected = base.alg.actor(obs, stochastic_output=False).detach()

    cfg = compose_cfg(
        "num_envs=8",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.engine_adapter_hidden_dim=8",
        "+runner.actor.touchdown_adapter_hidden_dim=16",
        "+runner.actor.post_loop_progress_threshold=0.0",
        "+runner.actor.touchdown_altitude_threshold=100.0",
        "+runner.algorithm.actor_update_scope=touchdown_adapter",
    )
    runner = make_runner(cfg, tmp_path / "touchdown")
    runner.load(str(checkpoint))
    actor = runner.alg.actor
    actor.reset()
    actual = actor(obs, stochastic_output=False).detach()
    assert torch.equal(expected, actual)

    before = {name: parameter.detach().clone() for name, parameter in actor.named_parameters()}
    collect(runner)
    runner.alg.update()
    after = dict(actor.named_parameters())
    for name, value in before.items():
        trainable = name.startswith("touchdown_adapter.")
        if trainable:
            assert not torch.equal(value, after[name]), name
        else:
            assert torch.equal(value, after[name]), name


def test_touchdown_adapter_is_phase_and_altitude_gated_and_jit_matches_policy(tmp_path):
    cfg = compose_cfg(
        "num_envs=1",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.touchdown_adapter_hidden_dim=16",
        "+runner.actor.touchdown_altitude_threshold=0.15",
    )
    runner = make_runner(cfg, tmp_path)
    actor = runner.alg.actor.eval()
    obs = runner.env.get_observations()
    obs["policy"][:, 27] = 0.0
    obs["policy"][:, 12] = 0.1
    actor.reset()
    gated_off_before = actor(obs, stochastic_output=False).detach()
    with torch.no_grad():
        output = next(
            module
            for module in reversed(list(actor.touchdown_adapter.modules()))
            if isinstance(module, torch.nn.Linear)
        )
        output.bias.fill_(0.25)
    actor.reset()
    assert torch.equal(gated_off_before, actor(obs, stochastic_output=False).detach())

    obs["policy"][:, 27] = 1.0
    obs["policy"][:, 12] = 0.2
    with torch.no_grad():
        output.bias.zero_()
    actor.reset()
    above_gate_before = actor(obs, stochastic_output=False).detach()
    with torch.no_grad():
        output.bias.fill_(0.25)
    actor.reset()
    above_gate_after = actor(obs, stochastic_output=False).detach()
    assert torch.equal(above_gate_before, above_gate_after)

    obs["policy"][:, 12] = 0.1
    with torch.no_grad():
        output.bias.zero_()
    actor.reset()
    near_ground_before = actor(obs, stochastic_output=False).detach()
    with torch.no_grad():
        output.bias.fill_(0.25)
    actor.reset()
    expected = actor(obs, stochastic_output=False)
    assert not torch.equal(near_ground_before, expected)
    actor.reset()
    with pytest.warns(FutureWarning, match="torch.jit.script"):
        exported = torch.jit.script(actor.as_jit().eval())
    actual = exported(obs["policy"])
    torch.testing.assert_close(actual, expected)


def test_standalone_policy_saved_before_optional_adapters_still_loads(tmp_path):
    cfg = compose_cfg(
        "num_envs=1",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
    )
    runner = make_runner(cfg, tmp_path)
    actor = runner.alg.actor
    del actor.engine_adapter
    del actor.post_loop_adapter
    del actor.post_loop_progress_index
    del actor.post_loop_progress_threshold
    path = tmp_path / "legacy_policy.pt"
    torch.save(actor, path)

    policy = load_policy(path)
    obs = runner.env.get_observations().cpu()
    policy.reset()
    action = policy(obs, stochastic_output=True)
    assert action.shape == (1, ACTION_DIM)
    with pytest.warns(FutureWarning, match="torch.jit.script"):
        exported = torch.jit.script(policy.as_jit().eval())
    assert exported(obs["policy"]).shape == (1, ACTION_DIM)


def test_ranked_engine_count_resamples_once_at_post_loop_handoff(tmp_path):
    cfg = compose_cfg(
        "num_envs=2",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.resample_engine_count_post_loop=true",
        "+runner.actor.distribution_cfg.stochastic_engine_selection=ranked_count",
        "+runner.actor.distribution_cfg.ranked_count_temperature=0.01",
        "+runner.actor.distribution_cfg.ranked_count_hold_steps=1000",
    )
    runner = make_runner(cfg, tmp_path)
    actor = runner.alg.actor
    output = next(
        module
        for module in reversed(list(actor.mlp.modules()))
        if isinstance(module, torch.nn.Linear)
    )
    obs = runner.env.get_observations()
    obs["policy"][:, 27] = 0.0
    actor.reset()
    with torch.no_grad():
        output.weight[:3].zero_()
        output.bias[:3].fill_(-10.0)
    with torch.inference_mode():
        before = actor(obs, stochastic_output=True)[:, :3].sum(-1)
        assert torch.equal(before, torch.tensor([0.0, 0.0]))
        assert torch.equal(actor.distribution.params[3], torch.tensor([0.0, 0.0]))

        output.bias[:3].fill_(1.0)
        obs["policy"][:, 27] = 1.0
        at_handoff = actor(obs, stochastic_output=True)[:, :3].sum(-1)
        assert torch.equal(at_handoff, torch.tensor([2.0, 2.0]))
        assert torch.equal(actor.distribution.params[3], torch.tensor([1.0, 1.0]))

        # The handoff samples only once: later logits do not interrupt the coherent burn.
        output.bias[:3].fill_(-10.0)
        held = actor(obs, stochastic_output=True)[:, :3].sum(-1)
        assert torch.equal(held, torch.tensor([2.0, 2.0]))
        assert torch.equal(actor.distribution.params[3], torch.tensor([0.0, 0.0]))

        actor.reset(torch.tensor([True, False]))
        after_done = actor(obs, stochastic_output=True)[:, :3].sum(-1)
        assert torch.equal(after_done, torch.tensor([0.0, 2.0]))
        assert torch.equal(actor.distribution.params[3], torch.tensor([1.0, 0.0]))


def test_ranked_engine_count_is_held_deterministically_after_post_loop_handoff(tmp_path):
    cfg = compose_cfg(
        "num_envs=2",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.resample_engine_count_post_loop=true",
        "+runner.actor.distribution_cfg.deterministic_engine_selection=floor_expected",
        "+runner.actor.distribution_cfg.stochastic_engine_selection=ranked_count",
    )
    runner = make_runner(cfg, tmp_path)
    actor = runner.alg.actor
    output = next(
        module
        for module in reversed(list(actor.mlp.modules()))
        if isinstance(module, torch.nn.Linear)
    )
    obs = runner.env.get_observations()
    actor.reset()
    with torch.no_grad():
        output.weight[:3].zero_()
        output.bias[:3].fill_(1.0)
        obs["policy"][:, 27] = 1.0
        at_handoff = actor(obs, stochastic_output=False)[:, :3].sum(-1)
        assert torch.equal(at_handoff, torch.tensor([2.0, 2.0]))

        # Deployment now preserves the handoff count just like the persistent rollout sampler.
        output.bias[:3].fill_(-10.0)
        hidden = actor.get_hidden_state()
        actor.reset(hidden_state=hidden)
        held = actor(obs, stochastic_output=False)[:, :3].sum(-1)
        assert torch.equal(held, torch.tensor([2.0, 2.0]))

        actor.reset(torch.tensor([True, False]))
        after_done = actor(obs, stochastic_output=False)[:, :3].sum(-1)
        assert torch.equal(after_done, torch.tensor([0.0, 2.0]))


def test_ranked_engine_count_reselects_once_at_touchdown_gate_and_jit_matches(tmp_path):
    cfg = compose_cfg(
        "num_envs=1",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.touchdown_adapter_hidden_dim=16",
        "+runner.actor.touchdown_altitude_threshold=0.15",
        "+runner.actor.resample_engine_count_post_loop=true",
        "+runner.actor.resample_engine_count_touchdown=true",
        "+runner.actor.distribution_cfg.deterministic_engine_selection=floor_expected",
        "+runner.actor.distribution_cfg.stochastic_engine_selection=ranked_count",
        "+runner.actor.distribution_cfg.ranked_count_temperature=0.01",
        "+runner.actor.distribution_cfg.ranked_count_hold_steps=1000",
    )
    runner = make_runner(cfg, tmp_path)
    actor = runner.alg.actor.eval()
    base_output = next(
        module
        for module in reversed(list(actor.mlp.modules()))
        if isinstance(module, torch.nn.Linear)
    )
    touchdown_output = next(
        module
        for module in reversed(list(actor.touchdown_adapter.modules()))
        if isinstance(module, torch.nn.Linear)
    )
    obs = runner.env.get_observations()
    obs["policy"][:, 27] = 1.0
    obs["policy"][:, 12] = 0.2
    with torch.no_grad():
        base_output.weight[:3].zero_()
        base_output.bias[:3].zero_()
        touchdown_output.weight.zero_()
        touchdown_output.bias.zero_()
        touchdown_output.bias[:3].fill_(1.0)

    actor.reset()
    with torch.inference_mode():
        assert actor(obs, stochastic_output=True)[:, :3].sum() == 1
        assert actor.distribution.params[3].item() == 1.0
        obs["policy"][:, 12] = 0.1
        assert actor(obs, stochastic_output=True)[:, :3].sum() == 2
        assert actor.distribution.params[3].item() == 1.0
        touchdown_output.bias[:3].fill_(-10.0)
        assert actor(obs, stochastic_output=True)[:, :3].sum() == 2
        assert actor.distribution.params[3].item() == 0.0

    with torch.no_grad():
        touchdown_output.bias[:3].fill_(1.0)
    obs["policy"][:, 12] = 0.2
    actor.reset()
    assert actor(obs, stochastic_output=False)[:, :3].sum() == 1
    obs["policy"][:, 12] = 0.1
    assert actor(obs, stochastic_output=False)[:, :3].sum() == 2
    with torch.no_grad():
        touchdown_output.bias[:3].fill_(-10.0)
    assert actor(obs, stochastic_output=False)[:, :3].sum() == 2

    with torch.no_grad():
        touchdown_output.bias[:3].fill_(1.0)
    actor.reset()
    with pytest.warns(FutureWarning, match="torch.jit.script"):
        exported = torch.jit.script(actor.as_jit().eval())
    obs["policy"][:, 12] = 0.2
    assert exported(obs["policy"])[:, :3].sum() == 1
    obs["policy"][:, 12] = 0.1
    assert exported(obs["policy"])[:, :3].sum() == 2
    with torch.no_grad():
        exported.touchdown_adapter[-1].bias[:3].fill_(-10.0)
    assert exported(obs["policy"])[:, :3].sum() == 2


def test_ranked_engine_count_is_held_in_jit_export(tmp_path):
    cfg = compose_cfg(
        "num_envs=1",
        "runner.actor.class_name=starship.train.models:ContinuousAdapterRNNModel",
        "+runner.actor.continuous_adapter_hidden_dim=16",
        "+runner.actor.resample_engine_count_post_loop=true",
        "+runner.actor.distribution_cfg.deterministic_engine_selection=floor_expected",
        "+runner.actor.distribution_cfg.stochastic_engine_selection=ranked_count",
    )
    runner = make_runner(cfg, tmp_path)
    actor = runner.alg.actor.eval()
    output = next(
        module
        for module in reversed(list(actor.mlp.modules()))
        if isinstance(module, torch.nn.Linear)
    )
    obs = runner.env.get_observations()
    obs["policy"][:, 27] = 1.0
    with torch.no_grad():
        output.weight[:3].zero_()
        output.bias[:3].fill_(1.0)
    with pytest.warns(FutureWarning, match="torch.jit.script"):
        exported = torch.jit.script(actor.as_jit().eval())
    with torch.no_grad():
        assert exported(obs["policy"])[:, :3].sum() == 2
        exported.mlp[-1].bias[:3].fill_(-10.0)
        assert exported(obs["policy"])[:, :3].sum() == 2
        exported.reset()
        assert exported(obs["policy"])[:, :3].sum() == 0


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
