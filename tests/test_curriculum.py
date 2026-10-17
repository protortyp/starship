import json

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from starship.train.curriculum import Curriculum, CurriculumConfig
from starship.train.runner import StarshipRunner
from tests.test_train import SHORT_EPISODES, collect, compose_cfg, make_env


def two_stage(promote_landed=0.9, promote_evals=2) -> CurriculumConfig:
    return CurriculumConfig(
        enabled=True,
        promote_landed=promote_landed,
        promote_evals=promote_evals,
        stages=[
            {
                "name": "base",
                "overrides": ["sim.spawn.altitude=60", "sim.spawn.vertical_speed=-30"],
            },
            {
                "name": "harder",
                "overrides": ["sim.spawn.vertical_speed=-35", "reward.time_cost=0.5"],
            },
        ],
    )


def test_stage_overrides_accumulate():
    curriculum = Curriculum(compose_cfg(), OmegaConf.structured(two_stage()))
    sim0, reward0 = curriculum.configs(0)
    sim1, reward1 = curriculum.configs(1)
    base_time_cost = compose_cfg().reward.time_cost
    spawn0, spawn1 = sim0.spawn, sim1.spawn
    assert (spawn0.altitude, spawn0.vertical_speed, reward0.time_cost) == (60, -30, base_time_cost)
    assert (spawn1.altitude, spawn1.vertical_speed, reward1.time_cost) == (60.0, -35.0, 0.5)
    assert curriculum.target_configs()[0].spawn.vertical_speed == -35.0


def test_attitude_stages_preserve_spawns_and_resume_indices():
    import math

    cfg = compose_cfg("curriculum=landing")
    curriculum = Curriculum(cfg, cfg.curriculum)
    assert len(curriculum.stages) == 9
    weights = ("roll_travel_cost", "horizontal_rotation_cost", "sideslip_cost", "reflip_cost")
    for stage in range(5):
        sim, reward = curriculum.configs(stage)
        assert [getattr(reward, key) for key in weights] == [0, 0, 0, 0]
        assert sim.max_roll_rate == pytest.approx(math.pi / 2)
    spawn4 = curriculum.configs(4)[0].spawn
    for stage in range(5, 9):
        sim, reward = curriculum.configs(stage)
        assert sim.spawn == spawn4
        expected = [1, 1, 0.25, 2.5] if stage == 5 else [2, 2, 0.5, 5]
        assert [getattr(reward, key) for key in weights] == expected
        assert sim.max_roll_rate == pytest.approx(math.radians({7: 60, 8: 45}.get(stage, 90)))
    assert 20 < 7.2 * reward.roll_travel_cost + 5.7 * reward.horizontal_rotation_cost < 30
    assert 1.6 * reward.horizontal_rotation_cost == pytest.approx(3.2)
    curriculum.load_state_dict({"stage": 4, "passes": 0})
    assert curriculum.stage == 4
    assert not curriculum.is_final


def test_overrides_outside_sim_and_reward_are_rejected():
    cfg = two_stage()
    cfg.stages[0]["overrides"] = ["runner.algorithm.gamma=0.9"]
    with pytest.raises(ValueError, match="must start with"):
        Curriculum(compose_cfg(), OmegaConf.structured(cfg)).configs(0)


def test_promotion_needs_consecutive_passing_evals():
    curriculum = Curriculum(compose_cfg(), OmegaConf.structured(two_stage(promote_evals=2)))
    assert not curriculum.observe_eval(0.95)
    assert not curriculum.observe_eval(0.5)
    assert not curriculum.observe_eval(0.95)
    assert curriculum.observe_eval(0.92)
    curriculum.advance()
    assert curriculum.is_final
    assert not curriculum.observe_eval(1.0)
    assert not curriculum.observe_eval(1.0)


def make_runner(cfg, log_dir) -> StarshipRunner:
    curriculum = Curriculum(cfg, OmegaConf.structured(two_stage(promote_landed=0.0)))

    def env(n, seed):
        return make_env(cfg, n, seed)

    runner_cfg = OmegaConf.to_container(cfg.runner, resolve=True)
    runner_cfg["save_interval"] = 100
    return StarshipRunner(
        env(cfg.num_envs, 0),
        runner_cfg,
        log_dir,
        "cpu",
        eval_env=env(8, 1),
        eval_interval=1,
        curriculum=curriculum,
        target_eval_env=env(8, 2),
    )


def test_runner_promotes_logs_and_resumes(tmp_path):
    cfg = compose_cfg(*SHORT_EPISODES)
    runner = make_runner(cfg, tmp_path / "a")
    assert runner.env.env.config.spawn.vertical_speed == -30.0
    runner.learn(3)

    assert runner.curriculum.stage == 1
    assert runner.env.env.config.spawn.vertical_speed == -35.0
    assert runner.env.env.reward_config.time_cost == 0.5
    events = json.loads((tmp_path / "a" / "curriculum.json").read_text())
    assert [(e["stage"], e["name"], e["iteration"]) for e in events] == [(1, "harder", 1)]
    assert (tmp_path / "a" / "model_stage0_passed.pt").exists()
    assert runner.last_metrics["curriculum/stage"] == 1.0
    assert "eval/target/rate/landed" in runner.last_metrics

    resumed = make_runner(cfg, tmp_path / "b")
    resumed.load(str(tmp_path / "a" / "model_2.pt"))
    assert resumed.curriculum.stage == 1
    assert resumed.env.env.config.spawn.vertical_speed == -35.0


def test_start_stage_skips_solved_stages():
    cfg = two_stage()
    cfg.start_stage = 1
    curriculum = Curriculum(compose_cfg(), OmegaConf.structured(cfg))
    assert curriculum.stage == 1
    assert curriculum.configs()[1].time_cost == 0.5
    cfg.start_stage = 5
    with pytest.raises(ValueError, match="start_stage"):
        Curriculum(compose_cfg(), OmegaConf.structured(cfg))


def assert_fresh(runner):
    for env in (runner.env, runner.eval_env, runner.target_eval_env):
        state = env.env.sim.snapshot()
        np.testing.assert_array_equal(state.steps, 0)
        np.testing.assert_allclose(state.pos[:, 2], env.cfg.spawn.altitude)
        np.testing.assert_allclose(state.vel[:, 2], env.cfg.spawn.vertical_speed)
    assert runner.alg.actor.get_hidden_state() is None
    assert runner.alg.critic.get_hidden_state() is None
    assert runner.alg.transition.observations is None
    assert runner.alg.storage.step == 0
    assert runner.alg.storage.saved_hidden_state_a is None
    assert runner.alg.storage.saved_hidden_state_c is None
    assert not runner.tracker._return.any()
    assert runner.tracker.num_finished == 0


def test_initial_spawns_use_curriculum_stage(tmp_path):
    runner = make_runner(compose_cfg("num_envs=8"), tmp_path)
    assert runner.env.cfg.spawn.altitude == 60
    assert_fresh(runner)


@pytest.mark.parametrize("rollback", [False, True])
def test_checkpoint_recovery_resets_stage_and_rollout(tmp_path, rollback):
    cfg = compose_cfg("num_envs=8")
    runner = make_runner(cfg, tmp_path)
    runner.curriculum.advance()
    runner._apply_stage()
    runner.best_eval = (1, 0.8, 10.0)
    checkpoint = tmp_path / "model_best.pt"
    runner.save(str(checkpoint))
    saved = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    if rollback:
        runner.best_eval = None
        runner.load(str(checkpoint))
        assert runner.best_eval == (1, 0.8, 10.0)
    runner.curriculum.stage = 0
    runner._apply_stage()
    collect(runner, steps=2)
    assert runner.alg.storage.step == 2
    assert runner.alg.actor.get_hidden_state() is not None
    with torch.no_grad():
        next(runner.alg.actor.parameters()).add_(1)
    if rollback:
        runner.curriculum.stage = 1
        runner.rollback_drop, runner.rollback_patience = 0.5, 2
        runner.total_env_steps, runner.next_iteration = 1234, 9
        runner.current_learning_iteration = 8
        runner._maybe_roll_back(8, {"eval/env/rate/landed": 0.2})
        assert runner.rollbacks == 0
        runner._maybe_roll_back(8, {"eval/env/rate/landed": 0.2})
        assert runner.rollbacks == 1
        assert (runner.total_env_steps, runner.next_iteration) == (1234, 9)
        assert runner.current_learning_iteration == 8
    else:
        runner.best_eval = None
        runner.load(str(checkpoint))
    assert runner.best_eval == (1, 0.8, 10.0)
    assert runner.curriculum.stage == 1
    assert_fresh(runner)
    for key, value in runner.alg.actor.state_dict().items():
        torch.testing.assert_close(value, saved[key])
    fresh_obs = runner.env.get_observations()["policy"]
    collect(runner)
    torch.testing.assert_close(runner.alg.storage.observations[0]["policy"], fresh_obs)
    assert runner.alg.update()["optim/updates"] > 0


def test_rollback_refreshes_next_training_observations(tmp_path, monkeypatch):
    runner = make_runner(compose_cfg("num_envs=8"), tmp_path)
    runner.best_eval = (0, 1.0, 100.0)
    runner.save(str(tmp_path / "model_best.pt"))
    runner.rollback_drop, runner.rollback_patience = 0.5, 1
    runner.curriculum.promote_evals = 100
    monkeypatch.setattr(
        runner,
        "evaluate",
        lambda *args, **kwargs: {"eval/env/rate/landed": 0.0, "eval/env/return": -100.0},
    )
    act = runner.alg.act
    checked = []

    def check_obs(obs):
        if runner.rollbacks and runner.alg.storage.step == 0:
            assert_fresh(runner)
            torch.testing.assert_close(obs["policy"], runner.env.get_observations()["policy"])
            checked.append(True)
        return act(obs)

    monkeypatch.setattr(runner.alg, "act", check_obs)
    runner.learn(2)
    assert checked == [True]


def test_old_run_configs_with_removed_settings_still_load(tmp_path):
    from starship.train.conf import load_run_config

    cfg = compose_cfg()
    hydra = tmp_path / ".hydra"
    hydra.mkdir()
    saved = OmegaConf.to_container(cfg, resolve=False)
    saved["sim"]["vehicle"]["aero"]["axial_nose"] = 0.4  # a setting the schema no longer has
    (hydra / "config.yaml").write_text(OmegaConf.to_yaml(saved))
    loaded = load_run_config(tmp_path)
    assert "axial_nose" not in loaded.sim.vehicle.aero
    assert loaded.num_envs == cfg.num_envs
    assert loaded.runner.algorithm.lam == cfg.runner.algorithm.lam  # free-form section kept
