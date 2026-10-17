import json

import pytest
from omegaconf import OmegaConf

from starship.train.curriculum import Curriculum, CurriculumConfig
from starship.train.runner import StarshipRunner
from tests.test_train import SHORT_EPISODES, compose_cfg, make_env


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
