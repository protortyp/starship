import itertools
import json

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from starship.sim import SimConfig
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


def test_interpolated_stages_expand_to_cumulative_numeric_overrides():
    config = CurriculumConfig(
        enabled=True,
        stages=[
            {
                "name": "base",
                "overrides": ["sim.landing.pad_radius=100", "sim.landing.max_vertical_speed=10"],
            },
            {
                "name": "tighten landing",
                "interpolate": {
                    "steps": 4,
                    "overrides": [
                        "sim.landing.pad_radius=60",
                        "sim.landing.max_vertical_speed=2",
                    ],
                },
            },
            {"name": "final reward", "overrides": ["reward.time_cost=0.5"]},
        ],
    )
    curriculum = Curriculum(compose_cfg(), OmegaConf.create(config))
    assert len(curriculum.stages) == 6
    assert [stage["name"] for stage in curriculum.stages[1:5]] == [
        "tighten landing (1/4)",
        "tighten landing (2/4)",
        "tighten landing (3/4)",
        "tighten landing (4/4)",
    ]
    assert [curriculum.configs(stage)[0].landing.pad_radius for stage in range(1, 5)] == [
        90,
        80,
        70,
        60,
    ]
    assert [curriculum.configs(stage)[0].landing.max_vertical_speed for stage in range(1, 5)] == [
        8,
        6,
        4,
        2,
    ]
    assert curriculum.target_configs()[1].time_cost == pytest.approx(0.5)


def test_fractional_interpolation_inserts_nonuniform_midpoints():
    config = CurriculumConfig(
        enabled=True,
        stages=[
            {"name": "base", "overrides": ["sim.landing.pad_radius=100"]},
            {
                "name": "tighten landing",
                "interpolate": {
                    "fractions": [0.25, 0.375, 1],
                    "overrides": ["sim.landing.pad_radius=60"],
                },
            },
        ],
    )

    curriculum = Curriculum(compose_cfg(), OmegaConf.create(config))

    assert [stage["name"] for stage in curriculum.stages[1:]] == [
        "tighten landing (1/3)",
        "tighten landing (2/3)",
        "tighten landing (3/3)",
    ]
    assert [curriculum.configs(stage)[0].landing.pad_radius for stage in range(1, 4)] == [
        90,
        85,
        60,
    ]


@pytest.mark.parametrize(
    ("interpolation", "message"),
    [
        ({"steps": 0, "overrides": ["sim.landing.pad_radius=10"]}, "positive integer"),
        (
            {"fractions": [0.5, 0.4, 1], "overrides": ["sim.landing.pad_radius=10"]},
            "strictly increasing",
        ),
        (
            {"fractions": [0.5, 0.9], "overrides": ["sim.landing.pad_radius=10"]},
            "end at 1",
        ),
        (
            {
                "steps": 2,
                "fractions": [0.5, 1],
                "overrides": ["sim.landing.pad_radius=10"],
            },
            "either steps or fractions",
        ),
        ({"steps": 2, "overrides": ["sim.wind.enabled=true"]}, "must be numeric"),
        ({"steps": 2, "overrides": ["runner.algorithm.gamma=0.9"]}, "must start with"),
    ],
)
def test_invalid_interpolated_stage_is_rejected(interpolation, message):
    config = CurriculumConfig(
        enabled=True,
        stages=[{"name": "bad interpolation", "interpolate": interpolation}],
    )
    with pytest.raises(ValueError, match=message):
        Curriculum(compose_cfg(), OmegaConf.create(config))


def test_attitude_stages_preserve_spawns_and_resume_indices():
    import math

    cfg = compose_cfg("curriculum=landing")
    curriculum = Curriculum(cfg, cfg.curriculum)
    assert len(curriculum.stages) == 18
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
    assert curriculum.configs(8)[0].wind == SimConfig().wind
    for stage, speed, turbulence in ((9, 5, 1), (10, 10, 2), (11, 15, 2), (12, 15, 2)):
        sim, _ = curriculum.configs(stage)
        assert sim.spawn == spawn4
        assert (sim.wind.surface_speed, sim.wind.turbulence) == (speed, turbulence)
        assert (sim.wind.shedding_amplitude > 0) == (sim.wind.flap_noise > 0) == (stage == 12)
    curriculum.load_state_dict({"stage": 4, "passes": 0})
    assert curriculum.stage == 4
    assert not curriculum.is_final


def test_loop_curriculum_tightens_recovery_limits():
    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose("loop")
    curriculum = Curriculum(cfg, cfg.curriculum)
    assert len(curriculum.stages) == 14
    rates = [curriculum.configs(stage)[0].loop_max_rate for stage in range(14)]
    tolerances = [curriculum.configs(stage)[0].loop_upright_tolerance for stage in range(14)]
    assert rates == pytest.approx(
        np.radians([120, 90, 82.5, 75, 67.5, 60, 52.5, 52.5, 48.75, 45, 41.25, 37.5, 33.75, 30])
    )
    assert tolerances == pytest.approx(
        np.radians([30, 20, 20, 20, 20, 20, 20, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5])
    )


def test_salto_landing_curriculum_finishes_with_a_strict_single_loop():
    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose("salto_land")
    assert cfg.iterations == 2000
    curriculum = Curriculum(cfg, cfg.curriculum)
    assert len(curriculum.stages) == 213
    horizontal, horizontal_reward = curriculum.configs(56)
    assert horizontal.landing.pad_radius == pytest.approx(718.75)
    assert horizontal.landing.max_vertical_speed == pytest.approx(73.75)
    assert horizontal.landing.max_horizontal_speed == pytest.approx(36.8359375)
    assert horizontal_reward.horizontal_speed_weight == pytest.approx(0.5)
    pad, _ = curriculum.configs(57)
    assert pad.landing.pad_radius == pytest.approx(718.359375)
    assert pad.landing.max_vertical_speed == pytest.approx(73.75)
    assert pad.landing.max_horizontal_speed == pytest.approx(36.8359375)
    stage60, stage60_reward = curriculum.configs(60)
    assert stage60.landing.max_vertical_speed == pytest.approx(73.28125)
    assert stage60_reward.distance_weight == pytest.approx(0.05)
    assert stage60_reward.horizontal_distance_weight == pytest.approx(0.05)
    assert stage60_reward.horizontal_speed_weight == pytest.approx(0.5)
    stage62, stage62_reward = curriculum.configs(62)
    assert stage62.landing.max_vertical_speed == pytest.approx(73.046875)
    assert stage62_reward.horizontal_speed_weight == pytest.approx(1.0)
    stage63, _ = curriculum.configs(63)
    assert stage63.landing.max_vertical_speed == pytest.approx(72.96875)
    bridge74, _ = curriculum.configs(74)
    assert bridge74.landing.pad_radius == pytest.approx(687.5)
    assert bridge74.landing.max_vertical_speed == pytest.approx(67.5)
    assert bridge74.landing.max_horizontal_speed == pytest.approx(34.25)
    first_dense, _ = curriculum.configs(78)
    assert first_dense.landing.pad_radius == pytest.approx(640.625)
    assert first_dense.landing.max_vertical_speed == pytest.approx(58.75)
    assert first_dense.landing.max_horizontal_speed == pytest.approx(31.5)
    fifty, _ = curriculum.configs(85)
    assert fifty.landing.pad_radius == pytest.approx(575)
    assert fifty.landing.max_vertical_speed == pytest.approx(50)
    assert fifty.landing.max_horizontal_speed == pytest.approx(28)
    forty, forty_reward = curriculum.configs(93)
    assert forty.landing.max_vertical_speed == pytest.approx(40)
    assert forty_reward.safe_deceleration == pytest.approx(10)
    first_braking_bridge, first_braking_reward = curriculum.configs(94)
    assert first_braking_bridge.landing.max_vertical_speed == pytest.approx(38.75)
    assert first_braking_reward.safe_deceleration == pytest.approx(2)
    blocked_bridge, _ = curriculum.configs(97)
    assert blocked_bridge.landing.pad_radius == pytest.approx(456.25)
    assert blocked_bridge.landing.max_vertical_speed == pytest.approx(35.625)
    assert blocked_bridge.landing.max_horizontal_speed == pytest.approx(22.375)
    original_boundary, original_reward = curriculum.configs(98)
    assert original_boundary.landing.pad_radius == pytest.approx(450)
    assert original_boundary.landing.max_vertical_speed == pytest.approx(35)
    assert original_boundary.landing.max_horizontal_speed == pytest.approx(22)
    assert original_reward.speed_weight == pytest.approx(10)
    assert original_reward.vertical_speed_weight == 0
    count_bridge, count_bridge_reward = curriculum.configs(99)
    assert count_bridge.landing.pad_radius == pytest.approx(700)
    assert count_bridge.landing.max_vertical_speed == pytest.approx(95)
    assert count_bridge.landing.max_horizontal_speed == pytest.approx(60)
    assert count_bridge_reward.speed_weight == 0
    assert count_bridge_reward.vertical_speed_weight == pytest.approx(10)
    assert count_bridge_reward.horizontal_distance_weight == pytest.approx(0.1)
    assert count_bridge_reward.safe_deceleration == pytest.approx(1)
    first_count_step, _ = curriculum.configs(100)
    assert first_count_step.landing.pad_radius == pytest.approx(700 + (450 - 700) / 51)
    assert first_count_step.landing.max_vertical_speed == pytest.approx(95 + (35 - 95) / 51)
    assert first_count_step.landing.max_horizontal_speed == pytest.approx(60 + (22 - 60) / 51)
    restored_boundary, _ = curriculum.configs(150)
    assert restored_boundary.landing.pad_radius == pytest.approx(450)
    assert restored_boundary.landing.max_vertical_speed == pytest.approx(35)
    assert restored_boundary.landing.max_horizontal_speed == pytest.approx(22)
    next_vertical, next_vertical_reward = curriculum.configs(151)
    assert next_vertical.landing.pad_radius == pytest.approx(450)
    assert next_vertical.landing.max_vertical_speed == pytest.approx(34.58333333333333)
    assert next_vertical.landing.max_horizontal_speed == pytest.approx(22)
    assert next_vertical_reward.speed_weight == 0
    assert next_vertical_reward.vertical_speed_weight == pytest.approx(10)
    next_lateral, next_lateral_reward = curriculum.configs(152)
    assert next_lateral.landing.pad_radius == pytest.approx(445.8333333333333)
    assert next_lateral.landing.max_vertical_speed == pytest.approx(34.58333333333333)
    assert next_lateral.landing.max_horizontal_speed == pytest.approx(21.666666666666664)
    assert next_lateral_reward.speed_weight == 0
    assert next_lateral_reward.vertical_speed_weight == pytest.approx(10)
    original_next, _ = curriculum.configs(153)
    assert original_next.landing.pad_radius == pytest.approx(441.6666666666667)
    assert original_next.landing.max_vertical_speed == pytest.approx(34.166666666666664)
    assert original_next.landing.max_horizontal_speed == pytest.approx(21.333333333333332)
    solved_step, _ = curriculum.configs(166)
    assert solved_step.landing.pad_radius == pytest.approx(327.77777777777777)
    assert solved_step.landing.max_vertical_speed == pytest.approx(22.77777777777778)
    assert solved_step.landing.max_horizontal_speed == pytest.approx(13.666666666666666)
    midpoint, _ = curriculum.configs(167)
    assert midpoint.landing.pad_radius == pytest.approx(325)
    assert midpoint.landing.max_vertical_speed == pytest.approx(22.5)
    assert midpoint.landing.max_horizontal_speed == pytest.approx(13.5)
    original_step, _ = curriculum.configs(168)
    assert original_step.landing.pad_radius == pytest.approx(322.22222222222223)
    assert original_step.landing.max_vertical_speed == pytest.approx(22.22222222222222)
    assert original_step.landing.max_horizontal_speed == pytest.approx(13.333333333333334)
    second_midpoint, _ = curriculum.configs(169)
    assert second_midpoint.landing.pad_radius == pytest.approx(319.44444444444446)
    assert second_midpoint.landing.max_vertical_speed == pytest.approx(21.944444444444443)
    assert second_midpoint.landing.max_horizontal_speed == pytest.approx(13.166666666666666)
    second_original_step, _ = curriculum.configs(170)
    assert second_original_step.landing.pad_radius == pytest.approx(316.6666666666667)
    assert second_original_step.landing.max_vertical_speed == pytest.approx(21.666666666666668)
    assert second_original_step.landing.max_horizontal_speed == pytest.approx(13)
    third_midpoint, _ = curriculum.configs(171)
    assert third_midpoint.landing.pad_radius == pytest.approx(313.8888888888889)
    assert third_midpoint.landing.max_vertical_speed == pytest.approx(21.38888888888889)
    assert third_midpoint.landing.max_horizontal_speed == pytest.approx(12.833333333333334)
    fourth_midpoint, _ = curriculum.configs(172)
    assert fourth_midpoint.landing.pad_radius == pytest.approx(312.5)
    assert fourth_midpoint.landing.max_vertical_speed == pytest.approx(21.25)
    assert fourth_midpoint.landing.max_horizontal_speed == pytest.approx(12.75)
    third_original_step, _ = curriculum.configs(173)
    assert third_original_step.landing.pad_radius == pytest.approx(311.1111111111111)
    assert third_original_step.landing.max_vertical_speed == pytest.approx(21.11111111111111)
    assert third_original_step.landing.max_horizontal_speed == pytest.approx(12.666666666666666)
    first_touchdown_bridge, _ = curriculum.configs(176)
    assert first_touchdown_bridge.landing.pad_radius == pytest.approx(298.4375)
    assert first_touchdown_bridge.landing.max_vertical_speed == pytest.approx(19.875)
    assert first_touchdown_bridge.landing.max_horizontal_speed == pytest.approx(11.9375)
    second_touchdown_bridge, _ = curriculum.configs(177)
    assert second_touchdown_bridge.landing.pad_radius == pytest.approx(296.875)
    assert second_touchdown_bridge.landing.max_vertical_speed == pytest.approx(19.75)
    assert second_touchdown_bridge.landing.max_horizontal_speed == pytest.approx(11.875)
    third_touchdown_bridge, _ = curriculum.configs(178)
    assert third_touchdown_bridge.landing.pad_radius == pytest.approx(295.3125)
    assert third_touchdown_bridge.landing.max_vertical_speed == pytest.approx(19.625)
    assert third_touchdown_bridge.landing.max_horizontal_speed == pytest.approx(11.8125)
    measured_touchdown_midpoint, _ = curriculum.configs(179)
    assert measured_touchdown_midpoint.landing.pad_radius == pytest.approx(293.75)
    assert measured_touchdown_midpoint.landing.max_vertical_speed == pytest.approx(19.5)
    assert measured_touchdown_midpoint.landing.max_horizontal_speed == pytest.approx(11.75)
    original_touchdown_step, _ = curriculum.configs(180)
    assert original_touchdown_step.landing.pad_radius == pytest.approx(287.5)
    assert original_touchdown_step.landing.max_vertical_speed == pytest.approx(19)
    assert original_touchdown_step.landing.max_horizontal_speed == pytest.approx(11.5)
    safe_stage = len(curriculum.stages) - 8
    safe, _ = curriculum.configs(safe_stage)
    assert safe.landing.pad_radius == pytest.approx(15)
    assert safe.landing.max_vertical_speed == pytest.approx(2)
    assert safe.landing.max_horizontal_speed == pytest.approx(1)
    dense = [curriculum.configs(stage)[0].landing for stage in range(77, safe_stage + 1)]
    for previous, current in itertools.pairwise(dense):
        assert previous.pad_radius - current.pad_radius <= 12.5
        assert previous.max_vertical_speed - current.max_vertical_speed <= 1.25
        assert previous.max_horizontal_speed - current.max_horizontal_speed <= 0.75
        assert previous.max_tilt - current.max_tilt <= np.radians(5) + 1e-12
        assert previous.max_angular_rate - current.max_angular_rate <= np.radians(10) + 1e-12
    cleanup = [
        curriculum.configs(stage)[1]
        for stage in range(len(curriculum.stages) - 8, len(curriculum.stages))
    ]
    assert [reward.roll_travel_cost for reward in cleanup] == [0, 0.1, 0.25, 0.5, 1, 1, 1, 1]
    assert [reward.reflip_cost for reward in cleanup] == [0, 0, 0, 0, 0, 2.5, 5, 10]
    assert [reward.loop_overshoot_cost for reward in cleanup] == [5, 5, 5, 5, 5, 6.25, 7.5, 10]
    assert [
        curriculum.configs(stage)[0].require_single_loop_before_landing
        for stage in range(len(curriculum.stages) - 8, len(curriculum.stages))
    ] == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    sim, reward = curriculum.target_configs()
    assert sim.require_loop_before_landing
    assert sim.require_single_loop_before_landing
    assert reward.loop_overshoot_cost == 10
    assert reward.roll_travel_cost == 1
    assert reward.reflip_cost == 10


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


def test_salto_landing_expert_config_separates_stunt_and_landing_training():
    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose("salto_land_expert")
    assert cfg.iterations == 2000
    assert cfg.reward.reflip_cost == 10
    assert cfg.runner.algorithm.critic_warmup_updates == 100
    assert cfg.runner.algorithm.post_warmup_learning_rate == 2.5e-7
    assert cfg.runner.algorithm.post_warmup_critic_learning_rate == 1e-6
    assert cfg.runner.algorithm.actor_update_scope == "landing_expert"
    assert cfg.runner.actor.landing_expert
    assert not cfg.runner.actor.update_landing_normalization
    assert cfg.runner.actor.class_name == "starship.train.models:ContinuousAdapterRNNModel"
    assert cfg.runner.actor.resample_engine_count_post_loop
    assert cfg.runner.actor.resample_engine_count_touchdown
    assert cfg.runner.actor.distribution_cfg.std_range == [0.02, 0.02]
    assert cfg.runner.actor.distribution_cfg.std_range_overrides.throttle == [0.005, 0.005]


def test_landing_expert_pretrain_resets_curriculum_and_activates_only_expert():
    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose("landing_expert_pretrain")
    assert not cfg.load_checkpoint_curriculum
    assert cfg.reset_critic
    assert cfg.reset_actor_optimizer
    assert cfg.reward.fuel_cost == 0
    assert cfg.reward.fuel_bonus == 0
    assert cfg.reward.horizontal_distance_weight == 0.10
    assert cfg.reward.crash_miss_weight == 1
    assert cfg.curriculum.stages[0].name == "0 loose touchdown gate"
    curriculum = Curriculum(cfg, cfg.curriculum)
    assert len(curriculum.stages) == 25
    loose, _ = curriculum.configs(0)
    strict, _ = curriculum.target_configs()
    assert (loose.landing.pad_radius, loose.landing.max_vertical_speed) == (100, 10)
    assert loose.landing.max_horizontal_speed == 5
    assert (strict.landing.pad_radius, strict.landing.max_vertical_speed) == (15, 2)
    assert strict.landing.max_horizontal_speed == 1
    for previous_stage, current_stage in itertools.pairwise(range(len(curriculum.stages))):
        previous, _ = curriculum.configs(previous_stage)
        current, _ = curriculum.configs(current_stage)
        assert previous.landing.pad_radius - current.landing.pad_radius <= 3.55
        assert previous.landing.max_vertical_speed - current.landing.max_vertical_speed <= 0.34
        assert previous.landing.max_horizontal_speed - current.landing.max_horizontal_speed <= 0.17
    assert cfg.runner.actor.landing_expert
    assert cfg.runner.actor.landing_direct_hidden_dim == 256
    assert cfg.runner.actor.landing_direct_engine_logit_biases == [2, -2, -2]
    assert cfg.runner.actor.landing_direct_throttle_bias == 0
    assert cfg.runner.actor.landing_progress_override == 1
    assert cfg.runner.actor.update_landing_normalization
    assert cfg.runner.actor.post_loop_progress_threshold == 0
    assert cfg.runner.actor.resample_engine_count_post_loop
    assert cfg.runner.actor.resample_engine_count_touchdown
    assert cfg.runner.actor.distribution_cfg.stochastic_engine_selection == "ranked_count"
    assert cfg.runner.actor.distribution_cfg.deterministic_engine_selection == "round_expected"
    assert cfg.runner.algorithm.actor_update_scope == "landing_expert"
    assert cfg.runner.algorithm.post_warmup_learning_rate == 3e-5
    assert cfg.runner.algorithm.post_warmup_critic_learning_rate == 1e-5
    assert cfg.runner.algorithm.value_normalization


def test_landing_expert_transfer_starts_at_solved_salto_boundary():
    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose("salto_land_expert_transfer")
    assert not cfg.load_checkpoint_curriculum
    assert cfg.reset_critic
    assert cfg.reset_actor_optimizer
    assert cfg.curriculum.start_stage == 179
    assert cfg.runner.actor.landing_direct_hidden_dim == 256
    assert cfg.runner.actor.landing_progress_override == 1
    assert cfg.runner.actor.distribution_cfg.deterministic_engine_selection == "round_expected"
    curriculum = Curriculum(cfg, cfg.curriculum)
    assert curriculum.stage == 179
    assert curriculum.name.endswith("(4/8)")
    assert cfg.runner.actor.post_loop_progress_threshold > 0.99


def test_landing_expert_finetune_preserves_features_and_uses_low_noise():
    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose("landing_expert_finetune")
    assert cfg.reset_critic
    assert cfg.reset_actor_optimizer
    assert cfg.reward.crash_miss_weight == 1
    assert cfg.runner.algorithm.value_normalization
    assert cfg.runner.algorithm.learning_rate == 1e-5
    assert cfg.runner.algorithm.post_warmup_learning_rate == 1e-5
    assert cfg.runner.algorithm.post_warmup_critic_learning_rate == 1e-5
    assert cfg.runner.algorithm.actor_update_scope == "landing_direct_control_head"
    assert not cfg.runner.actor.update_landing_normalization
    assert cfg.runner.actor.distribution_cfg.std_range == [0.05, 0.05]


def test_checkpoint_can_transfer_into_a_fresh_curriculum(tmp_path):
    cfg = compose_cfg("num_envs=8")
    source = make_runner(cfg, tmp_path / "source")
    source.curriculum.advance()
    source._apply_stage()
    source.best_eval = (1, 0.8, 10.0)
    checkpoint = tmp_path / "source" / "model.pt"
    source.save(str(checkpoint))

    transferred = make_runner(cfg, tmp_path / "transferred")
    transferred.load(str(checkpoint), load_curriculum=False)

    assert transferred.curriculum.stage == 0
    assert transferred.env.cfg.spawn.vertical_speed == -30
    assert transferred.best_eval is None


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
