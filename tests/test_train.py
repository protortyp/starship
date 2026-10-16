import math

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from rsl_rl.runners import OnPolicyRunner

from starship.control.policy import PolicyController
from starship.sim import LEFT, MAIN, RIGHT
from starship.train import RocketEnv
from starship.train.distribution import BernoulliDistribution
from starship.train.rsl_env import RslRocketEnv, actions_to_bitmask
from starship.train.train import export_policy


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


def test_training_runs_and_exports_a_usable_policy(tmp_path):
    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose("train", overrides=["num_envs=64", "runner.num_steps_per_env=8"])
    env = RslRocketEnv(RocketEnv(cfg.num_envs, device="cpu"))
    runner_cfg = OmegaConf.to_container(cfg.runner, resolve=True)
    runner = OnPolicyRunner(env, runner_cfg, log_dir=str(tmp_path), device="cpu")
    runner.learn(2)
    export_policy(runner, tmp_path)

    controller = PolicyController(env.env.sim, tmp_path / "policy.pt")
    actions = controller(env.env.sim.snapshot())
    assert actions.shape == (64,)
    assert actions.dtype == np.int32
    assert ((actions >= 0) & (actions < 8)).all()
