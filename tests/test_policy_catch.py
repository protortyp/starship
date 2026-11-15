import math

import numpy as np
import pytest
import torch
from torch import nn

from starship.sim import CRASHED, LANDED, Simulator
from starship.train.actions import ACTION_DIM, ARM_HEAD, LEGACY_ACTION_DIM
from starship.train.checkpoint import _output_layer, load_policy
from starship.train.distribution import HybridDistribution
from starship.train.env import RocketEnv
from starship.train.runner import export_policy
from tests import test_train
from tests.test_catch import at_plane
from tests.test_moving_catch import moving_config


def policy_config():
    cfg = moving_config()
    cfg.catch.policy_controlled = True
    return cfg


def test_policy_controls_closing_and_reopening_without_clearance_override():
    cfg = policy_config()
    # Fins have cleared here: the automatic controller would close.
    sim = at_plane(cfg, clearance=10)
    opened = sim.snapshot().arm_angle[0]
    sim.step(arm_close=[0])
    assert sim.snapshot().arm_angle[0] == opened
    sim.step(arm_close=[1])
    closed = sim.snapshot().arm_angle[0]
    assert opened - closed == pytest.approx(
        cfg.catch.angular_speed * cfg.dt * cfg.substeps, abs=1e-6
    )
    sim.step(arm_close=[0])
    assert sim.snapshot().arm_angle[0] == pytest.approx(opened)

    # Fins have not passed here: the automatic controller would hold open.
    high = at_plane(cfg, clearance=100)
    high.step(arm_close=[1])
    assert high.snapshot().arm_angle[0] < opened


def test_early_closure_causes_fin_strike():
    cfg = policy_config()
    sim = at_plane(cfg, clearance=30)  # base at 75 m, aft fins overlap arm height
    bodies = sim.state.body.numpy().copy()
    flaps = sim.state.flaps.numpy().copy()
    flaps["angle"] = math.pi / 4
    for _ in range(22):
        sim.state.body.assign(bodies)  # hold the geometry while the real motor closes
        sim.state.flaps.assign(flaps)
        sim.step(arm_close=[1], flaps=np.full((1, 4), math.pi / 4))
        if sim.snapshot().status[0] == CRASHED:
            break
    assert sim.snapshot().status[0] == CRASHED
    assert sim.snapshot().catch_failure[0] == 4  # aft fin against arm


@pytest.mark.parametrize(("closed", "expected"), [(False, CRASHED), (True, LANDED)])
def test_policy_must_finish_closing_before_pin_crossing(closed, expected):
    sim = at_plane(policy_config())
    if closed:
        sim.state.arm_angle.zero_()
    sim.step(arm_close=[1])
    assert sim.snapshot().status[0] == expected


def test_arm_action_is_binary_and_independent_of_mirroring():
    env = RocketEnv(2, policy_config(), device="cpu")
    env.mirrored.assign(np.array([0, 1], dtype=np.int32))
    actions = np.zeros((2, ACTION_DIM), dtype=np.float32)
    actions[:, ARM_HEAD] = [0, 1]
    env.step(actions)
    np.testing.assert_array_equal(env.sim.commands.arm_close.numpy(), [0, 1])
    assert env.sim.snapshot().arm_angle[0] > env.sim.snapshot().arm_angle[1]


def test_arm_logit_receives_policy_gradient():
    logits = torch.zeros(4, ACTION_DIM, requires_grad=True)
    dist = HybridDistribution()
    dist.update(logits)
    actions = dist.sample()
    assert torch.isin(actions[:, ARM_HEAD], torch.tensor([0.0, 1.0])).all()
    dist.log_prob(actions).sum().backward()
    assert (logits.grad[:, ARM_HEAD].abs() > 0).all()


@pytest.mark.parametrize("field", ["enabled", "moving", "flap_collisions"])
def test_policy_control_requires_moving_collision_geometry(field):
    cfg = policy_config()
    setattr(cfg.catch, field, False)
    with pytest.raises(ValueError, match=r"(moving catch|flap collisions)"):
        Simulator(1, cfg, device="cpu")


def test_legacy_action_checkpoint_and_adam_migration(tmp_path):
    cfg = test_train.compose_cfg("num_envs=8")
    old = test_train.make_runner(cfg, tmp_path / "old", eval_envs=4)
    test_train.collect(old)
    old.alg.update()
    actor = old.alg.actor
    _, layer = _output_layer(actor)
    # Recreate an eleven-action export and optimizer from a trained network.
    for name in ("weight", "bias"):
        parameter = getattr(layer, name)
        replacement = nn.Parameter(parameter.detach()[:LEGACY_ACTION_DIM].clone())
        state = old.alg.optimizer.state.pop(parameter)
        for key in ("exp_avg", "exp_avg_sq"):
            state[key] = state[key][:LEGACY_ACTION_DIM].clone()
        old.alg.optimizer.state[replacement] = state
        for group in old.alg.optimizer.param_groups:
            group["params"] = [replacement if p is parameter else p for p in group["params"]]
        setattr(layer, name, replacement)
    layer.out_features = LEGACY_ACTION_DIM
    actor.distribution.output_dim = LEGACY_ACTION_DIM
    actor.eval()
    actor.reset()
    checkpoint = tmp_path / "legacy.pt"
    old.save(str(checkpoint))
    exported = tmp_path / "policy.pt"
    export_policy(actor, exported)
    policy = load_policy(exported)
    resumed = test_train.make_runner(cfg, tmp_path / "new", eval_envs=4)
    resumed.load(str(checkpoint))
    resumed.alg.actor.eval()
    obs = resumed.env.get_observations()
    with torch.no_grad():
        expected = actor(obs)
        for model in (policy, resumed.alg.actor):
            actual = model(obs)
            torch.testing.assert_close(actual[:, :LEGACY_ACTION_DIM], expected)
            assert not actual[:, ARM_HEAD].any()
    _, new_layer = _output_layer(resumed.alg.actor)
    for name in ("weight", "bias"):
        old_state = old.alg.optimizer.state[getattr(layer, name)]
        new_state = resumed.alg.optimizer.state[getattr(new_layer, name)]
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(new_state[key][:LEGACY_ACTION_DIM], old_state[key])
            assert not new_state[key][ARM_HEAD].any()
    resumed._reset_rollout()
    resumed.alg.train_mode()
    test_train.collect(resumed)
    assert resumed.alg.update()["optim/updates"] > 0
