"""Replay correctness and dynamics checks before learned gradients can steer a policy."""

import numpy as np
import pytest
import torch
import warp as wp

from starship.sim import SimConfig, Simulator
from starship.train.actions import ACTION_DIM
from starship.train.encoding import ObsConfig
from starship.train.env import RocketEnv
from starship.train.model_experiment import sensitivity_metrics, simulator_rollout
from starship.train.model_state import PHYSICAL_DIM, STATE_DIM, capture_state, restore_state
from starship.train.world_model import Replay, WorldModel, fit_step, physical_errors


def test_recording_preserves_terminal_state_and_default_behavior():
    cfg = SimConfig()
    cfg.max_time = cfg.dt * cfg.substeps
    plain = RocketEnv(4, cfg, device="cpu", seed=7)
    recorded = RocketEnv(4, cfg, device="cpu", seed=7, record_model_state=True)
    action = torch.zeros(4, ACTION_DIM)
    plain.step(action)
    recorded.step(action)
    np.testing.assert_array_equal(recorded.obs.numpy(), plain.obs.numpy())
    np.testing.assert_array_equal(recorded.reward.numpy(), plain.reward.numpy())
    assert recorded.sim.terminal.done.numpy().all()
    after = recorded.model_after.numpy()
    # Penultimate and last channels contain time and steps, before respawn zeroes them.
    np.testing.assert_allclose(after[:, -2], cfg.dt * cfg.substeps, atol=1e-7)
    np.testing.assert_array_equal(after[:, -1], 1)
    np.testing.assert_array_equal(recorded.sim.state.steps.numpy(), 0)
    np.testing.assert_array_equal(recorded.model_before.numpy()[:, -1], 0)
    assert plain.model_before.shape[0] == 0


def test_full_state_roundtrip_and_simulator_replay():
    env = RocketEnv(3, obs_config=ObsConfig(mirror=False), device="cpu", record_model_state=True)
    action = torch.randn(3, ACTION_DIM)
    action[:, :3] = 1
    for _ in range(7):
        env.step(action)
    state = wp.to_torch(env.model_before).clone()
    expected = wp.to_torch(env.model_after).clone()
    sim = Simulator(3, env.config, device="cpu", auto_reset=False)
    wp.launch(restore_state, dim=3, inputs=[wp.from_torch(state), sim.state], device="cpu")
    captured = wp.zeros((3, STATE_DIM), dtype=float, device="cpu")
    wp.launch(capture_state, dim=3, inputs=[sim.state, captured], device="cpu")
    torch.testing.assert_close(wp.to_torch(captured), state, rtol=0, atol=0)
    predicted, alive = simulator_rollout(sim, state, action.unsqueeze(0))
    assert alive.all()
    torch.testing.assert_close(predicted, expected[:, :PHYSICAL_DIM], rtol=1e-6, atol=1e-5)


def test_windows_exclude_resets_and_split_by_environment():
    done = torch.zeros(12, 5, dtype=torch.bool)
    done[4] = True
    replay = Replay({"done": done, "state": torch.zeros(12, 5, 43)})
    train = replay.windows(3)
    valid = replay.windows(3, validation=True)
    assert set(train[:, 1].tolist()).isdisjoint(valid[:, 1].tolist())
    assert not set(train[:, 0].tolist()) & {2, 3, 4}
    assert {0, 1, 5, 6, 7, 8, 9} == set(valid[:, 0].tolist())
    batch = replay.sample(valid, 20, 3)
    assert not batch["done"].any()
    with pytest.raises(ValueError, match="horizon"):
        replay.windows(13)
    with pytest.raises(ValueError, match="two environments"):
        Replay({"done": done[:, :1]})


def test_multistep_fit_learns_action_effect_and_gradient():
    # An action-dependent velocity increment has a known action Jacobian. Learning
    # its sign and magnitude is the minimum requirement for policy-gradient use.
    torch.manual_seed(7)
    model = WorldModel(64)
    count, horizon = 64, 4
    actions = torch.randn(horizon, count, ACTION_DIM) * 0.5
    state = torch.zeros(count, PHYSICAL_DIM)
    state[:, 9] = 1
    states, targets = [], []
    for action in actions:
        states.append(state.clone())
        state = state.clone()
        state[:, 3] += 0.2 * action[:, 3]
        targets.append(state)
    batch = {"state": torch.stack(states), "next_state": torch.stack(targets), "action": actions}
    model.calibrate(batch["state"].flatten(0, 1), batch["next_state"].flatten(0, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    first = fit_step(model, optimizer, batch)
    for _ in range(150):
        last = fit_step(model, optimizer, batch)
    assert last < first * 0.02
    state = batch["state"][0]
    action = torch.zeros(count, ACTION_DIM, requires_grad=True)
    gradient = torch.autograd.grad(model(state, action)[:, 3].sum(), action)[0][:, 3]
    torch.testing.assert_close(gradient, torch.full_like(gradient, 0.2), atol=0.04, rtol=0)
    predicted = model.rollout(state, actions)
    torch.testing.assert_close(predicted[..., 6:10].norm(dim=-1), torch.ones(horizon, count))


def test_actuator_alignment_cannot_hide_wrong_vehicle_gradients():
    actual = torch.ones(4, PHYSICAL_DIM)
    predicted = actual.clone()
    predicted[:, :13] *= -1
    metrics = sensitivity_metrics(actual, predicted, torch.tensor([True, True, False, False]))
    assert metrics["actuators"]["cosine_mean"] == pytest.approx(1)
    for group in ("translation", "rotation"):
        assert metrics[group]["samples"] == 2
        assert metrics[group]["cosine_mean"] == pytest.approx(-1)
        assert metrics[group]["relative_l2"] == pytest.approx(2)


def test_attitude_error_handles_quaternion_sign_and_exact_replay():
    state = torch.randn(3, PHYSICAL_DIM)
    state[:, 6:10] = torch.nn.functional.normalize(state[:, 6:10], dim=-1)
    target = state.clone()
    target[:, 6:10] *= -1
    assert physical_errors(state, state)["attitude_rmse_deg"] == 0
    assert physical_errors(state, target)["attitude_rmse_deg"] == 0


def test_null_sensitivity_reports_invented_vehicle_response():
    metrics = sensitivity_metrics(
        torch.zeros(4, PHYSICAL_DIM), torch.ones(4, PHYSICAL_DIM), torch.ones(4, dtype=torch.bool)
    )
    assert metrics["translation"]["samples"] == 0
    assert metrics["translation"]["null_samples"] == 4
    assert metrics["translation"]["null_predicted_norm_mean"] == pytest.approx(6**0.5)
    assert metrics["translation"]["absolute_rmse"] == pytest.approx(1)
