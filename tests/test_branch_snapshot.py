import random

import numpy as np
import pytest
import torch
import warp as wp

from starship.train.branch_snapshot import BranchSnapshot
from starship.train.env import RocketEnv
from starship.train.rsl_env import RslRocketEnv
from starship.train.runner import StarshipRunner
from tests.test_train import SHORT_EPISODES, compose_cfg


@pytest.fixture(params=[str(device) for device in wp.get_devices()])
def device(request):
    return request.param


def test_snapshot_replays_noise_memory_rewards_and_autoresets(tmp_path, device):
    from omegaconf import OmegaConf

    cfg = compose_cfg(
        *SHORT_EPISODES, "num_envs=4", "sim.wind.surface_speed=5", "sim.wind.turbulence=1"
    )
    env = RslRocketEnv(RocketEnv(4, OmegaConf.to_object(cfg.sim), device=device, seed=91))
    runner = StarshipRunner(env, OmegaConf.to_container(cfg.runner), tmp_path, device)
    actor, critic = runner.alg.actor.eval(), runner.alg.critic.eval()

    def advance():
        obs = env.get_observations()
        with torch.inference_mode():
            action = actor(obs, stochastic_output=True)
            value = critic(obs)
            next_obs, reward, done, _ = env.step(action)
            actor.reset(done)
            critic.reset(done)
        return [
            v.detach().cpu().numpy().copy()
            for v in (action, value, next_obs["policy"], reward, done)
        ] + [np.array([random.random(), np.random.random()])]  # noqa: NPY002 - test global replay

    for _ in range(3):
        advance()
    snapshot = BranchSnapshot.capture(env, actor, critic, provenance={"purpose": "test"})
    snapshot.save(tmp_path / "branch.pt")
    baseline = [advance() for _ in range(30)]
    final = BranchSnapshot.capture(env, actor, critic)
    restored = BranchSnapshot.load(tmp_path / "branch.pt")
    restored.restore(env, actor, critic)
    repeated = [advance() for _ in range(30)]
    assert sum(row[4].sum() for row in baseline) > 0
    for before, after in zip(baseline, repeated, strict=True):
        for a, b in zip(before, after, strict=True):
            np.testing.assert_array_equal(a, b)
    repeated_final = BranchSnapshot.capture(env, actor, critic)
    for key in final.arrays:
        np.testing.assert_array_equal(final.arrays[key], repeated_final.arrays[key])
    with torch.no_grad():
        next(actor.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="parameters or normalizer differ"):
        restored.restore(env, actor, critic)
    runner.writer.close()
