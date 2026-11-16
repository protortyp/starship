from copy import deepcopy

import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from rsl_rl.modules.rnn import RNN

from starship.train.actions import ACTION_DIM
from starship.train.env import RocketEnv
from starship.train.metrics import EpisodeTracker
from starship.train.mixture_env import MixtureRocketEnv
from starship.train.rsl_env import RslRocketEnv


def profiles(device, layout=None):
    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose("train", overrides=["curriculum=none"])
    result = {}
    layout = layout or [("frontier", 4), ("warm", 2), ("cold", 2)]
    for i, (name, n) in enumerate(layout):
        sim = OmegaConf.to_object(cfg.sim)
        sim.spawn.altitude = 100 + i * 100
        sim.spawn.pitch = 0
        sim.spawn.vertical_speed = -5
        sim.max_time = 0.4 + i * 0.2
        result[name] = RslRocketEnv(
            RocketEnv(
                n,
                sim,
                OmegaConf.to_object(cfg.obs),
                OmegaConf.to_object(cfg.reward),
                device=device,
                seed=771 + i,
            )
        )
    return result


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    ("layout", "blocks"),
    [
        pytest.param([("frontier", 4), ("warm", 2), ("cold", 2)], (2, 1, 1), id="three_profiles"),
        pytest.param(
            [("frontier", 192), ("transition", 32), ("warm", 16), ("cold", 16)],
            (48, 8, 4, 4),
            id="speed_bridge",
        ),
        pytest.param(
            [("frontier", 176), ("speed", 16), ("angle", 32), ("warm", 16), ("cold", 16)],
            (44, 4, 8, 4, 4),
            id="angle_with_speed_rehearsal",
        ),
    ],
)
def test_mixture_matches_standalone_through_autoresets_and_config_restore(device, layout, blocks):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    children, reference = profiles(device, layout), profiles(device, layout)
    mixed = MixtureRocketEnv(children, blocks)
    block_width = sum(blocks)
    for block in range(mixed.num_envs // block_width):
        start = block * block_width
        for indices, width in zip(mixed.indices, blocks, strict=True):
            within = indices[
                (indices >= block * block_width) & (indices < (block + 1) * block_width)
            ]
            assert within.cpu().tolist() == list(range(start, start + width))
            start += width
    tracker = EpisodeTracker(mixed)
    memory = RNN(32, hidden_dim=8).to(device)
    separate_memories = [deepcopy(memory) for _ in mixed.children]
    for separate in separate_memories:
        separate.rnn.flatten_parameters()
    seen = torch.zeros(mixed.num_envs, device=device, dtype=torch.bool)
    for t in range(30):
        actions = torch.zeros((mixed.num_envs, ACTION_DIM), device=device)
        actions[:, 3] = torch.arange(mixed.num_envs, device=device) / mixed.num_envs + t / 100
        with torch.no_grad():
            before_obs = mixed.get_observations()["policy"]
            output = memory(before_obs)
            for indices, separate in zip(mixed.indices, separate_memories, strict=True):
                torch.testing.assert_close(output[:, indices], separate(before_obs[indices]))
        obs, reward, done, extras = mixed.step(actions)
        previous = [x.clone() for x in memory.hidden_state]
        memory.reset(done)
        for old, current in zip(previous, memory.hidden_state, strict=True):
            assert torch.equal(old[:, ~done], current[:, ~done])
            assert not current[:, done].any()
        for indices, separate in zip(mixed.indices, separate_memories, strict=True):
            separate.reset(done[indices])
        tracker.step(done)
        seen |= done
        for indices, child in zip(mixed.indices, reference.values(), strict=True):
            expected = child.step(actions[indices])
            assert torch.equal(obs["policy"][indices], expected[0]["policy"])
            assert torch.equal(reward[indices], expected[1])
            assert torch.equal(done[indices], expected[2])
            for key in extras:
                assert torch.equal(extras[key][indices], expected[3][key])
            for name in ["final_obs", "final_status", "final_pos", "final_steps", "engine_on"]:
                assert torch.equal(getattr(mixed, name)[indices], getattr(child, name))
        assert not extras["time_outs"].any()
    assert seen.all()
    assert tracker.total_episodes > mixed.num_envs
    assert mixed.profile_metrics()["warm"]["transitions"] == children["warm"].num_envs * 30
    originals = [deepcopy(e.cfg) for e in children.values()]
    changed = deepcopy(originals[0])
    changed.spawn.altitude = 777
    mixed.set_configs(changed, children["frontier"].env.reward_config)
    assert children["frontier"].cfg.spawn.altitude == 777
    assert children["warm"].cfg == originals[-2]
    assert children["cold"].cfg == originals[-1]
    assert [child.cfg for child in list(children.values())[1:]] == originals[1:]
    mixed.set_configs(originals[0], children["frontier"].env.reward_config)
    mixed.reset()
    assert [e.cfg for e in children.values()] == originals
    invalid = deepcopy(originals[0])
    invalid.dt *= 2
    with pytest.raises(ValueError, match="non-spawn physics"):
        mixed.set_configs(invalid, children["frontier"].env.reward_config)


def test_mixture_rejects_uneven_recurrent_blocks():
    with pytest.raises(ValueError, match="same number"):
        MixtureRocketEnv(profiles("cpu"), (3, 1, 1))
