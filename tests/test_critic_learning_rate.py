import pytest
import torch

from starship.train.ppo import InstrumentedPPO
from tests.test_train import SHORT_EPISODES, compose_cfg, make_runner


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_invalid_independent_critic_rate_is_rejected(rate):
    with pytest.raises(ValueError, match="critic_learning_rate"):
        InstrumentedPPO(critic_learning_rate=rate)


def test_independent_critic_can_learn_with_actor_rate_zero(tmp_path):
    cfg = compose_cfg(
        *SHORT_EPISODES,
        "runner.algorithm.learning_rate=0",
        "runner.algorithm.critic_learning_rate=0.001",
        "+runner.algorithm.freeze_observation_normalization=true",
    )
    runner = make_runner(cfg, tmp_path)
    actor = [p.detach().clone() for p in runner.alg.actor.parameters()]
    critic = [p.detach().clone() for p in runner.alg.critic.parameters()]
    runner.learn(1)
    assert all(torch.equal(a, b) for a, b in zip(actor, runner.alg.actor.parameters(), strict=True))
    assert any(
        not torch.equal(a, b) for a, b in zip(critic, runner.alg.critic.parameters(), strict=True)
    )
    assert [g["lr"] for g in runner.alg.optimizer.param_groups] == [0, 0.001]
    runner.alg.learning_rate = 0.0002
    runner.alg.apply_learning_rates()
    assert [g["lr"] for g in runner.alg.optimizer.param_groups] == [0.0002, 0.001]
    runner.writer.close()


def test_optimizer_moments_survive_shared_split_and_shared_roundtrip(tmp_path):
    cfg = compose_cfg(*SHORT_EPISODES)
    original = make_runner(cfg, tmp_path / "original")
    original.learn(1)
    cfg.runner.algorithm.learning_rate = 0.00001
    cfg.runner.algorithm.critic_learning_rate = 0.0001
    split = make_runner(cfg, tmp_path / "split")
    split.load(tmp_path / "original/model_0.pt")
    assert [g["lr"] for g in split.alg.optimizer.param_groups] == [0.00001, 0.0001]
    split.save(tmp_path / "split/model_split.pt")
    cfg.runner.algorithm.critic_learning_rate = None
    shared = make_runner(cfg, tmp_path / "shared")
    shared.load(tmp_path / "split/model_split.pt")
    assert len(shared.alg.optimizer.param_groups) == 1
    original_parameters = [p for g in original.alg.optimizer.param_groups for p in g["params"]]
    for runner in (split, shared):
        parameters = [p for g in runner.alg.optimizer.param_groups for p in g["params"]]
        for before, after in zip(original_parameters, parameters, strict=True):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            expected = original.alg.optimizer.state[before]
            actual = runner.alg.optimizer.state[after]
            assert expected.keys() == actual.keys()
            for key, value in expected.items():
                torch.testing.assert_close(value, actual[key], rtol=0, atol=0)
    for runner in (original, split, shared):
        runner.writer.close()
