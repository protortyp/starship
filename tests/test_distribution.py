import io
import math

import pytest
import torch
from rsl_rl.models import RNNModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict
from torch.distributions import Bernoulli, Categorical, Normal, kl_divergence

from starship.train.actions import ACTION_DIM, CONTINUOUS_HEADS, ENGINE_HEADS, HEAD_NAMES
from starship.train.distribution import HybridDistribution
from starship.train.ppo import InstrumentedPPO


def make(seed=0, batch=(5,)):
    torch.manual_seed(seed)
    dist = HybridDistribution(ACTION_DIM)
    out = torch.randn(*batch, ACTION_DIM)
    dist.update(out)
    return dist, out


def reference(dist, out):
    logits, mean = out[..., :ENGINE_HEADS], out[..., ENGINE_HEADS:]
    std = dist.log_std.exp().expand_as(mean)
    return Bernoulli(logits=logits), Normal(mean, std)


def test_rejects_other_action_dims():
    with pytest.raises(ValueError, match="output_dim"):
        HybridDistribution(ACTION_DIM + 1)
    with pytest.raises(ValueError, match="deterministic_engine_selection"):
        HybridDistribution(ACTION_DIM, deterministic_engine_selection="random")
    with pytest.raises(ValueError, match="stochastic_engine_selection"):
        HybridDistribution(ACTION_DIM, stochastic_engine_selection="random")
    with pytest.raises(ValueError, match="ranked_count_temperature"):
        HybridDistribution(ACTION_DIM, ranked_count_temperature=0)
    with pytest.raises(ValueError, match="ranked_count_hold_steps"):
        HybridDistribution(ACTION_DIM, ranked_count_hold_steps=0)


@pytest.mark.parametrize("batch", [(5,), (3, 4)])
def test_shapes(batch):
    dist, out = make(batch=batch)
    assert dist.input_dim == ACTION_DIM
    sample = dist.sample()
    assert sample.shape == (*batch, ACTION_DIM)
    assert torch.isin(sample[..., :ENGINE_HEADS], torch.tensor([0.0, 1.0])).all()
    for t in (dist.mean, dist.std, dist.entropy_per_head):
        assert t.shape == (*batch, ACTION_DIM)
    assert dist.entropy.shape == batch
    assert dist.log_prob(sample).shape == batch
    logits, mean, std, credit = dist.params
    assert logits.shape == (*batch, ENGINE_HEADS)
    assert mean.shape == std.shape == (*batch, CONTINUOUS_HEADS)
    assert torch.equal(mean, out[..., ENGINE_HEADS:])
    assert torch.equal(dist.mean[..., :ENGINE_HEADS], torch.sigmoid(logits))
    assert torch.equal(dist.mean[..., ENGINE_HEADS:], mean)
    assert torch.equal(dist.std[..., ENGINE_HEADS:], std)
    assert torch.equal(credit, torch.ones(batch))


def test_initial_std_and_clamp():
    dist, _ = make()
    assert torch.allclose(dist.params[2], torch.full((5, CONTINUOUS_HEADS), 0.5))
    with torch.no_grad():
        dist.log_std[:4] = 10.0
        dist.log_std[4:] = -10.0
    dist.update(torch.zeros(1, ACTION_DIM))
    std = dist.params[2][0]
    assert torch.allclose(std[:4], torch.full((4,), 2.0))
    assert torch.allclose(std[4:], torch.full((CONTINUOUS_HEADS - 4,), 0.05))


def test_log_prob_matches_torch_reference():
    dist, out = make()
    bern, normal = reference(dist, out)
    actions = dist.sample()
    expected = bern.log_prob(actions[..., :ENGINE_HEADS]).sum(-1) + normal.log_prob(
        actions[..., ENGINE_HEADS:]
    ).sum(-1)
    assert torch.allclose(dist.log_prob(actions), expected, atol=1e-6)


def test_entropy_matches_torch_reference():
    dist, out = make()
    bern, normal = reference(dist, out)
    per_head = torch.cat([bern.entropy(), normal.entropy()], dim=-1)
    assert torch.allclose(dist.entropy_per_head, per_head, atol=1e-6)
    assert torch.allclose(dist.entropy, per_head.sum(-1), atol=1e-6)
    dist.update(torch.zeros(1, ACTION_DIM))
    gauss = 0.5 * math.log(2 * math.pi * math.e * 0.25)
    assert dist.entropy.item() == pytest.approx(
        ENGINE_HEADS * math.log(2) + CONTINUOUS_HEADS * gauss
    )


def test_gradients_reach_log_std():
    dist, _ = make()
    (dist.log_prob(dist.sample()).sum() + dist.entropy.sum()).backward()
    assert dist.log_std.grad is not None
    assert torch.isfinite(dist.log_std.grad).all()


def test_kl_zero_for_identical_params_and_positive_otherwise():
    dist, out = make()
    params = dist.params
    assert torch.allclose(dist.kl_divergence(params, params), torch.zeros(5), atol=1e-6)
    other = HybridDistribution(ACTION_DIM)
    with torch.no_grad():
        other.log_std.fill_(math.log(0.7))
    other.update(out + 0.3)
    kl = dist.kl_divergence(params, other.params)
    assert (kl > 0).all()
    bern, normal = reference(dist, out)
    bern2, normal2 = reference(other, out + 0.3)
    expected = kl_divergence(bern, bern2).sum(-1) + kl_divergence(normal, normal2).sum(-1)
    assert torch.allclose(kl, expected, atol=1e-5)


def test_kl_is_finite_for_saturated_logits():
    dist, _ = make()
    mean, std = torch.zeros(1, CONTINUOUS_HEADS), torch.full((1, CONTINUOUS_HEADS), 0.5)
    old = (torch.tensor([[40.0, -40.0, 0.0]]), mean, std)
    new = (torch.tensor([[-40.0, 40.0, 0.5]]), mean, std)
    kl = dist.kl_divergence(old, new)
    assert torch.isfinite(kl).all()
    assert kl.item() == pytest.approx(40.0 + 40.0 + 0.0310, rel=1e-3)


def test_deterministic_output_and_export_module():
    dist, out = make()
    expected = torch.cat([(out[..., :ENGINE_HEADS] > 0).float(), out[..., ENGINE_HEADS:]], -1)
    assert torch.equal(dist.deterministic_output(out), expected)
    module = dist.as_deterministic_output_module()
    assert torch.equal(module(out), expected)
    assert torch.equal(torch.export.export(module, (out,)).module()(out), expected)
    buffer = io.BytesIO()
    torch.save(module, buffer)
    buffer.seek(0)
    assert torch.equal(torch.load(buffer, weights_only=False)(out), expected)


def test_deterministic_floor_expected_engine_selection_and_export():
    dist = HybridDistribution(ACTION_DIM, deterministic_engine_selection="floor_expected")
    # Probabilities sum to 1.51: select exactly the highest-confidence engine. Thresholding would
    # select two, and rounding the expected count would also select two.
    logits = torch.tensor([[2.0, 0.04, -2.0]])
    out = torch.cat([logits, torch.zeros(1, CONTINUOUS_HEADS)], -1)
    expected = torch.tensor([[1.0, 0.0, 0.0]])
    assert torch.equal(dist.deterministic_output(out)[:, :ENGINE_HEADS], expected)
    module = dist.as_deterministic_output_module()
    assert torch.equal(module(out)[:, :ENGINE_HEADS], expected)
    assert torch.equal(
        torch.export.export(module, (out,)).module()(out)[:, :ENGINE_HEADS], expected
    )


def test_ranked_count_sampling_matches_deployment_structure():
    torch.manual_seed(2)
    dist = HybridDistribution(
        ACTION_DIM,
        deterministic_engine_selection="floor_expected",
        stochastic_engine_selection="ranked_count",
    )
    logits = torch.tensor([[2.0, 0.04, -2.0]]).expand(2000, -1)
    out = torch.cat([logits, torch.zeros(2000, CONTINUOUS_HEADS)], -1)
    dist.update(out)
    actions = dist.sample()
    # Ranked-count exploration only produces nested top-k subsets, never a lower-confidence
    # engine without every more-confident one also being enabled.
    on = actions[:, :ENGINE_HEADS].bool()
    assert (on[:, 1] <= on[:, 0]).all()
    assert (on[:, 2] <= on[:, 1]).all()
    assert torch.isfinite(dist.log_prob(actions)).all()
    assert dist.params[0].shape == (2000, ENGINE_HEADS + 1)
    assert dist.params[0][0].argmax() == 1
    assert actions[:, :ENGINE_HEADS].sum(-1).mode().values == 1
    assert dist.mean.shape == dist.std.shape == dist.entropy_per_head.shape == (2000, ACTION_DIM)


def test_ranked_count_log_prob_entropy_kl_and_gradients():
    torch.manual_seed(3)
    dist = HybridDistribution(ACTION_DIM, stochastic_engine_selection="ranked_count")
    out = torch.randn(7, ACTION_DIM, requires_grad=True)
    dist.update(out)
    actions = dist.sample()
    count = actions[:, :ENGINE_HEADS].sum(-1).long()
    categorical = Categorical(logits=dist.params[0])
    normal = Normal(dist.params[1], dist.params[2])
    expected = categorical.log_prob(count) + normal.log_prob(actions[:, ENGINE_HEADS:]).sum(-1)
    assert torch.allclose(dist.log_prob(actions), expected)
    assert torch.allclose(dist.entropy, categorical.entropy() + normal.entropy().sum(-1))
    same = dist.kl_divergence(dist.params, dist.params)
    assert torch.allclose(same, torch.zeros_like(same), atol=1e-6)
    dist.log_prob(actions).sum().backward()
    assert out.grad is not None
    assert torch.isfinite(out.grad).all()


def test_ranked_count_sampling_can_hold_counts_and_reset_finished_envs():
    dist = HybridDistribution(
        ACTION_DIM,
        stochastic_engine_selection="ranked_count",
        ranked_count_temperature=0.01,
        ranked_count_hold_steps=3,
    )

    def update(engine_logit):
        logits = torch.full((2, ENGINE_HEADS), engine_logit)
        dist.update(torch.cat([logits, torch.zeros(2, CONTINUOUS_HEADS)], -1))

    with torch.inference_mode():
        update(-10.0)
        assert torch.equal(dist.sample()[:, :ENGINE_HEADS].sum(-1), torch.tensor([0.0, 0.0]))
        update(1.0)
        # The new distribution overwhelmingly selects two engines, but the rollout action remains
        # at zero until its three-step hold expires.
        assert torch.equal(dist.sample()[:, :ENGINE_HEADS].sum(-1), torch.tensor([0.0, 0.0]))
        dist.reset(torch.tensor([True, False]))
        assert torch.equal(dist.sample()[:, :ENGINE_HEADS].sum(-1), torch.tensor([2.0, 0.0]))
        assert torch.equal(dist.sample()[:, :ENGINE_HEADS].sum(-1), torch.tensor([2.0, 2.0]))

    dist.reset()
    assert dist._held_engine_count is None
    assert dist._engine_count_steps_left is None


def test_ranked_count_can_use_deterministic_count_before_phase_handoff():
    dist = HybridDistribution(
        ACTION_DIM,
        deterministic_engine_selection="floor_expected",
        stochastic_engine_selection="ranked_count",
        ranked_count_temperature=10.0,
        ranked_count_hold_steps=1000,
    )
    logits = torch.tensor([[2.0, 0.04, -2.0]]).expand(128, -1)
    dist.update(torch.cat([logits, torch.zeros(128, CONTINUOUS_HEADS)], -1))
    dist.set_force_deterministic_engine_count(torch.ones(128, dtype=torch.bool))
    with torch.inference_mode():
        count = dist.sample()[:, :ENGINE_HEADS].sum(-1)
    assert torch.equal(count, torch.ones_like(count))


def test_engine_credit_applies_only_at_actual_count_decisions():
    dist = HybridDistribution(ACTION_DIM, stochastic_engine_selection="ranked_count")
    output = torch.randn(2, ACTION_DIM, requires_grad=True)
    dist.update(output)
    actions = dist.sample().detach()
    dist.set_engine_credit_mask(torch.tensor([False, True]))
    dist.log_prob(actions).sum().backward()
    assert torch.equal(output.grad[0, :ENGINE_HEADS], torch.zeros(ENGINE_HEADS))
    assert output.grad[1, :ENGINE_HEADS].abs().sum() > 0
    assert output.grad[:, ENGINE_HEADS:].abs().sum(dim=-1).gt(0).all()
    assert torch.equal(dist.params[3], torch.tensor([0.0, 1.0]))


def test_ppo_smoke_update_on_random_data():
    torch.manual_seed(0)
    num_envs, steps, obs_dim = 8, 6, 12
    obs = TensorDict({"policy": torch.randn(num_envs, obs_dim)}, batch_size=[num_envs])
    groups = {"actor": ["policy"], "critic": ["policy"]}
    model_cfg = {"hidden_dims": [16], "rnn_hidden_dim": 16, "obs_normalization": True}
    actor = RNNModel(
        obs,
        groups,
        "actor",
        ACTION_DIM,
        distribution_cfg={"class_name": "starship.train.distribution:HybridDistribution"},
        **model_cfg,
    )
    assert isinstance(actor.distribution, HybridDistribution)
    critic = RNNModel(obs, groups, "critic", 1, **model_cfg)
    storage = RolloutStorage("rl", num_envs, steps, obs, [ACTION_DIM])
    alg = InstrumentedPPO(
        actor, critic, storage, num_learning_epochs=2, num_mini_batches=2, learning_rate=1e-3
    )
    before = actor.distribution.log_std.detach().clone()
    # Rollouts run without autograd, as in rsl_rl's runner.
    with torch.inference_mode():
        for t in range(steps):
            actions = alg.act(obs)
            assert actions.shape == (num_envs, ACTION_DIM)
            obs = TensorDict({"policy": torch.randn(num_envs, obs_dim)}, batch_size=[num_envs])
            dones = (torch.rand(num_envs) < 0.2).float() if t > 0 else torch.zeros(num_envs)
            alg.process_env_step(obs, torch.randn(num_envs), dones, {"time_outs": dones * 0.0})
        alg.compute_returns(obs)
    metrics = alg.update()
    for key in ("loss/policy", "policy/kl_analytic", "policy/ess", "explore/entropy"):
        assert math.isfinite(metrics[key]), key
    for i, head in enumerate(HEAD_NAMES):
        assert math.isfinite(metrics[f"explore/entropy_head/{head}"])
        if i < ENGINE_HEADS:
            assert 0.0 < metrics[f"explore/on_prob/{head}"] < 1.0
        else:
            assert metrics[f"explore/std/{head}"] == pytest.approx(0.5, abs=0.05)
    assert not torch.equal(actor.distribution.log_std.detach(), before)
