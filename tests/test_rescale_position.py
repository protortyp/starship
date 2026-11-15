import pytest
import torch

from starship.train.catch_pretrain import sequence_output
from starship.train.rescale_position import rescale_model
from tests.test_train import compose_cfg, make_runner


def test_rescale_preserves_recurrent_actor_and_critic_with_zero_variance_channel(tmp_path):
    runner = make_runner(compose_cfg(), tmp_path)
    torch.manual_seed(814)
    raw = torch.randn(12, 4, 32) * 0.1
    raw[..., 6:9] *= 0.01
    scaled = raw.clone()
    scaled[..., 6:9] *= 100
    for model in (runner.alg.actor, runner.alg.critic):
        model.eval()
        with torch.no_grad():
            model.obs_normalizer._mean[..., 6:9] = torch.tensor([0.02, 0.001, -0.001])
            model.obs_normalizer._std[..., 6:9] = torch.tensor([0.003, 0.0, 0.0003])
            model.obs_normalizer._var.copy_(model.obs_normalizer._std.square())
            before = sequence_output(model, raw)
            recurrent = model.rnn.rnn.weight_hh_l0.clone()
            input_weights = model.rnn.rnn.weight_ih_l0.clone()
            count = model.obs_normalizer.count.clone()
            weight, ratio = rescale_model(model, 100)
            after = sequence_output(model, scaled)
        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ratio[0, 1], torch.tensor(0.01))
        assert torch.equal(weight[:, :6], input_weights[:, :6])
        assert torch.equal(weight[:, 9:], input_weights[:, 9:])
        assert torch.equal(model.rnn.rnn.weight_hh_l0, recurrent)
        assert torch.equal(model.obs_normalizer.count, count)
    runner.writer.close()


@pytest.mark.parametrize("multiplier", [0, -1, float("nan"), float("inf")])
def test_rescale_rejects_invalid_multiplier(multiplier):
    with pytest.raises(ValueError, match="finite and positive"):
        rescale_model(None, multiplier)
