import torch
from hydra import compose, initialize_config_module

from starship.train.catch_pretrain import demonstrations, imitation_loss


def test_demonstrations_catch_through_normal_actions_and_mask_reset_episodes():
    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose("catch_contact_local")
    data = demonstrations(cfg, 8, 9241)
    assert (data["status"] == 1).all()
    valid = data["valid"]
    assert valid[0].all()
    assert not ((~valid[:-1]) & valid[1:]).any()
    assert (~valid).any()  # randomized approaches finish at different times
    assert (data["returns"][~valid] == 0).all()
    actions = data["actions"]
    assert torch.isfinite(actions).all()
    assert (actions[..., 0][valid] == 1).all()
    assert (actions[..., 1:3][valid] == 0).all()
    assert (actions[0, :, 11] == 0).all()
    lengths = valid.sum(0)
    assert (actions[lengths - 1, torch.arange(8), 11] == 1).all()


def test_imitation_loss_trains_arm_and_flight_but_ignores_padding():
    logits = torch.zeros(3, 2, 12, requires_grad=True)
    targets = torch.ones_like(logits)
    valid = torch.tensor([[True, True], [True, False], [False, False]])
    loss = imitation_loss(logits, targets, valid)
    loss.backward()
    assert (logits.grad[valid] < 0).all()
    assert (logits.grad[~valid] == 0).all()
    changed = targets.detach().clone()
    changed[~valid] = -100
    torch.testing.assert_close(loss, imitation_loss(logits, changed, valid))


def test_aggregation_pads_time_without_joining_episode_histories():
    from starship.train.catch_pretrain import aggregate

    def data(length, count):
        return {
            "obs": torch.ones(length, count, 32),
            "valid": torch.ones(length, count, dtype=torch.bool),
            "status": torch.ones(count, dtype=torch.int),
        }

    result = aggregate(data(2, 3), data(4, 2))
    assert result["obs"].shape == (4, 5, 32)
    assert result["status"].shape == (5,)
    assert result["valid"][:2].all()
    assert not result["valid"][2:, :3].any()
    assert result["valid"][:, 3:].all()
    assert (result["obs"][2:, :3] == 0).all()


def test_mixed_collection_resets_recurrent_policy_and_checkpoint_resumes_at_zero(tmp_path):
    from starship.train.catch_pretrain import make_runner

    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose("catch_contact_local", overrides=["device=cpu", "num_envs=8", "eval_envs=8"])
    runner = make_runner(cfg, tmp_path, 0)
    data = demonstrations(cfg, 4, 51, policy=runner.alg.actor, teacher_probability=0.5)
    assert (data["status"] != 0).all()
    assert torch.isfinite(data["returns"]).all()
    runner.current_learning_iteration = -1
    runner.save_all(-1)
    runner.load(tmp_path / "model_-1.pt")
    assert runner.next_iteration == 0
    assert runner.total_env_steps == 0
    runner.writer.close()
