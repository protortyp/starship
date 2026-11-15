import torch
from hydra import compose, initialize_config_module

from starship.train.catch_pretrain import make_runner
from starship.train.critic_calibrate import fit


def test_critic_fit_preserves_actor_and_normalizers_and_resets_only_critic_moments(tmp_path):
    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose("catch_demo_local", overrides=["device=cpu", "num_envs=8", "eval_envs=8"])
    runner = make_runner(cfg, tmp_path, 0)
    actor_before = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    normalizer_before = {
        k: v.clone() for k, v in runner.alg.critic.obs_normalizer.state_dict().items()
    }
    actor_param = next(runner.alg.actor.parameters())
    critic_param = next(runner.alg.critic.parameters())
    for parameter in (actor_param, critic_param):
        runner.alg.optimizer.state[parameter] = {
            "step": torch.tensor(1.0),
            "exp_avg": torch.ones_like(parameter),
            "exp_avg_sq": torch.ones_like(parameter),
        }
    obs = runner.env.get_observations()["policy"].repeat(2, 1, 1)
    valid = torch.ones(2, 8, dtype=torch.bool)
    valid[1, 0] = False
    targets = torch.full((2, 8), 10.0)
    targets[1, 0] = 99999.0
    result = fit(runner, {"obs": obs, "returns": targets, "valid": valid}, tmp_path, updates=2)
    assert set(result["train_episode_indices"]).isdisjoint(result["validation_episode_indices"])
    assert len(result["train_episode_indices"]) + len(result["validation_episode_indices"]) == 8
    assert abs(result["bias_correction"]) < 1000
    assert all(torch.equal(v, actor_before[k]) for k, v in runner.alg.actor.state_dict().items())
    assert all(
        torch.equal(v, normalizer_before[k])
        for k, v in runner.alg.critic.obs_normalizer.state_dict().items()
    )
    assert actor_param in runner.alg.optimizer.state
    assert critic_param not in runner.alg.optimizer.state
    saved = torch.load(tmp_path / "model_calibrated.pt", weights_only=False)
    assert saved["iter"] == runner.current_learning_iteration
    runner.writer.close()
