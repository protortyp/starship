"""Train an LSTM landing policy with rsl_rl PPO. Configured with Hydra, e.g.

uv run starship-train iterations=200 num_envs=8192 runner.algorithm.learning_rate=3e-4
"""

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import starship.train.conf  # noqa: F401  (registers the config schema)
from starship.train.env import RocketEnv
from starship.train.rsl_env import RslRocketEnv
from starship.train.runner import StarshipRunner


def make_env(cfg: DictConfig, num_envs: int, seed: int, for_eval: bool = False) -> RslRocketEnv:
    sim = OmegaConf.to_object(cfg.sim)
    if for_eval:
        sim.easy_spawn_fraction = 0.0
    env = RocketEnv(
        num_envs,
        sim,
        OmegaConf.to_object(cfg.obs),
        OmegaConf.to_object(cfg.reward),
        device=cfg.device,
        seed=seed,
    )
    return RslRocketEnv(env)


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    out_dir = HydraConfig.get().runtime.output_dir
    torch.manual_seed(cfg.seed)
    env = make_env(cfg, cfg.num_envs, cfg.seed)
    eval_env = (
        make_env(cfg, cfg.eval_envs, cfg.seed + 1, for_eval=True) if cfg.eval_envs > 0 else None
    )
    runner = StarshipRunner(
        env,
        OmegaConf.to_container(cfg.runner, resolve=True),
        log_dir=out_dir,
        device=str(env.device),
        eval_env=eval_env,
        eval_interval=cfg.eval_interval,
    )
    if cfg.checkpoint:
        runner.load(cfg.checkpoint)
        print(f"continuing from {cfg.checkpoint}")
    runner.learn(cfg.iterations)
    print(f"policy exported to {out_dir}/policy.pt")


if __name__ == "__main__":
    main()
