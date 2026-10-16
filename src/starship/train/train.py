"""Train a landing policy with rsl_rl PPO. Configured with Hydra, e.g.

uv run starship-train iterations=200 num_envs=8192 runner.algorithm.learning_rate=3e-4
"""

from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from rsl_rl.runners import OnPolicyRunner

import starship.train.conf  # noqa: F401  (registers the config schema)
from starship.train.env import RocketEnv
from starship.train.rsl_env import RslRocketEnv


def export_policy(runner: OnPolicyRunner, out_dir: str | Path) -> Path:
    """Save the deterministic policy (normalizer, MLP, threshold) as a pickled module."""
    path = Path(out_dir) / "policy.pt"
    torch.save(runner.alg.get_policy().as_jit().cpu().eval(), path)
    return path


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    out_dir = HydraConfig.get().runtime.output_dir
    torch.manual_seed(cfg.seed)
    env = RocketEnv(
        cfg.num_envs,
        OmegaConf.to_object(cfg.sim),
        OmegaConf.to_object(cfg.obs),
        OmegaConf.to_object(cfg.reward),
        device=cfg.device,
        seed=cfg.seed,
    )
    vec_env = RslRocketEnv(env)
    runner_cfg = OmegaConf.to_container(cfg.runner, resolve=True)
    runner = OnPolicyRunner(vec_env, runner_cfg, log_dir=out_dir, device=str(vec_env.device))
    runner.learn(cfg.iterations)
    path = export_policy(runner, out_dir)
    print(f"policy exported to {path}")


if __name__ == "__main__":
    main()
