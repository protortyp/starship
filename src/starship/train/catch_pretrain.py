"""Warm-start a catch policy from finite-actuator demonstrations, then validate policy alone.

uv run python -m starship.train.catch_pretrain --output logs/catch_demo_pretrain/<run>
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch import nn
from torch.nn import functional as F

import starship.train.conf  # noqa: F401
from starship.train.actions import ARM_HEAD, ENGINE_HEADS, LEGACY_ACTION_DIM
from starship.train.catch_teacher import teacher_actions
from starship.train.curriculum import Curriculum
from starship.train.env import RocketEnv
from starship.train.rsl_env import RslRocketEnv
from starship.train.runner import StarshipRunner, standalone_copy


def demonstrations(cfg, episodes, seed, policy=None, teacher_probability=1.0):
    curriculum = Curriculum(cfg, cfg.curriculum)
    sim, reward = curriculum.configs(0)
    env = RocketEnv(episodes, sim, OmegaConf.to_object(cfg.obs), reward, device="cpu", seed=seed)
    rng = np.random.default_rng(seed)
    if policy is not None:
        policy = standalone_copy(policy).cpu().eval()
        policy.reset()
    pending = np.ones(episodes, dtype=bool)
    status = np.zeros(episodes, dtype=np.int32)
    inputs, labels, rewards, masks = [], [], [], []
    for _ in range(env.sim.params.max_steps):
        action = teacher_actions(env)
        inputs.append(env.observe().numpy().copy())
        labels.append(action.numpy().copy())
        masks.append(pending.copy())
        executed = action
        if policy is not None:
            with torch.inference_mode():
                predicted = policy(
                    TensorDict({"policy": torch.from_numpy(inputs[-1])}, batch_size=[episodes]),
                    stochastic_output=False,
                ).numpy()
            choose_teacher = rng.random(episodes) < teacher_probability
            executed = np.where(choose_teacher[:, None], labels[-1], predicted)
        env.step(executed)
        rewards.append(env.reward.numpy().copy() * pending)
        done = env.sim.terminal.done.numpy().astype(bool) & pending
        if policy is not None:
            with torch.inference_mode():
                policy.reset(torch.from_numpy(env.sim.terminal.done.numpy().astype(bool)))
        status[done] = env.final_status.numpy()[done]
        pending[done] = False
        if not pending.any():
            break
    discounted = np.stack(rewards)
    for t in range(len(discounted) - 2, -1, -1):
        discounted[t] += reward.gamma * discounted[t + 1]
    return {
        "obs": torch.from_numpy(np.stack(inputs)),
        "actions": torch.from_numpy(np.stack(labels)),
        "valid": torch.from_numpy(np.stack(masks)),
        "returns": torch.from_numpy(discounted),
        "status": torch.from_numpy(status),
    }


def aggregate(original, added):
    """Pad time only; each column remains a complete, independently reset episode."""
    result = {}
    length = max(original["valid"].shape[0], added["valid"].shape[0])
    for key in original:
        if key == "status":
            result[key] = torch.cat((original[key], added[key]))
            continue
        arrays = []
        for data in (original, added):
            value = data[key]
            padded = value.new_zeros((length, *value.shape[1:]))
            padded[: value.shape[0]] = value
            arrays.append(padded)
        result[key] = torch.cat(arrays, dim=1)
    return result


def sequence_output(model, obs):
    shape = obs.shape
    normalized = model.obs_normalizer(obs.reshape(-1, shape[-1])).reshape(shape)
    features, _ = model.rnn.rnn(normalized)
    return model.mlp(features)


def imitation_loss(logits, targets, valid):
    binary = [0, 1, 2, ARM_HEAD]
    switches = F.binary_cross_entropy_with_logits(
        logits[..., binary], targets[..., binary], reduction="none"
    ).mean(-1)
    weights = logits.new_tensor([1, 20, 20, 0.1, 0.1, 0.1, 0.1, 1])
    continuous = (
        (
            logits[..., ENGINE_HEADS:LEGACY_ACTION_DIM].tanh()
            - targets[..., ENGINE_HEADS:LEGACY_ACTION_DIM].tanh()
        ).square()
        * weights
    ).mean(-1)
    return (switches + 10 * continuous)[valid].mean()


def make_runner(cfg, output, seed):
    sim, reward = Curriculum(cfg, cfg.curriculum).configs(0)
    obs = OmegaConf.to_object(cfg.obs)

    def env(n, rng):
        return RslRocketEnv(RocketEnv(n, sim, obs, reward, device=cfg.device, seed=rng))

    return StarshipRunner(
        env(cfg.num_envs, seed),
        OmegaConf.to_container(cfg.runner, resolve=True),
        output,
        cfg.device,
        eval_env=env(cfg.eval_envs, seed + 10000),
        curriculum=Curriculum(cfg, cfg.curriculum),
        verbose=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", default="catch_contact_local")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate-every", type=int, default=100)
    args = parser.parse_args()
    if min(args.episodes, args.updates, args.batch, args.evaluate_every) < 1:
        parser.error("counts must be positive")
    if args.rounds < 0:
        parser.error("rounds must be nonnegative")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    with initialize_config_module("starship.train.conf", version_base=None):
        cfg = compose(
            args.config,
            overrides=[
                f"device={args.device}",
                "num_envs=128",
                "eval_envs=128",
            ],
        )
    (args.output / ".hydra").mkdir()
    OmegaConf.save(cfg, args.output / ".hydra/config.yaml")
    data = demonstrations(cfg, args.episodes, args.seed)
    teacher_rate = float((data["status"] == 1).float().mean())
    print(f"demonstrations: {args.episodes} episodes, caught {teacher_rate:.1%}", flush=True)
    torch.save(data, args.output / "demonstrations.pt")
    runner = make_runner(cfg, args.output, args.seed)
    actor, critic = runner.alg.actor, runner.alg.critic
    valid = data["valid"]
    for model in (actor, critic):
        model.obs_normalizer.train()
        model.obs_normalizer.update(data["obs"][valid].to(args.device))
        model.train()  # cuDNN recurrent backward requires training mode.
        model.obs_normalizer.eval()
    with torch.no_grad():
        actor.distribution.log_std.fill_(math.log(0.05))
        output = [m for m in critic.mlp.modules() if isinstance(m, nn.Linear)][-1]
        output.bias.fill_(data["returns"][valid].mean().item())
    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
    data = {k: v.to(args.device) for k, v in data.items()}
    history = []
    best = (-1.0, -float("inf"))
    total_updates = args.updates * (args.rounds + 1)
    for update in range(1, total_updates + 1):
        if update > 1 and (update - 1) % args.updates == 0:
            round_index = (update - 1) // args.updates
            added = demonstrations(
                cfg,
                args.episodes,
                args.seed + round_index * 1000,
                policy=actor,
                teacher_probability=0.5**round_index,
            )
            torch.save(added, args.output / f"demonstrations-round-{round_index}.pt")
            data = aggregate(data, {k: v.to(args.device) for k, v in added.items()})
            print(f"aggregated round {round_index}: {data['valid'].shape[1]} episodes", flush=True)
        indices = torch.randint(data["valid"].shape[1], (args.batch,), device=args.device)
        obs = data["obs"][:, indices]
        mask = data["valid"][:, indices]
        actor_opt.zero_grad()
        loss = imitation_loss(sequence_output(actor, obs), data["actions"][:, indices], mask)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_opt.step()
        critic_opt.zero_grad()
        values = sequence_output(critic, obs).squeeze(-1)
        value_loss = ((values - data["returns"][:, indices])[mask] / 50).square().mean()
        value_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_opt.step()
        if update % args.evaluate_every == 0 or update == total_updates:
            metrics = runner.evaluate()
            rate, ret = metrics["eval/env/rate/landed"], metrics["eval/env/return"]
            row = {
                "update": update,
                "loss": loss.item(),
                "value_loss": value_loss.item(),
                "catch_rate": rate,
                "return": ret,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            if (rate, ret) > best:
                best = (rate, ret)
                # -1 is initialization, not a PPO iteration or a fabricated environment-step count.
                runner.current_learning_iteration = -1
                runner.save_all(-1)
                (args.output / "best-demo.json").write_text(json.dumps(row, indent=2))
            (args.output / "pretraining.json").write_text(
                json.dumps(
                    {
                        "teacher_rate": teacher_rate,
                        "episodes": args.episodes,
                        "aggregation_rounds": args.rounds,
                        "updates_per_round": args.updates,
                        "seed": args.seed,
                        "history": history,
                    },
                    indent=2,
                )
            )
    runner.writer.close()
    print(
        f"best standalone policy: {args.output / 'policy_-1.pt'} ({best[0]:.1%} catches)",
        flush=True,
    )


if __name__ == "__main__":
    main()
