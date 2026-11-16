"""Fixed-profile rollouts interleaved across contiguous recurrent minibatches."""

from copy import deepcopy

import torch
from tensordict import TensorDict

from starship.train.rsl_env import RslRocketEnv


class MixtureRocketEnv:
    """Keep profile membership fixed across autoresets and runner restorations.

    The first profile follows curriculum configs; other profiles retain their
    original configs. Each contiguous block contains the same profile counts.
    This adapter deliberately supports shared physics, rewards and landing rules;
    only spawn conditions and episode deadlines may differ.
    """

    def __init__(self, profiles: dict[str, RslRocketEnv], counts_per_block: tuple[int, ...]):
        self.names = tuple(profiles)
        self.children = tuple(profiles.values())
        if not self.children or len(self.children) != len(counts_per_block):
            raise ValueError("one positive block count is required per profile")
        if any(n <= 0 for n in counts_per_block):
            raise ValueError("block counts must be positive")
        blocks = [e.num_envs // n for e, n in zip(self.children, counts_per_block, strict=True)]
        if len(set(blocks)) != 1 or any(
            e.num_envs % n for e, n in zip(self.children, counts_per_block, strict=True)
        ):
            raise ValueError("profile sizes must contain the same number of complete blocks")
        self.num_envs = sum(e.num_envs for e in self.children)
        self.device = self.children[0].device
        self.num_actions = self.children[0].num_actions
        self._configs = tuple(
            (deepcopy(e.cfg), deepcopy(e.env.reward_config)) for e in self.children
        )
        reference = self._shared_config(self.children[0].cfg)
        for child in self.children:
            if child.device != self.device or self._shared_config(child.cfg) != reference:
                raise ValueError("mixture profiles must share device and non-spawn physics")
            if child.env.reward_config != self.children[0].env.reward_config:
                raise ValueError("mixture profiles must share reward rules")
        block_size = sum(counts_per_block)
        offsets = [sum(counts_per_block[:i]) for i in range(len(self.children))]
        self.indices = tuple(
            torch.tensor(
                [b * block_size + offset + j for b in range(blocks[0]) for j in range(n)],
                device=self.device,
            )
            for offset, n in zip(offsets, counts_per_block, strict=True)
        )
        self._gather = torch.argsort(torch.cat(self.indices))
        self.env = self  # runner reset and metric configuration interface
        self.profile_steps = [0] * len(self.children)
        self.profile_completed = [0] * len(self.children)
        self.profile_landed = [0] * len(self.children)

    @staticmethod
    def _shared_config(config):
        from dataclasses import asdict

        result = asdict(config)
        result.pop("spawn")
        result.pop("max_time")
        return result

    @property
    def cfg(self):
        return self.children[0].cfg

    @property
    def config(self):
        return self.cfg

    @property
    def sim(self):
        return self.children[0].env.sim

    @property
    def max_episode_length(self):
        return max(child.max_episode_length for child in self.children)

    def _join(self, tensors):
        return torch.cat(tensors, dim=0).index_select(0, self._gather)

    def __getattr__(self, name):
        children = self.__dict__.get("children")
        if children and isinstance(getattr(children[0], name, None), torch.Tensor):
            return self._join([getattr(child, name) for child in children])
        raise AttributeError(name)

    def get_observations(self):
        return TensorDict(
            {"policy": self._join([e.get_observations()["policy"] for e in self.children])},
            batch_size=[self.num_envs],
        )

    def set_configs(self, sim_config, reward_config):
        if self._shared_config(sim_config) != self._shared_config(self._configs[0][0]):
            raise ValueError("mixture cannot change non-spawn physics during training")
        if reward_config != self._configs[0][1]:
            raise ValueError("mixture cannot change reward rules during training")
        self.children[0].set_configs(sim_config, reward_config)
        for child, (sim, reward) in zip(self.children[1:], self._configs[1:], strict=True):
            child.set_configs(deepcopy(sim), deepcopy(reward))

    def reset(self):
        for child in self.children:
            child.env.reset()

    def step(self, actions):
        rows = []
        for i, (child, indices) in enumerate(zip(self.children, self.indices, strict=True)):
            row = child.step(actions.index_select(0, indices))
            rows.append(row)
            done = row[2].bool()
            self.profile_steps[i] += child.num_envs
            self.profile_completed[i] += int(done.sum())
            self.profile_landed[i] += int((done & (child.final_status == 1)).sum())
        return (
            TensorDict(
                {"policy": self._join([row[0]["policy"] for row in rows])},
                batch_size=[self.num_envs],
            ),
            self._join([row[1] for row in rows]),
            self._join([row[2] for row in rows]),
            {key: self._join([row[3][key] for row in rows]) for key in rows[0][3]},
        )

    def profile_metrics(self):
        return {
            name: {"transitions": steps, "completed": completed, "landed": landed}
            for name, steps, completed, landed in zip(
                self.names,
                self.profile_steps,
                self.profile_completed,
                self.profile_landed,
                strict=True,
            )
        }
