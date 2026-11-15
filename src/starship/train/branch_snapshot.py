"""Exact diagnostic restore at a control boundary, including simulator RNG and policy memory.

Use only with evaluation-mode models in a dedicated diagnostic process. Restoring also
restores process-global Python, NumPy and Torch RNGs. Unlike Simulator.copy_from, this
preserves random streams. Models, configurations, device and batch size must match.
Snapshots are trusted local Torch artifacts, not an interchange format for untrusted input.
"""

import copy
import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import warp as wp

from starship.sim.physics import Commands, State, Terminal


def _buffers(env):
    rocket = env.env
    buffers = {f"env.{k}": v for k, v in vars(rocket).items() if isinstance(v, wp.array)}
    for label, struct in (("state", State), ("terminal", Terminal), ("commands", Commands)):
        instance = getattr(rocket.sim, label)
        buffers.update({f"sim.{label}.{k}": getattr(instance, k) for k in struct.vars})
    return buffers


def _config(env):
    rocket = env.env
    return {
        "sim": asdict(rocket.config),
        "obs": asdict(rocket.obs_config),
        "reward": asdict(rocket.reward_config),
        "auto_reset": rocket.sim.auto_reset,
    }


def _memory(value, device="cpu"):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device).clone()
    if isinstance(value, tuple):
        return tuple(_memory(v, device) for v in value)
    if isinstance(value, list):
        return [_memory(v, device) for v in value]
    if value is not None:
        raise TypeError(f"Unsupported recurrent memory: {type(value)}")
    return None


def _fingerprint(model):
    if model.training:
        raise ValueError("Branch snapshots require evaluation-mode models")
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(repr((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(
            value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def _sync(env):
    if env.env.device.is_cuda:
        torch.cuda.synchronize(env.device)
    wp.synchronize_device(env.env.device)


@dataclass
class BranchSnapshot:
    version: int
    device: str
    num_envs: int
    config: dict
    arrays: dict
    models: dict
    python_rng: tuple
    numpy_rng: tuple
    torch_rng: torch.Tensor
    cuda_rng: torch.Tensor | None
    provenance: dict

    @classmethod
    def capture(cls, env, actor, critic=None, *, provenance=None):
        """Capture after done-state resets, immediately before the next policy forward."""
        _sync(env)
        models = {}
        for name, model in (("actor", actor), ("critic", critic)):
            if model is not None:
                models[name] = {
                    "fingerprint": _fingerprint(model),
                    "device": str(next(model.parameters()).device),
                    "hidden": _memory(getattr(getattr(model, "rnn", None), "hidden_state", None)),
                }
        return cls(
            version=1,
            device=env.device,
            num_envs=env.num_envs,
            config=copy.deepcopy(_config(env)),
            arrays={name: array.numpy().copy() for name, array in _buffers(env).items()},
            models=models,
            python_rng=random.getstate(),
            numpy_rng=copy.deepcopy(np.random.get_state()),  # noqa: NPY002 - capture global RNG
            torch_rng=torch.get_rng_state().clone(),
            cuda_rng=torch.cuda.get_rng_state(env.device) if env.env.device.is_cuda else None,
            provenance=copy.deepcopy(provenance or {}),
        )

    def restore(self, env, actor, critic=None):
        """Restore into the same model/configuration and batch shape; never reset episodes."""
        if self.version != 1 or self.device != env.device or self.num_envs != env.num_envs:
            raise ValueError("Snapshot version, device or batch size differs")
        if self.config != _config(env):
            raise ValueError("Snapshot simulator/observation/reward configuration differs")
        models = {name: m for name, m in (("actor", actor), ("critic", critic)) if m is not None}
        if models.keys() != self.models.keys():
            raise ValueError("Snapshot actor/critic selection differs")
        for name, model in models.items():
            saved = self.models[name]
            if saved["fingerprint"] != _fingerprint(model):
                raise ValueError(f"Snapshot {name} parameters or normalizer differ")
            if saved["device"] != str(next(model.parameters()).device):
                raise ValueError(f"Snapshot {name} device differs")
        _sync(env)
        buffers = _buffers(env)
        if buffers.keys() != self.arrays.keys():
            raise ValueError("Snapshot buffer schema differs")
        for name, target in buffers.items():
            current, saved = target.numpy(), self.arrays[name]
            if current.shape != saved.shape or current.dtype != saved.dtype:
                raise ValueError(f"Snapshot buffer layout differs: {name}")
        for name, target in buffers.items():
            target.assign(self.arrays[name])
        for name, model in models.items():
            if getattr(model, "rnn", None) is not None:
                model.rnn.hidden_state = _memory(
                    self.models[name]["hidden"], self.models[name]["device"]
                )
            if getattr(model, "distribution", None) is not None:
                model.distribution._distribution = None
        _sync(env)
        random.setstate(self.python_rng)
        np.random.set_state(self.numpy_rng)  # noqa: NPY002 - restore captured global RNG
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state(self.cuda_rng, env.device)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self, path)

    @classmethod
    def load(cls, path):
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(snapshot, cls) or snapshot.version != 1:
            raise ValueError("Unsupported branch snapshot")
        return snapshot
