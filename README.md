# starship

GPU-parallel 2D rocket landing simulator built on [NVIDIA Warp](https://github.com/NVIDIA/warp), meant as a reinforcement learning environment.

A 60 t, 30 m × 9 m rocket spawns lying flat at 3 km, falling at ~70 m/s. It has three on/off thrusters: the main engine, plus left and right thrusters at the nose. It has to flip upright and land on the pad at the origin. The physics covers gravity, crossflow drag (belly-flop ≈ 71 m/s terminal velocity vs. ≈ 142 m/s nose-first), ISA air density, and rotational air damping. Thousands of envs step in parallel in one Warp kernel.

## Setup

With Nix and direnv, `direnv allow` once, and the dev shell (Python 3.13, uv, and on Linux the runtime libraries for the Warp, pygame and Rerun wheels) loads automatically and runs `uv sync`. Without direnv, use `nix develop`. Without Nix, just `uv sync`.

## Usage

```sh
uv run starship-demo                          # headless: 4096 envs, scripted controller, outcome stats
uv run starship-demo --controller random      # or: idle
uv run starship-view                          # fly it yourself (W/A/D or arrow keys)
uv run starship-view --controller heuristic --envs 64
uv run starship-rerun --spawn-noise 1         # stream 64 envs to Rerun: follow view, overview, telemetry
uv run starship-rerun --save out/run.rrd      # or record to a file, open with `uv run rerun out/run.rrd`
```

Warp runs on the CPU on macOS; pass `--device cuda:0` on a machine with an NVIDIA GPU.

## Layout

- `src/starship/sim/`: configuration, Warp kernels and the batched `Simulator`
- `src/starship/control/`: scripted baseline controllers
- `src/starship/render/`: pygame viewer and Rerun recording
- `src/starship/train/`: batched `RocketEnv` with auto-reset and device-side observation encoding; policy training upcoming

## Development

```sh
uv run ruff format . && uv run ruff check . && uv run pytest
```
