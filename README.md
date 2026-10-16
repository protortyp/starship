# starship

GPU-parallel 2D rocket landing simulator built on [NVIDIA Warp](https://github.com/NVIDIA/warp), meant as a reinforcement learning environment.

A 60 t, 30 m × 9 m rocket spawns lying flat at 3 km, falling at ~70 m/s. It has three on/off thrusters: the main engine, plus left and right thrusters at the nose. It has to flip upright and land on the pad at the origin. The physics covers gravity, crossflow drag (belly-flop ≈ 71 m/s terminal velocity vs. ≈ 142 m/s nose-first), ISA air density, and rotational air damping. Thousands of envs step in parallel in one Warp kernel.

## Usage

```sh
uv sync
uv run starship-demo                          # headless: 4096 envs, scripted controller, outcome stats
uv run starship-demo --controller random      # or: idle
uv run starship-view                          # fly it yourself (W/A/D or arrow keys)
uv run starship-view --controller heuristic --envs 64
```

Warp runs on the CPU on macOS; pass `--device cuda:0` on a machine with an NVIDIA GPU.

## Layout

- `src/starship/sim/`: configuration, Warp kernels and the batched `Simulator`
- `src/starship/control/`: scripted baseline controllers
- `src/starship/render/`: pygame viewer
- `src/starship/train/`: RL training (upcoming)

## Development

```sh
uv run ruff format . && uv run ruff check . && uv run pytest
```
