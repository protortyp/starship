# starship

GPU-parallel 2D rocket landing simulator built on [NVIDIA Warp](https://github.com/NVIDIA/warp), meant as a reinforcement learning environment.

A Starship-like upper stage spawns lying flat at 3 km, falling at ~90 m/s. It is 50 m long and 9 m wide, with 100 t dry mass, 30 t of landing propellant and one Raptor. It has three on/off thrusters (main engine plus left/right nose thrusters, all burning from the same tank) and has to flip upright and land on the pad at the origin. The physics covers gravity, mass that drops as fuel burns, crossflow drag (belly-flop ≈ 82 m/s terminal velocity vs. ≈ 160 m/s nose-first), ISA air density, and rotational air damping. The RL env adds an 8-feature observation and a stage 1 landing reward. Thousands of envs step in parallel in one Warp kernel.

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
uv run starship-train                         # PPO (rsl_rl), Hydra overrides e.g. num_envs=8192 iterations=1000
uv run tensorboard --logdir logs              # training curves
uv run starship-view --policy logs/<run>/policy.pt   # watch a trained policy (also starship-rerun/-demo)
```

Warp runs on the CPU on macOS; pass `--device cuda:0` on a machine with an NVIDIA GPU.

## Layout

- `src/starship/sim/`: configuration, Warp kernels and the batched `Simulator`
- `src/starship/control/`: scripted baseline controllers
- `src/starship/render/`: pygame viewer and Rerun recording
- `src/starship/train/`: batched `RocketEnv`, observation encoding, reward, rsl_rl adapter and Hydra training config

## Development

```sh
just check && just test   # just fmt to auto-fix
```
