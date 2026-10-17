# starship

![A trained policy flips the belly-flopping rocket upright and lands it on the pad](docs/landing.gif)

GPU-parallel 2D rocket landing simulator built on [NVIDIA Warp](https://github.com/NVIDIA/warp), with a PPO + LSTM agent that learns to land it.

A Starship-like upper stage spawns lying flat at 3 km, falling at ~90 m/s. It is 50 m long and 9 m wide, with 100 t dry mass, 30 t of landing propellant and one Raptor. It has three on/off thrusters (main engine plus left/right nose thrusters, all burning from the same tank) and has to flip upright and land on the pad at the origin. The physics covers gravity, mass that drops as fuel burns, crossflow drag (belly-flop ≈ 82 m/s terminal velocity vs. ≈ 160 m/s nose-first), ISA air density, and rotational air damping. The RL env adds an 8-feature observation and a potential-shaped reward (distance, speed above a safe descent profile, tilt, fuel and time costs, landing bonus). Thousands of envs step in parallel in one Warp kernel.

On one RTX 3090 the simulator alone steps ~1.5 billion envs per second (1M envs, 10 physics substeps each); full PPO training runs at ~300k env steps per second with 16k envs, where the LSTM policy and its updates take nearly all of the time.

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
uv run starship-view --policy logs/<run>/policy.pt   # watch a trained policy (also starship-rerun/-demo)
```

Warp runs on the CPU on macOS; pass `--device cuda:0` on a machine with an NVIDIA GPU.

## Training

```sh
uv run starship-train curriculum=scratch      # PPO + LSTM (rsl_rl), Hydra overrides e.g. num_envs=16384
just logs                                     # TensorBoard in a tmux window + browser
uv run starship-diagnose <experiment> --probe # run health: curriculum, plasticity, failure breakdown
uv run starship-bench                         # find the best num_envs for a config on this machine
```

The automatic curriculum starts near the ground and adds one change per stage (belly-flop start, time cost, mirroring, wider altitudes, speeds, orientations and spin), promoting at ≥ 90% landed in two consecutive evals. Plan, status and findings: [docs/curriculum.md](docs/curriculum.md). Runs write `logs/<experiment>/<time>/`: TensorBoard events, checkpoints, `policy_best.pt` and periodic Rerun recordings of evals.

GPU runs on [vast.ai](https://vast.ai) (API key in `.env` as `VAST_API_KEY`, see `.env.example`):

```sh
just vast up                                  # rent an RTX 3090, push the code, install
just vast train curriculum=scratch            # start training in a remote tmux session
just vast logs                                # TensorBoard through an SSH tunnel
just vast rerun --watch 20                    # open new eval recordings as they appear
just vast down                                # destroy the instance
```

Several experiments run side by side with `just vast --name <exp> <cmd>`; see `just vast --help`.

## Layout

- `src/starship/sim/`: configuration, Warp kernels and the batched `Simulator`
- `src/starship/control/`: scripted baseline controllers and the trained-policy controller
- `src/starship/render/`: pygame viewer and Rerun recording
- `src/starship/train/`: batched `RocketEnv`, observation encoding, reward, rsl_rl adapter, PPO with diagnostics, curriculum and Hydra configs
- `tools/`: vast.ai deployment (`vast.py`) and the README GIF (`readme_gif.py`)
- `docs/`: curriculum plan and results

## Contributing

```sh
just setup                # uv sync + install the pre-commit hooks (ruff check/format, file checks)
just check && just test   # just fmt to auto-fix
```

CI runs the same hooks and the tests on every pull request. Branch off `main`; pull requests are squash-merged with titles like `feat(scope): ...`, `fix(scope): ...` or `chore(scope): ...`.
