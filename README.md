# starship

GPU-parallel 6-DOF simulator of a SpaceX Starship upper stage landing, built on [NVIDIA Warp](https://github.com/NVIDIA/warp), with a PPO + LSTM agent that learns to fly it.

The ship (V2 geometry, three sea-level Raptor 2 engines) falls belly-first and has to flip upright and land on a pad. Modelled: the U.S. Standard Atmosphere 1976; full rigid-body dynamics; liquid-oxygen and methane main and header tanks, whose draining moves the centre of mass and inertia; Raptor thrust that depends on ambient pressure, with spool-up, throttle and ±15° gimbal; four flaps with actuator lag; strip-wise crossflow and slender-body hull aerodynamics plus flat-plate flaps, with modified-Newtonian hull and flap loads up to Mach 25; touchdown against the exact hull shape. Parameters and sources: [docs/starship.md](docs/starship.md); what is left out: [docs/gaps.md](docs/gaps.md).

On one RTX 3090 the simulator alone runs ~150M control steps per second (1.5B physics steps; 1M envs). The trained policy lands ~98 % of fully randomized descents (1.5–5 km, random speed, offset, pitch and spin) on the pad: belly-flop, flip, landing burn and touchdown.

## Setup

With Nix and direnv, `direnv allow` once, and the dev shell (Python 3.13, uv, and on Linux the runtime libraries for the Warp and Rerun wheels) loads automatically and runs `uv sync`. Without direnv, use `nix develop`. Without Nix, just `uv sync`.

## Usage

```sh
uv run starship-rerun                          # passive belly-flops streamed to Rerun (chase cam, telemetry)
uv run starship-rerun --save out/run.rrd       # or record to a file, open with `uv run rerun out/run.rrd`
uv run python -m starship.plan.cem --device cuda:0   # Monte Carlo search for a landing trajectory
uv run starship-rerun --schedule out/cem_best.npz    # replay it
uv run python -m starship.sim.bench --device cuda:0  # raw simulator throughput
uv run python -m starship.sim.calibrate --flight IFT-5 --device cpu --data-dir telemetry
```

Warp runs on the CPU on macOS; pass `--device cuda:0` on a machine with an NVIDIA GPU.

The calibration harness fits held-descent aerodynamics to webcast altitude/speed, then validates
the same parameters on the other flights. Data setup, assumptions and results: [docs/calibration.md](docs/calibration.md).

## Training

```sh
uv run starship-train                         # PPO + LSTM (rsl_rl) with the landing curriculum; Hydra overrides
just logs                                     # TensorBoard in a tmux window + browser
uv run starship-diagnose <experiment> --probe # run health: curriculum, plasticity, failure breakdown
uv run starship-bench                         # find the best num_envs for a config on this machine
```

The automatic curriculum starts with the landing burn alone and adds one change per stage (engines off, low belly-flop, 3 km belly-flop, randomized spawns), promoting at ≥ 90% landed in two consecutive evals. Episodes that tumble end with a large penalty. Plan and status: [docs/curriculum.md](docs/curriculum.md). Runs write `logs/<experiment>/<time>/`: TensorBoard events, checkpoints, `policy_best.pt` and periodic Rerun recordings of evals.

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

- `src/starship/sim/`: configuration, atmosphere, mass, propulsion, aerodynamics, rigid-body integration and the batched `Simulator`
- `src/starship/plan/`: cross-entropy trajectory search on the batched simulator
- `src/starship/render/`: Rerun viewer and eval recordings
- `src/starship/train/`: `RocketEnv`, observation encoding, action mapping, reward, rsl_rl adapter, hybrid policy head, PPO with diagnostics, curriculum and Hydra configs
- `tools/`: vast.ai deployment (`vast.py`)
- `docs/`: vehicle reference values, simulation gaps, curriculum plan

## Contributing

```sh
just setup                # uv sync + install the pre-commit hooks (ruff check/format, file checks)
just check && just test   # just fmt to auto-fix
```

CI runs the same hooks and the tests on every pull request. Branch off `main`; pull requests are squash-merged with titles like `feat(scope): ...`, `fix(scope): ...` or `chore(scope): ...`.
