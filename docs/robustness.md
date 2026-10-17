# Robustness plan

Next steps after the solved landing curriculum, merged from two independent reviews. Each item is new curriculum stages appended after stage 4, one change per stage, with 20 % earlier-stage spawns mixed in and a fixed regression eval (≤ 2 pp landing loss) before widening a range.

| # | Item | Design | Status |
|---|---|---|---|
| 1 | Attitude metrics | Per-substep roll and world-horizontal rotation travel, peak rates, flip and reflip counts; in metrics and `starship-diagnose` | in progress |
| 2 | Attitude costs | L1 costs on roll and world-horizontal rotation (yaw about the vertical stays free for steering), air-relative sideslip cost, soft reflip cost; roll limit 90 → 60 → 45 °/s | in progress |
| 3 | Per-env vehicle parameters | Immutable per-episode struct with its own random stream; bit-identical to the fixed vehicle when all ranges are zero; privileged critic (true wind, parameters, faults via `obs_groups`), policy inputs unchanged | |
| 4 | Vehicle randomization | Dry mass 152–165 t, CoM ±0.5 m, inertia ±10 %, hull and flap scales ±10 %, thrust ±2 %, Isp ±0.5 %; IFT-5 fit as a second mixture centre (with `main_fill=0`), not a default | |
| 5 | Sensors | Observation noise, bias and 0–100 ms latency | |
| 6 | Wind | World-frame mean profile + Dryden turbulence (MIL-F-8785C), aero on air-relative velocity, no wind input to the policy; surface wind 0–5 → 15 m/s, light → moderate turbulence, severe held out | implemented (`sim.wind`, `starship.sim.wind`; stages 9–12 pending) |
| 7 | Engine faults | Delayed ignition → partial thrust → failed relight with retries → engine-out in the burn; spool input becomes spool × health | |
| 8 | 20 km start | 280 m/s at 64°; reference-glide spawns, 240 s, bounded distance potential, 28 observations and checkpoint migration; stages 9–13 | implemented; training pending |

Notes: training spawns carry 15 t of main-tank residuals (landing mass ≈ 198 t, one-engine T/W ≈ 1.16). Vortex shedding and flap force noise are implemented (default off) as the last wind stage. Sources: `docs/starship.md`, `docs/gaps.md`.

Follow-up (training speed): an iteration takes ~1.5 s, split evenly between collection and the PPO update, at ~65 % GPU utilization; collection is launch-overhead bound (~5.8 ms per step vs ~0.25 ms of physics). Capture the rollout step (policy forward, sampling, Warp env step) in a CUDA graph and enable TF32; try bf16 autocast and `torch.compile` only if a benchmark shows a gain.
