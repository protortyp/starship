# Curriculum

6-DOF sim (`starship.sim`, values in `docs/starship.md`). One change per stage; promotion at ≥ 90 % eval landings in 2 consecutive evals.

## Environment

- **Actions**: 3 Bernoulli engine-on heads; 8 continuous (shared throttle, shared 2-axis gimbal, 4 flaps, roll RCS) as absolute targets; actuator lag and rate limits come from the sim.
- **Observations** (28, heading-invariant): original 24 body/actuator/fuel channels; altitude capped at 5 km and pad-vector norm at 5.1 km. Append `log1p(h/1000)/3`, Mach, `q/5000 Pa`, `log1p(horizontal miss/1000)/3`; Mach/q use still-air velocity. Port/starboard mirror as augmentation. Old actor/critic checkpoints and standalone policies load with zero new input weights, 0/1 normalizer padding and zero new Adam moments; legacy statistics stay intact.
- **Reward**: γΦ(s′) − Φ(s), zero terminal potential; distance, excess speed and near-ground tilt shaping, motion/fuel/time costs, landing bonus and limit-excess crash penalty. Stages 0–8 unchanged; stages 5+ add attitude costs. Stage 9+ uses `−0.05 d/(1+d/1000)` (−47.62/−25/0 at 20 km/1 km/0 distance); default `distance_saturation=0` retains the linear potential.
- **Entry** (stage 10+): speed/angle below horizontal, heading ±2° towards the pad, belly normal along velocity (90° AoA; `pitch` offsets that attitude). Uptrack placement integrates a no-thrust point mass with 280 m² drag area, actual mass and local density to the 300 m flip corridor; offset noise is cross-track error. This reference estimates reachability, not a demonstrated landing. Deadline 240 s / 2,400 control steps; promotion resets episodes and recurrent state together.
- **Termination**: landed, crashed, out of bounds, timeout, **tumbled** (pitch > 115° from vertical, pitch/yaw rate > 40°/s, or roll rate > 90°/s, tightened to 60°/s at stage 7 and 45°/s at stage 8) with the largest penalty.
- **Metrics**: landing rate; touchdown speed, tilt, miss; crash breakdown by limit, tumble breakdown by cause, end altitude of tumbles, out-of-bounds and timeouts; peak q, Mach and load, flip and ignition altitude; landing rate under calm vs windy conditions; reward per term; per-actuator usage and saturation; plasticity and per-promotion landing dip and critic-loss spike (`starship-diagnose`).

## Stages

| # | change | status |
|---|---|---|
| 0 | upright near the ground, engines lit (landing burn only) | passed (land3d_v6, it 1574) |
| 1 | engines off at spawn (light them in time, 2 s spool) | passed (land3d_v6, it 2099; 4 collapses recovered by rollback) |
| 2 | low-altitude belly-flop (flip + burn) | passed (land3d_v7_kl_long, it ~4100) |
| 3 | 3 km belly-flop | passed (land3d_v8_lam098, it ~4775) |
| 4 | altitude, speed, offset, pitch, spin ranges | solved (land3d_v8_lam098, it ~5400: 98 % eval, 98–99 % full scenario) |
| 5 | attitude costs at half weight; stage-4 spawns | passed (attitude_v1, it ~6725) |
| 6 | attitude costs at full weight | passed (attitude_v1, it ~6800) |
| 7 | roll limit 60°/s | passed (attitude_v1, it ~7050) |
| 8 | roll limit 45°/s | training (attitude_v1: ~97 % eval, roll travel 389° → 188°) |
| 9 | light wind: surface 0–5 m/s, light Dryden turbulence | |
| 10 | moderate wind: 0–10 m/s, moderate turbulence, 6 m/s gusts | |
| 11 | strong wind: 0–15 m/s, 9 m/s gusts | |
| 12 | vortex shedding and flap force noise | |
| 13 | bounded distance potential, same spawns | |
| 14 | 5 km forward handover, 110 ±5 m/s at 65 ±6° (34–59 m/s horizontal) | |
| 15 | 8 km handover, 200 ±10 m/s at 60 ±5° | |
| 16 | 12 km handover | |
| 17 | 20 km handover, 280 ±10 m/s at 64 ±3° (IFT-5 target) | |

## Single-salto landing

The `salto_land` curriculum first learns one powered end-over-end rotation and recovery, then keeps
the same recurrent policy alive while tightening touchdown speed, pad radius, attitude and angular
rate. It uses 53 stages and a 2,000-update default budget; promotion requires at least 45% of 256
deterministic evaluation episodes twice in succession. The final stage additionally requires exactly
one inverted-to-upright recovery with no re-flip and penalizes axial roll and rotation beyond 360°.

Current run: `salto_braking_polish_v2`. It resumed the learned loop-and-ground controller at stage 16,
passed stages 16–24, and reached a 41.4% best at the original 85 m/s stage after 2,000 continuation
updates (1.136B environment steps). A fixed-seed 1,024-episode probe remained flat at 35.4% from the
35.6% measured at update 1,500; failed contacts averaged 86.1 m/s and 79% broke the vertical-speed
limit. The next continuation therefore uses 0.625 m/s braking steps from 86.25 to 80 m/s. This is not
yet a safe-pad result.

## Findings

- The first telemetry calibration (IFT-5, 2026-10-17) fits speed to 4.16 m/s RMSE but does not complete IFT-6’s held-descent window. Mass, drag and flap effectiveness remain coupled; keep the training defaults neutral. Details and cross-flight errors: [calibration.md](calibration.md).

- Crash penalties must stay bounded but keep ranking crashes (log of the weighted limit excess); linear ones swamped the value targets (−1,300), hard caps made every bad crash equal.
- Terminal ordering matters: with timeouts cheaper than crashes, policies climbed away from the ground.
- Tumble checks must exclude roll: roll inertia is ~18× smaller, random flaps spun it past 40°/s.
- Without horizontal-speed shaping and an upright term over the whole spawn range, policies tipped over and flew off sideways; with both, landings appeared at ~250 it and stage 0 passed (a 97 % peak, one collapse, recovered by rollback). A second seed reproduced it.
- Lower exploration noise on the continuous heads (std 0.2 vs 0.5) did not help.

- Stage 2 stalled for ~2,000 iterations at a 50–58 % peak. Causes, in order of effect: 32-step rollouts (3.2 s of LSTM backprop) could not cover the flip-to-burn sequence, and lr 1e-4 wrecked the policy within two updates. 128 steps x 4096 envs with lr 3e-5 and a KL stop of 0.01 passed it in ~125 iterations (98 % eval, 86 % train); lr 3e-5 with the KL stop alone held ~40 % train but a volatile eval.
- Rollback and resume must restore a coherent state (envs, LSTM state, best score) and evals must count every episode; timeouts are terminal, not bootstrapped (fixed after an outside review).

- Stage 3 cycled between 45–64 % eval peaks and near-miss collapses (crashes at ~2 m/s, 16–62 m off) for ~1,000 iterations; GAE λ 0.95 → 0.98 (credit decay 1.9 → 4.7 s) passed it in ~200 iterations (96 % eval).

- Promotions shift the returns faster than the critic follows: at stage 10 the critic loss peaked 16x its prior median and train landing fell 94 % → 0 % within ~5 updates at a KL of ~0.004 (stage 9: 93 % → 16 %, 43 iterations to recover). The checkpoint at promotion still landed 70 % on stage 10, so the damage came from updates driven by wrong advantages, not from the stage itself.

- The KL early stop must freeze only the policy. It used to end the whole update, so when one policy step overshot `max_kl` (40 % of stage-12 iterations) the critic also got one step, fell behind (loss 70 → 720) and its stale advantages took stage 12 from 90 % to 0 % in ~15 iterations without any curriculum change; rollback recovered it. The critic now trains on the remaining minibatches.

- Independent Bernoulli engine heads made deterministic threshold decoding much worse than the
  stochastic training policy: on the same 256 stage-11 scenarios, `p > 0.5` landed 21.1%, while
  selecting the most confident `floor(sum(p))` engines landed 84.8%. The expected-count decoder is
  opt-in and is used by the salto policy in training evaluation, export and diagnostics.
- A 1% salto promotion gate advanced on lucky contacts and left an unusable deterministic controller.
  Requiring 45% landed in two consecutive 256-episode evaluations retained capability across the
  braking transitions. Fine 1.25 m/s stages then progressed only after the run was allowed to train
  for hundreds of updates rather than being restarted on short-window variance.
- Cumulative absolute horizontal rotation was not enough to define a single salto. A deterministic
  probe of an otherwise promising stage-18 controller averaged about 930° of horizontal travel and
  repeated re-flips. Final touchdown now checks exactly one inverted-to-upright recovery and zero
  re-flips, with dense overshoot, roll-travel and re-flip costs to make that sparse invariant learnable.

## Carried over from 2D

- Shaping must be γΦ′ − Φ with zero terminal potential; otherwise it biases or pays for survival.
- Speed shaping only above a safe descent profile; braking at any altitude makes policies hover.
- Fixed lr; rsl_rl's adaptive schedule pinned it at 1e-2. Reward scaling and value normalization hurt.
- entropy_coef 0.002 with lr 1e-4 made policies decisive and broke plateaus.
- Near-ground starts first; the full scenario alone was not learnable.
