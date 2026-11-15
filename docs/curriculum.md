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

## Carried over from 2D

- Shaping must be γΦ′ − Φ with zero terminal potential; otherwise it biases or pays for survival.
- Speed shaping only above a safe descent profile; braking at any altitude makes policies hover.
- Fixed lr; rsl_rl's adaptive schedule pinned it at 1e-2. Reward scaling and value normalization hurt.
- entropy_coef 0.002 with lr 1e-4 made policies decisive and broke plateaus.
- Near-ground starts first; the full scenario alone was not learnable.

## Upper-stage catch (local RTX 3090)

A separate `curriculum=catch` adds pivoting arms and hull/fin collision checks; see [catch.md](catch.md).

| Stage | Task | Status |
|---|---|---|
| 0 | Short upright approach, lower fins pass, arms close, pins captured | solved in deterministic holdout: 1,023/1,024 catches (99.9%), learned flight + arm control |
| 1 | Larger approach errors | solved: 505/512 fresh lit-engine approaches (98.6%) |
| 2 | Engine ignition | solved: 507/512 deterministic and 248/256 stochastic fresh approaches |
| 3–5 | Flip, randomized descent, wind | gradual flip curriculum: 700 m / 0–0.8 rad / 60 m/s gate passed under corrected catch physics (fresh 88.6% deterministic, 85.9% stochastic); comparing direct 1.1 rad training with 0.9/1.0 rad bridge stages; full belly-flop unsolved |
| 6–9 | Late-entry handovers, 5–20 km | not trained |

Findings (2026-11-15): the initial fixed-arm baseline omitted fin collision. Its closed gap was
11 m but its 45° lower fins spanned 15.4 m. It reached iteration 1902 at 0% catches and was stopped;
its checkpoints/GIFs are not evidence of a physically valid catch. The moving-arm replacement opens
to 30°, closes at 15°/s only after both lower fins clear, includes swept collision of all four fins,
and exposes arm opening to the policy. That automatic-tower run was stopped at 0% catches
to add direct policy control: a twelfth action now commands closing/reopening, without clearance
overrides. The fresh `catch_policy_local` run learns flight and arm timing together. A scripted 2 m/s descent clears the geometry and closes the
arms before reaching the pins; that test does not establish learned flight performance.

GPU sizing: 4,096 envs / 4 minibatches exhausted the 24 GB GPU immediately. The initial 1,024-env /
8-minibatch smoke test fitted, but the long run produced allocation spikes as episode lengths
changed. The moving-arm local preset reduces this to 512 envs / 16 minibatches with 128-step rollouts.

Validation covers finite-rate policy commands, reopening, early fin strikes, late missed catches,
Bernoulli gradients and old checkpoint/optimizer migration: 247 tests passed, 2 skipped. A local
CUDA smoke run completed three PPO updates, evaluations and Rerun recordings with the new action.
At that point no catch stage had passed evaluation; see the demonstration result below.

Observability probe (2026-11-15, `catch_policy_local/2026-11-15_12-50-20/policy_1499.pt`,
stage 0, deterministic seeds 0–7): all eight starts requested closure at t=0 with probability
above 0.9999. Four terminated on aft-fin/arm contact, four on hull/arm contact, after 1.04–1.67 s.
Their mean undiscounted return was -13.57 (terminal reward -20). Forcing the arms open, or using
the automatic clearance rule at policy frequency, avoided these early contacts but all eight
flights then tumbled after 13.63–15.38 s; mean return -197.44 (terminal reward -200).
The automatic rule never requested closure in these rollouts. These interventions show that arm
timing causes the early contacts, but flight control is also unsolved. The reward ordering is
consistent with a cheap-failure local optimum; it does not establish the policy's internal reasoning.
Next experiment should address collision-vs-tumble incentives and test explicit clearance/pin
features separately, using this probe to compare outcomes. No reward/input change was made here.

Reward-only experiment (2026-11-15): stop the saturated baseline after >2,270 iterations at 0%
catches. Start fresh `catch_reward_local` with the same 32 observations, architecture, spawn,
collision geometry and policy arm control. Set catch crash/tumble/timeout/out-of-bounds terminal
penalty floors to 200; crash severity still adds cost. This removes the 20-vs-200 structural-exit
discount while preserving success rewards. Evaluate arm decisions and flight outcomes before
adding explicit geometry features. This experiment is not yet evidence of a solved stage.

The reward-only run `catch_reward_local/2026-11-15_13-35-16` still collided early at checkpoint
100 (8/8 probe seeds). At iteration 145 the critic feature rank was 1/128, with 69% cell and 77%
output-gate saturation, while fitting a new roughly -200 failure baseline. Next isolate critic
initialization: `catch_value_init_local` retains these rewards and observations but initializes
only the critic's final bias to -200. The intent is to represent the constant failure baseline in
the bias instead of driving recurrent features into saturation. Saved checkpoint values still
override this initializer on resume. This is an optimization experiment, not a solved catch.

At checkpoint 400 of the critic-initialization run, all eight probe seeds delayed closure to
2.6–2.8 s, compared with t=0 at checkpoint 150, but still collided with uncleared fins or the hull.
Held-open attempts reached the capture plane fast and misaligned; their roughly -236 returns
remain worse than the roughly -195 early contacts. Equal penalty floors alone have not removed
all incentives for early structural exits. An opt-in 38-input geometry preset is now available
for a separate observation test; the running 32-input experiment is unchanged.

At checkpoint 600, the 32-input critic-initialized run still hit aft fins in all four probe seeds,
closing at 3.4–3.6 s. A new `catch_contact_local` preset adds 100 only for simulator-confirmed
structural contacts (hull/fins against tower/arms); failed capture constraints and ground contact
retain their existing cost. Its critic bias starts at -300 to match the new contact floor. All
32 observations, spawn and physics remain unchanged. This tests the remaining contact-vs-capture
failure incentive; explicit geometry observations remain a separate opt-in experiment.

At contact-run checkpoint 299, four probe seeds kept the arms open and failed capture constraints;
the policy had stopped choosing immediate structural contact. By iteration 1,100 it still had
zero catches and mostly timed out. A full-physics stage-0 reference controller caught 256/256
randomized validation episodes through the same 12-action interface, including mirror handling,
finite actuator dynamics and collisions. This establishes simulator feasibility, not learned
success. Initial behavior cloning (`catch_demo_pretrain/2026-11-15_14-14-20`, 256 demonstrations,
2,000 supervised updates) achieved 0% standalone catches: the best-return checkpoint cleared the
fins but drifted into an arm with its hull on both probed seeds. Next test aggregates corrective
labels on learner-visited states; the neural policy alone remains the evaluation target.

Demonstration aggregation (`catch_demo_pretrain/2026-11-15_14-16-37`) solved the short approach.
The saved update-6,000 neural policy, evaluated without teacher assistance on fresh seeds, caught
512/512 and 511/512 episodes in two independent sequential batches (1,023/1,024 total, 99.9%).
Mean pin alignment errors were 0.198 m and 0.195 m. Each batch exceeds the 90% promotion target.
The holdout used seed 173241 (+10000 for evaluation) and a frozen checkpoint copy; raw metrics
and provenance are in that run's `holdout/validation.json`. Training used the original 32 inputs,
full hull/fin collision checks, and policy-controlled finite-rate arms. The teacher only supplied
training labels; its 256/256 result is not included in these learned-policy counts. PPO resumes
from the validated checkpoint in `catch_demo_ppo_local`; subsequent curriculum stages, stochastic
PPO robustness and full belly-flop/re-entry catching remain unproven.

The initial PPO handoff exposed an exploration mismatch: the frozen successful actor caught only
2/256 stochastic episodes at continuous pre-tanh standard deviation 0.05, versus 253/256 at
0.005 and 253/256 at 0.001 (fresh batches; binary heads unchanged). The first continuation was
stopped; the immutable validated checkpoint is preserved. `catch_demo_narrow_local` restarts
from those same network weights with continuous log-std and its configured floor both 0.005,
and PPO learning rate 1e-5. This changes exploration, not the deterministic policy or catch physics.

PPO handoff diagnostic: with learning rate zero, two updates of observation normalization alone
reduced the cloned policy's evaluation catch rate to 0%. Repeating with frozen actor/critic input
statistics retained 100% deterministic evaluation and 98.8% stochastic training catches. The
`catch_demo_local` preset freezes these statistics and permits std 0.005; ordinary training defaults
are unchanged. The next continuation uses the validated network with these handoff settings.

After freezing scales, PPO at 1e-5 still degraded the actor. Refit only the critic on 256 fresh
flights from the frozen successful policy (seed 661239, 2,000 supervised updates); preserve actor
weights and normalizers. `catch_demo_stable_local/2026-11-15_14-25-18` resumes with std 0.005,
learning rate 1e-6, evaluations every five updates and rollback after one evaluation drops more
than 0.1 below the best. Stage 0 retained roughly 96–100% stochastic catches and passed both PPO
evaluations, promoting at iteration 10 to stage 1. Larger-approach catches remain at zero in the
early continuation. The validated stage-0 checkpoint and its successful GIFs are preserved.

Stage-1 postmortem: `catch_demo_stable_local/2026-11-15_14-25-18` completed 3,000 updates
(196.6M environment steps) at 0% larger-approach catches. The critic lost effective rank to 1/128;
its transition loss peaked at 446 times the preceding median. Engine use remained [1,0,0] and
switch entropy was near zero. At checkpoint 2999, a fresh 256-episode audit measured critic
Monte Carlo EV 0.273 across states and 0.119 at episode start, versus training bootstrapped EV
0.514. Raw encodings were finite; normalized absolute p99 was 15.65 for body axial velocity,
10.58 for engine-1/2 spool and 13.12 for dynamic pressure. Thus scale distribution shift and
critic/exploration failure are measurable concerns, not evidence of billion-scale raw inputs.

`catch_bridge_local/2026-11-15_17-47-33` restarts from the preserved actor and refitted critic,
with four intermediate approaches before the original larger-approach stage. The 115 m/-8 m/s
bridge passed both evaluations at iteration 55; early 130 m/-12 m/s training catches are about
38–40%. Its checkpoint 9 audit found no normalized-channel p99 above 10, though initial-state
critic EV was -0.094. Continue checking the critic as success/failure diversity changes; do not
allow a completed failed run to be mistaken for active training. The progress publisher now
matches the exact training tmux session (the updater's prefix previously caused false “running”).

Exploration audit found a reproducible clamp-boundary bug: float32 `log_std` initialized at
`log(0.005)` falls just below the Python-double lower bound, yielding zero entropy gradients
for all eight continuous noise parameters. Checkpoint 99 of the bridge run still had every
log-std exactly unchanged. Use parameter-dtype clamp bounds and project parameters into bounds
after actor optimizer steps, so later updates cannot strand them outside the gradient interval.
Deterministic actions are unchanged. At bridge checkpoint 99 (130 m/-12 m/s), fresh Monte Carlo
critic EV was 0.209 with +97.97 return-unit bias; the critic still needs investigation. Its simple
one-engine stopping estimate had 17 m median margin, unlike the original -30 m/s stage's -141 m.

Critic-only fit experiment (`catch_bridge_noise_local/2026-11-15_17-55-47/critic-fit-probe`):
keep the actor fixed and fit complete returns from 256 flights, with 192 training and 64 validation
episodes. Validation all-state EV rose from 0.238 to 0.908; a separate fresh 256-episode test
(seed 92817) gave EV 0.644, bias +2.27 and RMSE 97.49. Initial-state fresh EV remained only 0.060,
so the critic's early predictions and generalization are still weak. This is evidence of an
optimization/data problem, not proof of full value-function mastery. Continue comparing complete
returns with bootstrapped training targets. `catch_bridge_calibrated_local/2026-11-15_17-59-49`
tests that calibrated critic in PPO with the same actor, frozen encodings, learnable noise floor,
and less frequent rollback (30% relative drop for three evaluations). Early catches remain ~40%.

Position-scale experiment: lateral target channels had normalized standard deviations only 0.016
and 0.037 because positions divided by 3,000 were dominated by the normalizer epsilon 0.01.
`catch_position_local/2026-11-15_18-08-22` changes the denominator to 30 with a function-preserving
actor/critic input transform. Maximum trace-output changes were 4.3e-6 (actor) and 3.8e-5 (critic);
256 matched seed-92817 episodes retained identical outcomes and lengths, 106/256 catches. The
first 100-plus PPO updates still gave roughly 40% catches: better input sensitivity alone has
not yet solved bridge 2. Do not label this migration as improved learned performance.

Upright engine-ignition approach solved with network-only control:
`catch_approach_pretrain/2026-11-15_18-21-43/model_demo_9600.pt` catches 507/512 fresh
engine-off approaches (seed 487231), 505/512 already-lit approaches (seed 594823), and
248/256 stochastic engine-off approaches (seed 921893). These start at 200±50 m,
-30±5 m/s with lateral offsets and use the unchanged arm/fin collision checks. A reference
controller supplies demonstrations and dataset aggregation; inference uses only the trained
network. The controller aligns above the arms, allocates thrust across one or three engines,
and gates closing on pin alignment and speed. It deliberately rejects belly-flop/entry starts.
Position denominator is 30; frozen normalization and std floor 0.005 preserve the handoff.

Critic quality is not implied by catch success: fresh all-state Monte Carlo EV is -0.395
(engine-off), -1.583 (lit), and -0.237 (stochastic). Engine-off RMSE is 35.43 against return
standard deviation about 30; stochastic RMSE is 50.42. Continue critic calibration and fresh
validation. `learning_audit --stochastic` now measures the sampled policy reproducibly and
keeps its output separate from deterministic audits. Supervised checkpoints are immutable
`model_demo_UPDATE.pt`/`policy_demo_UPDATE.pt`, retain PPO iteration -1, and record their
supervised update separately; PR progress labels these as supervised updates, not PPO steps.

The belly-flop transition remains unsolved. An experimental rate-limited upright controller
still tumbled in all 256 low-belly-flop trials: one trace accumulated >100 m/s lateral speed
while turning, then exceeded the 90 degree/s roll limit. Fixed neutral flaps and sideways
thrust require investigation; no simulator limits have been relaxed to make these pass.

Full-belly-flop PPO diagnostic (`catch_approach_ppo/2026-11-15_18-34-09`): the preserved
approach passes stage 6, but 200 updates give no belly-flop catches. Fresh checkpoint-199 audit
(seed 819238, 256 episodes) finds 256 tumbles, median duration 2.96 s and median end altitude
917 m, with no completed flip. Unlike the unsuccessful actor, the critic now predicts this
failure regime: Monte Carlo all-state EV 0.930, initial-state EV 0.816, RMSE 10.34, initial bias
+12.09. High EV alone is not evidence of a good policy. Frozen upright normalization produces
large belly-flop outliers (actor position-channel absolute p99 about 602, attitude about 128).
The audit now names channels, measures actor and critic normalization separately, and saves
terminal time, pose, velocity, rates, pin miss and flip/ignition altitude for each episode.

`catch_flip_local/2026-11-15_18-37-48` restarts from the immutable approach checkpoint, retaining
the existing 512×128 rollout batch and frozen normalizers. Six intermediate pitches from 0.1
to 1.1 rad precede the original 90-degree stage; speed, height and offset widen gradually.
Stages 0–6 and the final entry target remain unchanged. The first 0.1-rad stage has roughly
66–74% early catches, not yet solved. A separate critic-only fit to stochastic approach returns
barely improved held-out RMSE (38.67 to 38.24); it was not substituted into this PPO run.

Tilt-bridge learning audit: at update 99, 183/256 fresh flights catch; critic all-state EV is
0.553, but initial-state EV is -0.010 with +79.55 bias. The initial std 0.005 leaves most actor
updates stopped by the KL guard after a few minibatches. A matched noise probe at update 279
(seed 819242, 256 sampled flights each) catches 67.2%, 64.8%, 61.7% and 10.2% at continuous
std 0.005, 0.01, 0.02 and 0.05 respectively. The 0.02 continuation permits about 74 actor
optimizer steps per update versus 4 in the matched baseline window, with similar early catch
rates. This restores learning opportunity, not proof of improved catching. The baseline was
stopped after roughly 530 updates while the higher-noise experiment continues.

Critic phase weighting experiment: fit the same update-279 actor's critic on 1,024 complete
fresh deterministic episodes, with an episode-disjoint 768/256 train/validation split. A pilot
assigns half the regression loss to the first 20 control steps (2 s), instead of allowing long
successful episodes to dominate; actor and normalizers stay fixed. On a separate matched
512-flight test (seed 819245), ordinary fitting gives all-state/initial EV 0.683/0.195 and
initial RMSE 148.91; early weighting gives 0.797/0.336 and initial RMSE 135.02, bias +3.14.
The pilot selected by a blend of all-state and initial-state validation MSE. The reusable
`critic_calibrate --initial-steps 20 --initial-weight 0.5` instead consistently weights the
same initial window for both fitting and selection; its fresh validation is checked separately.
The default remains ordinary fitting. Calibration now includes the unmodified critic as a
candidate, so a wholly unsuccessful fit cannot silently replace it with a worse checkpoint.

The reusable phase-weighted tool selected update 1,900 and reproduced the pilot's fresh
512-flight EV/RMSE results exactly. The next PPO comparison uses that calibrated critic and
std 0.02; its actor is verified bit-for-bit equal to the noise-only control at initialization.

PPO comparison from identical actor weights: after 50 updates, a matched fresh 512-flight test
(seed 819246, checkpoint 329) catches 428 with the calibrated critic versus 402 with noise only.
Initial observations are identical; calibration recovers 30 failures and introduces 4. The
paired improvement is 5.08 percentage points (bootstrap 95% interval 2.93–7.23 points).
The calibrated continuation passes the 0.1-rad stage at iteration 384; a separate fresh test
(seed 819249) catches 452/512 (88.3%), so the promotion is not proof of universal mastery.
Initial critic bias has drifted to -62.0 as the actor improves. An optional
`runner.algorithm.critic_learning_rate` now separates value learning from the small actor
rate imposed by its KL guard. Shared/split optimizer checkpoint loading preserves Adam moments;
the configured rates are reapplied after resume and actor scheduling leaves the critic rate fixed.
The default remains one shared rate; faster-critic experiments require separate fresh validation.

Tilt/altitude ablation of the passed actor (seed 819248, 256 flights per condition): changing
only altitude from 200 to 250 m retains 73.4% catches. At 200 m, fixed pitch 0.125 rad catches
29.7%, pitch 0.15 rad 0.8%, and pitch 0.2 rad 0%. The narrow attitude distribution is the main
cliff. A uniform 0–0.15 rad range retains 75.4% at 200 m and 72.3% at 250 m; 0–0.2 rad retains
52.7% at 200 m. `catch_flip_range` therefore keeps easier attitudes in each expansion, and
separates subsequent altitude and descent-speed changes. Stages 0–7 and the final entry target
remain unchanged. Parallel runs `catch_flip_range_fast/2026-11-15_19-02-17` (critic LR 1e-5)
and `catch_flip_range_control/2026-11-15_19-02-17` (shared LR 1e-6) start from the identical
passed actor/critic and must re-establish the stage-7 threshold after reset before advancing.
The initial observed range success is a counterfactual baseline, not a trained result.

Range failure localization (checkpoint 439, fresh seed 819250): the fast-critic policy catches
93–100% below initial tilt 0.1 rad, 78% at 0.1–0.125, and only 9% at 0.125–0.15. Its fresh
all-state/initial critic EV is 0.848/0.609, versus 0.826/0.597 for the shared-rate control;
catch rates remain similar (401/512 versus 404/512). Replay interventions predominantly hit
the tower while arms are still open, so arm timing does not explain this failure boundary.
A matched per-head noise probe (seed 819251, checkpoint 519) failed to improve the high-angle
subset; increasing gimbal/flap noise reduced total catches. Do not adopt those perturbations.

The replay probe now exports every requested/executed action and post-tanh continuous target,
with Rerun flight-command plots. Targets are explicitly in the policy frame before mirroring.
Its position-scale explanation reads the actual observation config (30 for these runs), replacing
the stale hard-coded 3,000. A reference-controller check at 0.2 rad and 250 m (256 matched
seed-819252 starts) catches 243 with lit engines versus 4 with cold engines. This identifies an
ignition/attitude-control interaction, not a physical infeasibility result; broader lit-engine
starts are being investigated as demonstration data before returning to cold-start control.

Lit-angle demonstrations (`catch_tilt_pretrain/2026-11-15_19-17-26`): the network at supervised
update 6,700 catches 512/512 fresh lit-engine starts over 0–0.2 rad at 250±50 m (seed 819254),
but only 1/512 matched cold starts. The single-stage result therefore does not solve ignition.
`catch_tilt_demo` explicitly follows the lit stage with the same spawn envelope and cold engines,
then retains the wider flip/entry target. It does not relax collision or actuator constraints.

`catch_pretrain --stage 8 --replay-stage 6` now mixes complete cold-upright episodes with the
lit-angle demonstrations, including each stage again in dataset aggregation. Normalizer fitting
sees both distributions; episode labels preserve provenance and padding never joins histories.
Both stages are evaluated independently, and the minimum catch rate selects the checkpoint, so
success on the new task cannot hide forgetting the earlier one. The target curriculum stage stays
8 in saved checkpoints; replay evaluation does not change it. The mixed run
`catch_tilt_replay_pretrain/2026-11-15_19-23-34` is ongoing; neither mixed-task success nor
transfer to cold tilted starts is assumed from the lit-only result.

Mixed-stage validation: `catch_tilt_replay_pretrain/2026-11-15_19-23-34/model_demo_8900.pt`
was selected by a 100% minimum internal catch rate across the two demonstrated tasks. Fresh
seed-819259 evaluation catches 504/512 cold upright approaches and 511/512 lit tilted approaches.
At update 5,800, a separate fresh test already caught 379/512 cold tilted approaches absent
from the demonstrations, versus the lit-only policy's 1/512 on its earlier cold-tilt test.
Those tests use different seeds/checkpoints; the transfer result is progress, not a paired
causal estimate or proof of a full flip.

The selected actor is preserved unchanged except for continuous std 0.02 in
`catch_tilt_warmstart/2026-11-15_19-29-03/model_-1.pt`, explicitly advanced to the cold-tilt
stage with PPO iteration still -1. A 1,024-episode stochastic audit (seed 819260) catches
55.4%, with critic all-state EV -0.029 and initial bias +156.45 return units. This run needs
critic calibration before PPO; the cold tilted task is not marked solved. The same early-window
calibration and fresh-seed validation used in the successful earlier comparison are applied
before this handoff. Moving-arm geometry, collisions and the final entry curriculum are unchanged.

The cold-tilt critic fit selects update 1,800. Fresh stochastic evaluation on 512 episodes
(seed 819261) catches 54.7%, with all-state EV 0.372, initial EV 0.463 and initial bias +13.69;
the actor is unchanged. `catch_tilt_ignition_local/2026-11-15_19-32-20` resumes PPO at stage 9
from this calibrated checkpoint, using the inherited frozen normalizers, actor LR 1e-6,
critic LR 1e-5 and continuous std 0.02. Promotion and full-entry success still require actual
policy evaluations; this calibration result alone does not establish either.

Cold-start PPO passed stage 9 and reached the 0–0.3 rad stage. At checkpoint 269,
512 fresh deterministic episodes (seed 819264) catch 417/512 (81.4%). The new fixed-angle
initial-state audit exposes the remaining boundary: 251/251 catch below 0.15 rad,
82/97 at 0.15–0.2, and 84/164 at 0.2–0.3. Complete-return critic EV is 0.757 across
all states but only 0.437 at spawn and 0.190 within the hardest angle bin; initial bias
there is -41.36 return units. The easy bins have negative EV despite small absolute errors
because their return variance is very low, so the report includes RMSE, bias and target
standard deviation alongside EV. No observation-scale warnings appeared in this audit.
Training continues; neither this gate promotion nor the aggregate score establishes a full flip.

The matched stochastic checkpoint-269/359 comparison uses seed 819265 and verifies identical
initial observations: catches improve from 404/512 to 437/512 (42 gained, 9 lost), a 6.45 pp
gain with paired-bootstrap 95% interval 3.91–9.18 pp. Initial-state critic EV improves from
0.344 to 0.404. The run subsequently passes the 0–0.3 rad gate and reaches stage 11 (350 m).

Stage-11 checkpoint 449 catches 373/512 fresh deterministic starts (seed 819266). Above
0.2 rad it catches only 61/176; normalized tower-heading, attitude and lateral-target p99
magnitudes reach 13–23. A 16-seed arm intervention probe yields the same 9 catches, 5 tumbles
and 2 hull/tower contacts under policy versus automatic closure. This points to flight control
rather than early closure in these sampled failures.

A normalization experiment preserves the function while changing input coordinates: fit means
and variances on the checkpoint's 130,502 valid audited transitions, then compensate the LSTM
input weight and bias for `(x - mean) / (std + eps)`. The maximum recorded-sequence output
change is 1.46e-5 for the actor and 2.98e-4 for the critic; all 512 matched rollout terminal
statuses remain identical, while the largest normalized p99 falls below 5. This establishes
behavior preservation, not improved learning. Both 200-update arms reset Adam history, resume
checkpoint 449 with seed 819267, and keep normalizers frozen. Runs are
`catch_normalizer_control_local/2026-11-15_19-44-49` and
`catch_normalizer_rescaled_local/2026-11-15_19-44-49`; checkpoint handoffs and intervention
provenance are in the corresponding `catch_normalizer_{control,rescaled}_handoff` directories.
The main run continues independently; the normalization change is experimental, not adopted.

The checkpoint-449 actuator audit adds another distinction: raw Gaussian exploration is not
actuator exploration after `tanh`. At spawn, 53.1% of gimbal-3 means and 64.6% of roll-RCS means
have `abs(tanh(mean)) > 0.99`. Despite raw std approximately 0.02, their median transformed
plus/minus-sigma interval half-widths are only 0.0149% and 0.00471% of the full actuator range.
Across valid flight steps, roll-RCS raw mean magnitude p99 is 11.37. The audit exports means
and standard deviations and reports both initial and whole-flight saturation. These intervals
are not transformed standard deviations or measured actuator response, and saturation alone
does not prove a control error: a full-range command may be appropriate.

After 200 matched PPO updates, the normalization experiment's fresh 1,024-episode stochastic
holdout (seed 819272) catches 769 with rescaling versus 749 with original statistics. There
are 42 gained and 22 lost catches: +1.95 pp, paired-bootstrap 95% interval +0.49 to +3.52 pp.
The gain is modest and neither arm passes 350 m; initial-state critic EV is 0.518 rescaled
versus 0.554 control. At 100 updates, a separate matched test caught 400/512 versus 389/512.
The compensated statistics are a promising training intervention, not a solved encoding fix.

Inference-time action caps are rejected: on 256 matched seed-819269 flights at checkpoint 449,
original commands catch 188, all continuous means capped to ±3 catch 186, ±2 catch 183, and
roll-RCS-only ±2 catches 188. All reports are labeled counterfactual and preserve the source
checkpoint. Separately, changing only spawn altitude at this checkpoint (512 episodes,
seed 819266) catches 454 at 250 m, 439 at 300 m and 373 at 350 m; larger tilts account for
most failures. This supports testing a smaller altitude transition if direct learning stalls.

`catch_lambda995_local/2026-11-15_19-54-58` tests longer temporal credit from the same
checkpoint-449 optimizer-reset control, seed 819267, for 200 updates: only GAE lambda changes
from 0.98 to 0.995. Gamma, rollout horizon, learning rates, input statistics and physical
constraints remain unchanged. Outcome selection awaits fresh matched evaluations.

The completed lambda-0.995 experiment catches 750/1,024 versus 749/1,024 for its matched
lambda-0.98 control (seed 819272): 21 gained and 20 lost catches, paired 95% interval
-1.17 to +1.37 pp. Initial critic EV is 0.548 versus 0.554. Longer GAE credit is not adopted.

Two longer runs now resume the rescaled checkpoint 649 with seed 819274 for 2,000 updates:
`catch_conditioned_control_local/2026-11-15_20-03-57` retains actor LR 1e-6;
`catch_conditioned_lr3_local/2026-11-15_20-03-57` uses 3e-6. Both retain lambda 0.98,
critic LR 1e-5, frozen normalization, optimizer history and the KL guard. Independent
512-flight stochastic audits are scheduled every 250 updates; matched comparisons require
identical initial observations and curriculum stage. The main PR publisher hands off to the
conditioned control after the original run's final render, continuing description-only GIFs.

`uv run python -m starship.train.return_uncertainty CHECKPOINT --starts 128 --repeats 8`
now separates action-noise variability from initial-state critic error. `learning_audit` accepts
`--stochastic --action-seed N` independently of the spawn seed and uses distinct default output
paths. The repeated audit asserts identical initial observations, predictions and checkpoint
hashes; it reports within-start variance, noise-corrected between-start variance, and critic
errors against both individual returns and repeat means. Its EV ceiling is a finite-repeat
variance-component estimate, not a proven bound or a control-quality score.

On conditioned-control checkpoint 749, eight repeats of 128 starts (spawn seed 819283,
action seeds 819284–819291) catch 80.7%; 35 starts have mixed outcomes. Estimated single-flight
EV ceiling is 0.633, while the critic's individual-return EV is 0.391 and mean-return EV 0.576
(bias -11.05, mean-target RMSE 71.75). Thus action noise contributes materially, but does not
explain all critic error. These are matched repeated starts, not 1,024 independent spawn states.

Coherent-control search breaks the 350 m plateau. With conditioned-control checkpoint 899,
a fixed 350 m / 0.3 rad start tumbles under the original policy. A grid over short throttle
and pitch-gimbal offsets finds valid catches without changing physics, actuator limits or
contacts. On 512 fresh matched randomized starts (seed 819294), subtracting 2 from the raw
throttle command for the first 60 steps catches 507 versus 412: 95 gained, none lost, +18.55 pp
(paired-bootstrap 95% interval +15.23 to +22.07 pp). This is an external intervention, not a
standalone policy result. Artifacts are `coherent-control-probe` and
`coherent-control-validation` under the source run.

Warm-start distillation in `catch_coherent_distill_local/2026-11-15_20-24-56` teaches this
correction while replaying unmodified 250 m trajectories. Selected update 600 catches 505/512
fresh deterministic and 491/512 stochastic 350 m starts (seed 819297), but earlier-stage
retention remains weaker. A 20-step correction of throttle -2 and gimbal3 -0.5, applied only
when initial tilt exceeds 0.15 rad, works across both altitudes: source-policy interventions
catch 508/512 at 350 m (seed 819294) and 500/512 at 250 m (seed 819296).

`catch_coherent_pair_distill_local/2026-11-15_20-28-21` trains both stages with that shorter
correction, frozen input statistics and one round of student/teacher trajectory aggregation.
The minimum internal catch rate selects supervised update 500. Fresh **network-only** tests
catch 1,012/1,024 at 350 m and 997/1,024 at 250 m (seed 819300), plus 984/1,024 stochastic
350 m starts (seed 819301). The untrained next stage, 0–0.5 rad at 350 m, catches 410/512
(seed 819302). No pulse or external controller is used in these student evaluations. PPO
iteration remains the inherited 899; supervised updates are separate, and stale PPO Adam
moments are cleared. Executed research scripts and full demonstration data are archived in
the run directories. The actor is preserved during a subsequent phase-weighted critic fit;
fresh calibration validation precedes further PPO. Full belly-flop/entry remains unsolved.

The critic fit selects update 700. On 512 fresh stochastic flights (seed 819304), actor weights
and outcomes remain identical (487 catches); full-flight value RMSE improves 68.78 → 58.42
and EV 0.104 → 0.353, while initial RMSE remains about 62.5. The source is preserved.
`catch_coherent_ppo_local/2026-11-15_20-35-20` resumes from that calibrated student for 2,000
PPO updates with actor LR 1e-6, critic LR 1e-5 and the full existing curriculum. The older
learning-rate comparison had no clear winner at 500 updates (407/512 versus 402/512,
seed 819276), so it does not justify a larger actor rate for this handoff.


The coherent PPO continuation passed the 350 m / 0–0.5 rad gate, but its passed-stage
checkpoint catches only 919/1,024 fresh starts (seed 819327). At 500 m, independent
512-flight stochastic audits improve from 20.1% at iteration 1049 to 37.7% at 1549;
initial critic EV improves from -0.084 to 0.167. Terminal fuel measurements rule out
fuel exhaustion as the dominant failure. Replacing engine/throttle commands alone with
the approach teacher makes catches worse. A full-controller handover after tilt <0.1 rad
and angular speed <0.3 rad/s is more useful: on matched 1,024 starts (seed 819329),
policy 989 catches 197 alone versus 594 with handover; policy 1549 catches 421 versus
735. Increasing the reference controller's unaligned holding altitude from 35 m to
60/100/150 m worsens its matched 512-flight result. These interventions change actions,
not physics, and are training-data experiments rather than standalone learned results.

`catch_handover_distill_local/2026-11-15_20-54-32` distills policy 1549 plus that handover,
with policy 989 replay at 350 m and one student/teacher trajectory aggregation round.
Internal evaluation selects supervised update 1200. Fresh network-only tests catch
727/1,024 at 500 m and 970/1,024 at 350 m (seed 819331). Stochastic 500 m evaluation
falls to 527/1,024 (seed 819332): robustness to action sampling remains an issue, and
its initial critic overestimates returns by 58.36. PPO ancestry remains iteration 1549;
these supervised updates are not PPO steps. The original PPO run continues for comparison.
Scripts, demonstrations, immutable candidates and holdout reports are archived in these
run directories. Full re-entry remains unsolved.

The holdout exposed a braking-diagnostic race: its Warp kernel defaulted to CUDA while
writing CPU buffers, occasionally producing incomplete values. The launch now explicitly
uses CPU and rejects nonfinite physics or nonpositive mass. Previously recorded braking
estimates may be unreliable; catch outcomes and critic-return metrics use separate paths.


`catch_handover_robust_distill_local/2026-11-15_20-58-40` addresses the sampling gap
by collecting student/teacher mixtures with stochastic student actions, keeping the
same reference labels and 350 m replay. Internal evaluation selects supervised update
400. Fresh tests catch 773/1,024 deterministic 500 m starts and 993/1,024 at 350 m
(seed 819335), plus 722/1,024 stochastic 500 m starts (seed 819336). These are separate
fresh seed sets, not a paired comparison with the previous student's percentages.
The stochastic initial critic bias is -0.54 and EV 0.406; the 0.3–0.5 rad group catches
178/402, identifying the largest-tilt starts as the remaining weakness.

A subsequent critic-only fit selects update 500. On a fresh matched stochastic set
(seed 819338), both checkpoints catch 338/512 with bitwise-identical actors and identical
trajectories. Full-flight value RMSE improves 86.96 → 78.08, EV 0.601 → 0.677; initial
RMSE improves only 113.20 → 112.24. `catch_handover_ppo_local/2026-11-15_21-03-46`
resumes this calibrated student for 2,000 PPO updates (seed 819339, inherited iteration
1549), while the older coherent PPO continues for comparison. After 50 updates, an
independent 512-flight stochastic audit catches 404 (78.9%, seed 819340); its 0.3–0.5 rad
group catches 122/198. The training PR's description publisher now follows this run.

A 24-seed deterministic arm intervention probe of its policy 1559 catches 18 with learned
arm commands. Five failures occur before any close command: three tumbles, one hull/tower
strike and one aft-fin/tower strike. Those failures are unchanged with forced-open arms.
The sixth reaches the capture plane with closed arms but about 0.90 m pin-alignment error;
automatic arm timing changes the subsequent flight and succeeds on that seed. This small
probe supports prioritizing flight recovery and alignment over a general early-closure fix.
The full traces are in the run's `catch-probe/policy_1559` directory; the full re-entry
objective remains unsolved and the training PR remains draft.


Continued PPO in `catch_handover_ppo_local/2026-11-15_21-03-46` improves independent
512-flight stochastic catches from 404/512 at iteration 1599, to 424/512 at 1799 and
445/512 at 1899. A repeated-start audit of iteration 1599 (128 starts × 8 action seeds,
spawn seed 819361) finds mixed success/failure outcomes at 54/128 starts. Initial critic
EV is 0.351 against individual returns and 0.560 against per-start mean returns; the
finite-repeat estimate of the single-flight EV ceiling is 0.575, not a proven bound.
The older coherent PPO run was retired at saved checkpoint 2109 after its latest three
independent audits remained at 52.9%, 55.5% and 54.5%; all its artifacts are retained.

The new run passes the 500 m / 30 m/s / 0–0.5 rad gate at iteration 1969. Its immutable
`model_stage13_passed.pt` catches **932/1,024 deterministic** fresh starts (seed 819426)
and **450/512 stochastic** starts (seed 819427). Thus the gate passed, but sampled-action
robustness is still below 90%. Training continues at stage 14, which increases downward
speed to 45 m/s. A live promotion watcher archives 1,024 deterministic and 512 stochastic
fresh flights for each subsequent passed-stage snapshot in `promotion-holdouts`, separately
from the every-100-update audits. The executed watcher is archived as `watch_promotions.py`
in the run directory. Full belly-flop and re-entry catches remain unvalidated.


The same run passes the 45 m/s descent gate at iteration 2029. Its saved stage-14
checkpoint catches **929/1,024 deterministic** fresh starts (seed 819428) and
**454/512 stochastic** starts (seed 819429). Initial stochastic critic bias settles to
+0.61, though EV is only 0.068. Training advances to stage 15: the initial tilt range
expands from 0–0.5 to 0–0.8 rad at 500 m and 45 m/s. This is still a gradual flip
curriculum, not full belly-flop or entry validation.


Stage-15 intervention checks (same run, all unchanged physics): policy 2189 catches
24/32 deterministic probe seeds; failures are four tumbles, three aft-fin/arm strikes
and one capture-constraint failure. The three fin strikes request closure with negative
aft-fin clearance. Automatic arm commands catch 26/32, while forced-open arms retain all
four tumbles and cannot capture. These are small diagnostic samples, not holdouts.

A fixed 500 m / 0.8 rad search finds a successful gimbal3 raw-command bias of -0.5 for
40 or 80 steps, but the randomized matched test rejects it: on 512 starts (seed 819371),
source model 2189 catches 408, versus 331/328 with the correction applied only above
0.5 rad initial tilt. The 40-step intervention gains nine successes but loses 86.
Likewise, full reference-controller handover after uprighting catches 284/512 versus
395/512 without it (seed 819372); arm-only reference commands catch 407/512. Neither
result justifies a new full-controller distillation round.

Policy 2399's larger arm-clearance probe catches 855/1,024 baseline starts (seed 819373).
Vetoing closure while aft-fin clearance is negative catches 875: 20 gained, none lost,
paired-bootstrap 95% interval +1.17 to +2.83 percentage points. An automatic-rule close
veto catches 880 (26 gained, one lost). These are external action interventions, not
learned-policy results; no runtime safety guard was installed. Early closure is a real
failure mode, but fixing it alone leaves a substantial flight-control gap.

A later handover test uses policy 2499 and waits for tilt <0.1 rad, angular rate <0.3 rad/s,
absolute vertical speed <5 m/s and pin height between 12 and 50/100 m. On 512 matched
starts (seed 819375), baseline catches 451 and the two handovers catch 457/458. The
100 m variant gains 15, loses eight, and 42 of its 54 failures never meet the handover
condition. These small net gains do not justify replacing the ongoing PPO run. Executed
scripts and matched reports are archived under `wide-tilt-control-validation`,
`wide-tilt-handover-probe`, `arm-clearance-probe` and `late-handover-probe`.

The iteration-2342 health review finds no dead actor/critic units and active MLP/LSTM
weight updates. The near-zero center-engine entropy warning corresponds to ~99.96%
on commands, while each outer engine is on about 25% of the time. The slow distribution
update warning concerns noise scales near 0.02, not a frozen actor. Both warnings were
reviewed without changing entropy or noise solely to silence them. PPO and independent
audits continue; the full re-entry target still has zero evaluated catches.


The wider-tilt gate passes at iteration 2809 in the handover PPO run. The passed-stage-15
snapshot catches **897/1,024 deterministic** fresh starts (seed 819430) and **422/512
stochastic** starts (seed 819431), below the internal promotion rate. The next stage
changes only mean altitude, from 500 to 700 m, retaining 0–0.8 rad pitch, 45 m/s descent,
cold engines and ±50 m altitude noise. Catch rate falls sharply. At iteration 2899,
independent stochastic evaluation catches 57/512, with 339 tumbles and 116 crashes;
initial critic bias is +0.29 and EV 0.388. In a 16-seed deterministic probe of policy 2889,
14 flights tumble and two collide; forced-open and automatic arms give identical outcomes.
This transition primarily exposes flight-control generalization, not arm timing or a
persistently optimistic critic.

A counterfactual altitude sweep retains the exact passed-stage-15 checkpoint bytes and
changes only mean altitude. With 512 starts at seed 819376, catches are **441 at 500 m,
221 at 600 m, and 44 at 700 m**. These are configuration interventions, not retrained
policies. Reports and executed script are archived in `altitude-bridge-probe`. Large
normalized outliers also occur during the failing flights, but this does not establish
an encoding-scale cause.

`catch_altitude_bridge_local/2026-11-15_21-42-45` tests 550/600/650 m bridge stages before
returning to the original 700 m task. It resumes model 2809 at inherited iteration 2809;
its actor is verified identical to the passed-stage-15 snapshot. In this experimental
curriculum, stage 16 is 550 m and the old 700 m stage becomes stage 19. All subsequent
original stages remain, and an exact dataclass comparison verifies identical final sim
and reward configurations. The complete structured launch config and provenance are
archived in the run, so stage numbers must be interpreted with that run's config. It
trains for 2,000 updates with seed 819377 alongside the direct 700 m run; no physics,
collision rules or final objective are relaxed. Both have independent checkpoint and
promotion holdouts. The PR-description publisher still follows the direct run pending
evidence that the bridge improves the same 700 m task.

Recurrent phase auditing: `uv run python -m starship.train.recurrent_audit <learning-audit dir>`
replays the exact checkpoint from zero hidden state, verifies saved actor means and critic
values, and reports spawn, initial-window, later-flight and terminal-window gate saturation.
It masks autoreset frames and checks checkpoint SHA; `--window-steps` defaults to 20,
`--batch` bounds replay memory, and `--checkpoint` supports an identical relocated checkpoint
with its run config. Initial and terminal windows can overlap on short episodes. The
`cell_saturated` metric is the tanh cell-candidate gate, not the stored cell state.

The fixed-policy altitude sweep shows actor cell-candidate saturation at spawn rising
from 34.5% at 500 m to 39.3% at 600 m and 43.7% at 700 m. After the first two seconds,
the corresponding fractions are 3.6%, 11.7% and 34.9%; the final two seconds of 700 m
flights reach 57.8%. Thus saturation is already altitude-dependent before flight diverges,
but much of the aggregate increase accompanies later failure. Successful 500 m flights
also start with substantial saturation, so this does not establish an encoding defect.
Replay outputs match the recorded actor/critic outputs exactly for these three datasets,
the trained 700 m model 2999 and the 550 m bridge model 2859. Full reports and the original
multi-run research script are archived in the handover run's `lstm-phase-audit` directory.


The altitude bridge passes 550 m at iteration 3099. The immutable passed-stage-16 snapshot
catches 918/1,024 deterministic starts (seed 820132) and 439/512 stochastic starts
(seed 820133). It proceeds to 600 m; its first independent stochastic audit there catches
368/512 (model 3159, seed 819381). The direct 700 m run remains active as a comparison.

Matched checkpoint tests use 1,024 identical starts and common action sampling seeds at
each altitude (spawn 819601, actions 819602). At 600 m, original policy 2809 catches 459,
direct-700 m policy 3389 catches 631, and bridge policy 3159 catches 705. At 700 m the
same policies catch 79, 225 and 136 respectively. Bridge-minus-direct paired-bootstrap
95% intervals are +4.49 to +9.96 percentage points at 600 m, but -10.84 to -6.64 points at
700 m. The bridge improves its current task without yet surpassing direct training at the
harder altitude; their different update counts and training histories prevent interpreting
this as an equal-budget causal efficiency comparison. Results and executed script are in
the bridge run's `matched-altitude-comparison` directory. Both curricula retain the original
full re-entry target, which remains unsolved.

The bridge subsequently passes its 600 m training gate. Its immutable passed-stage-17
checkpoint catches 895/1,024 deterministic starts (seed 820134) and 397/512 stochastic
starts (seed 820135), then trains at 650 m. These holdouts remain below the training gate;
promotion alone is not evidence of robust mastery. Meanwhile, direct-700 m training
continues from model 3549 in `catch_handover_continue_local/2026-11-15_22-05-53`.
Model 3799 catches 146/512 stochastic starts; its initial-state critic explained variance
is 0.653 and bias is +1.68 return units. Predicting outcomes increasingly well has not yet
solved the early-turn control failure.

A short-control search on continuation policy 3599 identifies lower initial throttle as
useful at 700 m. On 1,024 fresh matched deterministic starts (seed 819605), a raw throttle
bias of -1 for the first 80 steps raises catches from 293 to 387. A separate expanded
search selects raw throttle -2 and gimbal-3 +0.5 for the first 80 steps. On another fresh
1,024 matched starts (seed 819606), that intervention raises catches from 284 to 433
(196 gains, 47 losses); a four-second throttle -3 alternative catches 394 (145 gains,
35 losses). Initial observations are asserted identical for each comparison. These are
counterfactual action interventions through ordinary actuator and collision constraints,
not learned-policy success rates or evidence of changed physics. Search-set hindsight
unions are not deployable performance. Scripts and paired results are archived under the
continuation run's `initial-control-search*`, `initial-control-validation`, and
`expanded-control-validation` directories. A separate student-distillation experiment
uses the eight-second intervention as labels with passed-500 m replay and stochastic
student rollouts; fresh policy-only validation is required before any PPO handoff.

The eight-second correction transfers to a standalone recurrent policy in
`catch_700_control_distill_local/2026-11-15_22-19-07`. Training uses frozen input statistics,
500 m passed-policy replay, and stochastic student/teacher rollouts, with another data
collection round after 1,000 supervised updates. Internal selection chooses update 1900;
PPO iteration 3599 is retained separately. Fresh matched 1,024-start tests compare it with
its unmodified source model 3599, asserting identical initial observations:

| Task | Actions | Source catches | Student catches | Gained / lost | Seed |
|---|---|---:|---:|---:|---:|
| 700 m | deterministic | 321 | 480 | 183 / 24 | 819611 |
| 700 m | stochastic | 298 | 427 | 176 / 47 | 819612 |
| 500 m, tilt 0–0.8 rad | deterministic | 874 | 919 | 67 / 22 | 819611 |
| 500 m, tilt 0–0.8 rad | stochastic | 830 | 910 | 119 / 39 | 819612 |

Critic-only calibration preserves every actor tensor and its input statistics. Another
512 stochastic starts (seed 819614) give identical trajectories and 223 catches before
and after calibration. All-state value RMSE improves from 102.18 to 92.27, explained
variance from 0.590 to 0.665; initial-state RMSE improves from 101.03 to 99.41, with bias
-10.35 to +1.21. The calibrated checkpoint starts a 2,000-update PPO continuation in
`catch_700_student_ppo_local/2026-11-15_22-24-15`, seed 819620. Its first numbered checkpoint
retains both frozen normalizers exactly. PR 67's fixed-seed Rerun GIFs now follow this
run, with no success filtering. The direct-700 m and altitude-bridge comparisons continue.

A 32-start deterministic arm probe of student policy 3609 finds 13 catches, seven tumbles,
five capture-constraint failures, four aft-fin/arm strikes and three tower contacts.
Automatic arm control catches only 12. All four policy aft-fin/arm strikes occur while
rising at about 1.4–5.8 m/s into fully closed arms, after the first close command had
positive fin clearance. These are rebound failures, not evidence that every fin strike
comes from closing before the initial downward fin passage. CSV geometry and actuator
traces are in `catch-probe/policy_3609`.

A matched 512-start final-descent intervention (seed 819630) switches to the reference
controller only after tilt <0.1 rad, angular rate <0.3 rad/s, |vertical speed| <5 m/s and
pin height in (12,50), (12,100), or (2,50) m. Catches increase from 237 to 251, 251 and
252 respectively; the last setting gains 24 and loses nine, with 216 remaining failures
never reaching its handover condition. This modest counterfactual gain does not solve
the larger early-turn failure population and is not installed as a runtime override.
Scripts and comparisons are archived under the student PPO run's `late-handover-probe`.

After 50 more PPO updates, student model 3649 catches 285/512 fresh stochastic starts
(seed 819621). A separate same-task comparison uses 1,024 identical 700 m initial states
and common stochastic action seeds (spawn 819640, actions 819641): student policy 3649
catches 575, continued direct-training policy 3979 catches 368, and bridge policy 3759
(currently trained at 650 m) catches 316. Different training histories and budgets mean
this is a checkpoint comparison, not a causal sample-efficiency claim. The weaker direct
continuation is retired with its checkpoints preserved; student PPO and altitude-bridge
PPO remain active. Executed scripts, policy hashes and trajectories are archived in the
student run's `matched-policy-comparison` directory.

### Catch-plane event correction

Student model 3949 reaches 390/512 catches in a fresh stochastic audit (seed 819624).
The same-seed arm probe improves from 13/32 catches at policy 3609 to 25/32 at policy
3999, with 13 gains and one loss. Three of the four earlier rebound strikes now catch.
The remaining two fin strikes expose a simulator event-detection bug: their pins pass
below the support with closed arms, then hit a forward fin several seconds later.

An individual-substep replay reproduces both original trajectories with zero altitude
error at control boundaries. At the missed crossings (seeds 16 and 31), fuel burn raises
the COM station by about 0.00030 m. The catch test previously used the pre-burn COM at both
endpoints: its end pin height remains slightly above 80 m while the actual post-burn pin
is already below it. The next step consequently starts below the gate and never captures.
The correction tests the end pin with post-burn mass properties, then interpolates fuel
and recomputes COM at the crossing. Capture limits and structural collision checks remain
unchanged. Regression tests cover safe, open-arm and offset crossings with both signs of
COM shift, including premature crossings that must remain flying.

The frozen policy 3999 now catches both reproduced cases (27/32 total). A matched
512-start stochastic replay of model 3949 catches 394 instead of 390, with four gains and
no losses. These are physics-correction effects, not policy learning improvements.
Evidence and scripts are in the student run's `catch-probe/policy_3999`,
`catch-probe/policy_3999-fixed-mass-crossing`, and
`learning-audit/model_3949-fixed-mass-crossing` directories. Earlier results in this log
used the old crossing detector; corrected-physics training must be identified separately.


### Corrected-physics 700 m promotion

`catch_plane_fixed_local/2026-11-15_22-55-59` resumes model 4289 with the catch-plane
correction, preserving the optimizer, curriculum and frozen actor/critic normalizers.
The separate bridge continuation, `catch_bridge_plane_fixed_local/2026-11-15_22-56-45`,
resumes model 4419 at its 650 m stage; both retain the original final re-entry target.
The main run passes original stage 16 (700 ±50 m, cold engines, tilt 0–0.8 rad,
descent 45 ±5 m/s) at iteration 4604. Fresh immutable passed-stage holdouts catch
913/1,024 deterministic starts (seed 822032) and 436/512 stochastic starts (seed 822033).
These are below the 90% training gate; promotion does not establish robust mastery.
Training proceeds to original stage 17, descent 60 ±5 m/s at the same altitude and tilt.

A prior counterfactual stage-17 probe of model 4389 catches 389/512 stochastic starts
(seed 819710), with initial critic bias +40.00 and explained variance 0.121. No smaller
velocity step is inserted: transfer is already substantial, but the critic needs monitoring
as it adapts. The matched critic comparison on the existing 700 m stage also distinguishes
sample variation from regression: on identical 512 starts (spawn 819720, actions 819721),
models 4389 and 4489 catch 443 and 445; initial-state RMSE improves from 77.54 to 74.22,
and explained variance from 0.336 to 0.381. The lower explained variance seen in separate
fresh audits is not by itself evidence that the critic deteriorated. Reports and scripts
are in the main run's `next-stage-probe`, `matched-critic-trend`, and `promotion-holdouts`.


### Faster-descent promotion and wider-tilt gap

The corrected main run passes original stage 17 (700 m, descent 60 ±5 m/s, tilt 0–0.8 rad)
at iteration 4764. Fresh passed-stage holdouts catch 907/1,024 deterministic starts
(seed 822034) and 440/512 stochastic starts (seed 822035). The first fresh audit after
training on this speed (model 4689, seed 819704) catches 433/512; initial critic bias is
+3.64, compared with +40.00 in the earlier counterfactual transfer probe. These use different
checkpoints and spawn seeds, so the comparison is descriptive rather than paired.

A counterfactual wider-tilt probe of model 4689 (original stage 18, seed 819730) catches
305/512 overall but zero of 153 starts above 0.8 rad. Of those hard starts, 150 exceed the
40°/s transverse angular-rate limit, typically near upright after about 4.09 seconds;
median absolute roll rate is only 0.025 rad/s. Aggregate initial critic explained variance
is 0.708: predicting which starts fail does not supply a successful turn-control strategy.
Direct 1.1 rad PPO remains active, with models 4789 and 4889 catching 321/512 and 323/512
fresh stochastic starts respectively.

Offline interventions retain ordinary actuator limits, collision checks and termination
rules. On policy 4779's 128-start exploration set (seed 819741), all 30 starts above 0.8 rad
tumble without intervention. The best delayed fixed pulse catches four of those starts;
a pitch-rate governor using current thrust and inertia catches at most two, mostly turning
some early tumbles into later crashes. Neither intervention is installed in training or
distilled. Scripts and outcomes are archived as `turn-braking-search` and
`turn-rate-governor-probe` in the main run.

The altitude comparison passes 650 m at iteration 4839: fresh holdouts catch 900/1,024
deterministic starts (seed 823036) and 428/512 stochastic starts (seed 823037). It moves
to 700 m but its model 4919 catches only 282/512 in the fresh stochastic audit. Its
checkpoints are preserved and this comparison is retired in favor of a more relevant
tilt-curriculum experiment; this is a resource-allocation choice, not an equal-budget
sample-efficiency conclusion.

`catch_tilt_bridge_local/2026-11-15_23-24-03` resumes main model 4769, adds 0–0.9 and
0–1.0 rad stages before original 0–1.1 rad, and trains for 2,000 updates (seed 819760).
Original stage 18 becomes stage 20, and original final stage 30 becomes stage 32.
Full simulator and reward dataclass equality is verified for both the restored 1.1 rad
stage and final re-entry target. The first resumed checkpoint preserves frozen actor and
critic normalizers exactly. Direct 1.1 rad training continues as the comparison; success on
the easier bridge stage must not be reported as success on the common harder task.

### Wider-tilt learning and exploration checks

On identical 512 starts and stochastic draws (spawn 819780, actions 819781), main-run
models 4789 and 5189 catch 333 and 339: 19 gains and 13 losses over 400 PPO updates.
This small net gain is not convincing evidence of broad improvement. Above 0.8 rad,
catches change from 0/135 to 1/135; all 99 starts above 0.9 rad tumble in both checkpoints.
Initial critic RMSE improves from 67.97 to 58.46 and explained variance from 0.717 to
0.786, while the critic remains weak within the difficult tilt band. Exact recurrent
replay matches saved actor and critic outputs. Actor cell-candidate saturation at spawn
is effectively unchanged (55.07% to 55.08%); saturation alone does not identify a cause.
The paired outcomes, finer tilt bins, recurrent reports and health snapshot are archived
under the main run's `matched-wide-tilt-progress`.

A torque decomposition on policy 4779's 128-start exploration set (seed 819741) shows
that engines continue accelerating the turn while aerodynamic torque opposes it. For
the 30 hard starts, median pitch rate reaches 0.55 rad/s at 3.5 seconds while engine
angular acceleration is still +0.32 rad/s². Instantaneous opposite-gimbal authority
could brake, but that bound ignores actuator lag. Some survivors subsequently reverse
too aggressively. Analytical pitch PD and governor-to-approach handovers do not solve
the catch; no controller is adopted. Evidence is in `turn-torque-probe`, `pitch-pd-probe`
and `governor-handover-probe`.

An inference-only exploration probe of policy 5189 uses 512 matched starts (spawn
819790, actions 819791). Only starts above 0.8 rad receive larger independent throttle
and pitch-gimbal deviations during the first six seconds. Baseline, 0.05, 0.1 and 0.2
raw standard deviations catch respectively 5, 6, 5 and 5 of 148 hard starts; baseline
standard deviation is about 0.02. Unmodified easier-start outcomes match exactly.
This does not support simply increasing independent noise, nor establish whether
training with different or temporally coherent exploration would help. Artifacts are
in `early-noise-probe`; both existing PPO experiments continue unchanged.

### Batch-size throughput and step accounting

A short benchmark with both existing jobs running measures approximately 28k, 45k,
69k and 114k training transitions/s for 512, 1,024, 2,048 and 4,096 environments.
The 4,096-environment case uses 13.5 GB total device memory. These are shared-GPU
throughput measurements with fresh benchmark policies, not learning-efficiency results.
`catch_large_batch_local/2026-11-16_00-08-48` starts a separate 4,096-environment trial
from main model 5719, retaining 128-step rollouts, optimizer settings, frozen normalizers
and all curriculum simulator/reward configurations. Compare added transitions and wall
time: equal update counts process eight times as much data in this trial.

This exposes a legacy counter bug: resume reconstructed total transitions as iteration
count times the new batch size, inflating this trial's initial console/TensorBoard axis.
Its `step-accounting.json` records the correction; model 5729 has 5,242,880 newly added
transitions. New checkpoints persist `total_env_steps`, and learning audits use that
restored count. Legacy checkpoints use their adjacent Hydra config's batch size when
available, falling back to the current batch for standalone files without configuration.
This fallback cannot reconstruct earlier batch changes; already affected checkpoints
need an explicit counter correction before resuming with the fixed code.

### Outside review, live coverage and the 0.9 rad bridge

An independent Astra review is archived under the main run's
`astra-independent-review-2026-11-16/REVIEW.md`, with a reproducible occupancy analysis.
In the matched 1,024-flight audit, starts above 0.9 rad constitute 17.8% of episodes but
only 3.0% of valid transitions because they terminate quickly. This estimates audit
occupancy, not live PPO gradient allocation. The review prioritizes testing heuristic
termination during training, measuring actual learning allocation, and discovering
complete successful hard-start trajectories for subsequent imitation.

`train/coverage` now measures actual rollout transitions by initial tilt and episode age
(first 2 s, 2–6 s, later), plus starts, completions and catches. It counts unfinished
episodes, preserves spawn labels across updates, and attributes terminal transitions
before assigning autoreset labels. Counts reconcile to each rollout batch. These are
sample counts, not actor/critic loss or gradient shares; those remain separate diagnostics.
Existing processes need a resume under the new code to emit these metrics.

The bridge run passes its 0–0.9 rad stage at iteration 6099 and advances to 0–1.0 rad.
Fresh passed-stage holdouts catch 916/1,024 deterministic starts (seed 824036) and
450/512 stochastic starts (seed 824037). Within 0.8–0.9 rad, catches are only 31/112
and 14/53 respectively. The aggregate gate therefore does not establish mastery of
the stage's hardest band. Preserve this distinction when comparing curricula.
