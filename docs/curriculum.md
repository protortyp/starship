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
| 3–5 | Flip, randomized descent, wind | gradual flip curriculum training; full belly-flop unsolved |
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
