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
| 3–5 | Flip, randomized descent, wind | 700 m / 0–0.8 rad / 60 m/s gate passed; direct/bridge PPO continues. Joint student now catches 61/673 fresh hard starts above 0.9 rad and 701/1024 across stage 18; stochastic/critic checks precede isolated PPO continuation. Hard-range mastery and full belly-flop remain unsolved |
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

### Historical attitude-limit telemetry

`sim.terminate_on_attitude_limits` defaults to true, preserving the existing training
task. Setting it false permits recovery beyond the heuristic tilt/transverse/roll
limits; actuator dynamics, collision checks, capture requirements, bounds and timeout
are unchanged. This is a different experimental flight rule, not proof of success
under the original limits. The first ablation must hold its stage, reward and optimizer
settings fixed and compare actual catches under both rules.

Every spawn and physics substep records historical violations independently of the
termination switch. Exported statistics include cumulative and first-event cause bits
(1 tilt, 2 transverse rate, 4 roll), first time/altitude, and peak body transverse rate.
The metadata survives continued flight and terminal autoreset. Capture poses follow
the simulator's existing interpolation, while time/angular velocity retain the
full-substep convention; timing precision is one physics step, not an exact continuous
crossing time. Live metrics distinguish catches with/without historical violations.
`limits/violated_share` uses all completed episodes; individual cause shares use only
violating episodes. Learning audits export the per-episode evidence and explicitly
record whether heuristic termination was enabled.

### Exact diagnostic branches

Before testing alternate controls, verify unchanged continuation with
`uv run python -m starship.train.branch_probe POLICY --stage 18 --episodes 32
--branch-step 20 --stochastic --device cuda:0 --output DIR` (one shell command).
The probe saves a full snapshot, replays complete episodes twice, and requires exact
observations, actions, rewards, done masks and terminal outcomes. It includes actuator
state, simulator random streams, reward bookkeeping, global random streams and
recurrent policy memory. The snapshot API also supports critic memory.

Snapshots require identical model parameters/normalizers, configurations, device and
batch size. Use evaluation-mode models in a dedicated diagnostic process: restoration
also rewinds process-global RNGs. These are trusted local Torch artifacts, not safe
inputs from strangers or training-resume checkpoints. Optimizer/rollout state and
rebatching are not supported. Passing this replay check establishes a reproducible
starting point for control interventions; it does not demonstrate recoverability.

### Capture constraints and exact-branch search (2026-11-16)

Learning audits now save terminal capture measurements and signed normalized excess
for each kinematic gate: horizontal pin alignment, pin height, downward/upward pin
speed, horizontal pin speed, tilt, angular rate and arm angle. Pin velocities include
the rotational lever-arm term at both connectors. Positive residuals violate a gate;
the JSON summary uses a 1e-5 residual tolerance for floating-point comparisons. These
measurements do not override swept contact failures or prove that a descending
catch-plane crossing happened. `CaptureMetrics(sim).sample()` reads terminal buffers;
select newly ended lanes before another step overwrites them.

The matched equal-transition batch experiment added 52,428,800 transitions to the
same source checkpoint. On 1,024 matched stage-18 starts, 512 environments caught
745 and 4,096 environments caught 751 (19 gains, 13 losses). Above 0.8 rad the counts
were 21 versus 19 of 272; above 0.9 both caught zero of 186. Larger batches have not
resolved the hard region. Evidence: the counted large-batch run's
`equal-transitions-427M/comparison.json`.

Two offline exact-branch CEM screens also found no catches. The first searched
256 candidates for 12 generations from a 0.953-rad start at t=2 s, varying throttle
and one gimbal axis over eight seconds. Remaining reward improved from -166 to -142
but the best trajectory still crashed badly oriented. The second searched 512
candidates for 30 generations from a different 0.949-rad start at t=0, varying throttle
and both gimbal axes over ten seconds. Its terminal geometry/speed proxy improved,
but it still missed the pins by about 75 m. That second proxy used center-of-mass
speed, not the newly added exact pin metrics. Both used relaxed heuristic attitude
termination and unchanged physical capture/contact rules, with exact baseline and
winning-intervention replay. Neither supplies demonstrations or evidence of learned
success. Artifacts: primary run `exact-branch-cem-095` and
`exact-branch-geometry-cem-095`. Future search must solve the complete catch, including
pin motion and arm closure; better proxy scores alone are insufficient.

### Following successful trajectories into harder starts (2026-11-16)

Starting near a feasible maneuver worked better than the two cold searches above.
For one fixed physical start, small throttle/two-axis-gimbal residuals over the first
eight seconds produced full catches at each 0.025-rad increment from 0.75 through
0.95 rad. Each winner replayed exactly, finished in 36–46 s, and never crossed the
original attitude limits. Only initial orientation changed between angles; translation,
velocity, angular rates, fuel and random streams matched. The attempted jump from
0.95 to 1.0 rad found no catch in 25 generations. This is offline trajectory
optimization with a frozen policy suffix, not a learned controller or full entry.

The fixed 0.95-rad correction caught 5/512 independent starts at 0.95±0.01 rad,
versus 0/512 without correction. Both strict and relaxed flight limits gave the same
five catches; none violated the strict limits. This demonstrates some transfer but
very weak robustness. Evidence: primary run `exact-catch-continuation/heldout`.

A subsequent search selected 32 independent, unmodified stage-18 starts with initial
tilt 0.8–1.0 rad and optimized 32 candidates per start for 30 generations. It used
the original termination, contact and capture rules throughout, with exact complete
winner replay. The unchanged policy caught 1/32; interpolated prior corrections
caught 5/32; per-start optimization caught 19/32, including 6/15 above 0.9 rad versus
zero there for the unchanged policy. These are search-training outcomes, not heldout
policy scores. `multi-start-catch-search/demonstrations.npz` preserves full recurrent
histories, executed actions, valid masks, discounted returns and explicit failure
statuses; only the 19 successful episodes qualify as successful demonstrations.

An isolated imitation trial mixes those 19 trajectories with 128 successful easy
episodes from the source actor. It freezes observation statistics, uses full recurrent
sequences and evaluates the standalone actor without action corrections. Critic and
PPO optimizer calibration remain separate prerequisites for any PPO handoff. Existing
PPO runs are not replaced by this trial.

The first full-actor imitation trial (`catch_branch_distill_local/2026-11-16_01-10-34`)
was rejected. After 400 updates at 3e-6, its matched stage-18 development catches fell
from 342/512 to 321/512. The 0.85–1.0-rad development set improved from 3/512 to 13/512,
but remained zero above 0.9 rad; no saved student met the retention criterion. The
selection therefore retained update zero. Its two fresh evaluations are the same
source policy, not evidence of a student gain. A follow-up isolates final throttle/
gimbal output rows and gives the first eight seconds five times the imitation weight;
it must still pass standalone policy evaluation before any handoff.

The restricted-output trial (`catch_branch_head_distill_local/2026-11-16_01-12-22`)
also failed retention and produced no catches above 0.9 rad; selection again retained
the source actor. An audit on the exact demonstration starts found only 3/19 catches
for the full-actor student at update 400 and 0/19 for the output-only student at update
1,000, versus 1/19 for the source. Even on recorded teacher histories, prefix action
errors remained substantial: full-actor tanh-space MSE for throttle/two gimbals was
0.0229/0.0415/0.0125, versus source 0.0314/0.0633/0.0314. Source suffix errors were
approximately 1e-8, supporting correct history alignment. This is a fitting/closed-loop
reproduction problem before it is merely a generalization problem. Evidence:
`multi-start-catch-search/imitation-audit`. Do not calibrate a critic or replace PPO
with either rejected student.

A bounded smoothing/amplitude search retained all 19 successful demonstrations while
reducing the control-variation objective for nine. Every selected trajectory replayed
exactly; six successes remain above 0.9 rad. The successful subset's median objective
decreased only from 0.1584 to 0.1471, so this is modest label cleanup, not a policy
improvement. Artifacts: `smooth-catch-demonstrations`; failed candidates remain marked
as failures and must be excluded from successful-demo training.

### Learning the hard maneuver while retaining the approach (2026-11-16)

A single-trajectory capacity test changed the diagnosis. With frozen observation
statistics, soft output targets and extra weight on the first eight seconds, 25 updates
at 1e-4 taught the actor a hard catch without action overrides. The recorded teacher
finished in 56.1 s; the learned actor took a different successful path in 37.5 s.
On independent starts it caught 43/699 above 0.9 rad, versus zero for the source, but
full stage-18 catches collapsed from 719/1024 to 27/1024. It is a useful specialist
teacher, not a replacement policy. Evidence: `catch_single_fit_local/2026-11-16_01-18-03`.

That specialist generated 256 autonomous successful flights from 4,096 new starts
at 0.85–1.0 rad, including 150 above 0.9. Joint soft-logit imitation combined them
with 512 source-policy successes at 0–0.7 rad, using full recurrent histories and
unchanged normalizers. After 1,000 updates, hard development catches rose from 4 to
89/1024 (59 above 0.9), while full-stage catches fell from 724 to 664/1024. The loss
was concentrated in an omitted retention range: at 0.7–0.8 rad, catches fell from
75 to 30 of 93. This identified a concrete coverage gap in the imitation data.

Two matched continuations started from that same actor with identical seeds/settings.
The treatment added 595 successful source-policy flights from 0.7–0.85 rad; the
control reused the original data. After another 1,000 updates, treatment caught
709/1024 full-stage and 127/1024 hard development starts, versus control 673 and 86.
The selected treatment then caught 701/1024 on untouched full-stage starts, versus
707 for the source, and 116/1024 on untouched hard starts, versus 3. Above 0.9 rad
within the hard set, it caught 61/673 versus zero. These are deterministic standalone
policy results under the original flight/contact/capture rules, not helper-assisted
rollouts. Evidence: `catch_joint_mid_fit_local/2026-11-16_01-24-04`; paired control:
`catch_joint_continue_fit_local/2026-11-16_01-24-04`.

The candidate's PPO artifact preserves the source iteration/transition count and
action standard deviations, replaces only actor weights, and clears stale actor Adam
moments. Its inherited critic must be fitted to new complete on-policy returns and
checked on fresh episodes before an isolated PPO run. Existing baselines remain live;
no end-to-end checkpoint is validated for merging or Git LFS yet.

### Completed heuristic-termination ablation

The matched 2,048-environment runs each added 52,428,800 transitions from source 5719
at fixed stage 18. Final matched stochastic evaluation caught 359/512 for control and
358/512 for recovery training. Evaluating either actor with relaxed termination did
not add catches, and neither caught after a historical attitude-limit violation.
The critic's initial-state explained variance was about 0.780 for control under its
own rules and 0.825 for recovery under its own rules; this did not imply improved
catch behavior. This bounded experiment found no recovery benefit; it does not prove
physical impossibility. Evidence: termination-control run
`paired-evaluations/model_5919/comparison.json`. Both bounded jobs completed normally.

### Rebuild ground flight before catch transfer

The user confirmed the original `land3d_v8_lam098`/`attitude_v1` weights are gone.
Local stunt checkpoints are different policies, not substitutes for the documented
98–99% landing baseline. The catch lineage started from fresh initialization and
catch demonstrations; it did not preserve that old controller.

Following the user's requested sequence, `--config-name landing_rebuild_local`
rebuilds ground landing with the current 32-input/12-action interface, position
denominator 30 m, vehicle and physical landing limits. The arm output is inactive;
catcher input channels are zero as defined by the existing ground encoder. Fresh
observation normalizers learn during this run; no encoder definitions change.
The five standard landing stages cover lit-engine burns, cold starts, low belly-flops,
3 km belly-flops, and randomized 1.5–5 km descents. The local preset uses 2,048
environments, 128-step rollouts and 16 minibatches; standard landing PPO settings
otherwise apply. It is a new initialization, not an exact reproduction of the lost run.

The decision to rebuild follows a deterministic screen of the main catch actor
(`policy_6929`) and the joint student: both landed 0/512 in each of five ground
profiles (current 700 m starts, landing stages 0, 1, 2 and 4) with current observation
scales. This establishes lack of immediate ground transfer, not that their weights
could never be adapted. Evidence: primary run `ground-transfer-current-encodings`.

Before catch fine-tuning, require at least 95% ground landings on fresh deterministic
and stochastic randomized-stage evaluations, not just the 90% curriculum gate.
Archive passed-stage weights, resolved configuration, hashes and validation results on
the separate checkpoint volume. Then initialize a short upright catch approach from
the validated ground actor, retain ground-flight evaluations during adaptation, and
extend the catch start backward through real descent states. End-to-end capture and
the original final checkpoint/LFS/merge requirements remain outstanding.

#### Early rebuild audit: allow braking to develop before changing the recipe

Run `landing_rebuild_local/2026-11-16_01-40-19` is still at stage 0 after 100
updates, with no landings. Complete stochastic 512-episode audits show initial-state
critic explained variance improving from -1.633 at update 25 to 0.527 at 50 and
0.904 at 100 (RMSE 8.44 → 1.98 → 1.08). Thus the early low-feature-rank warning
did not justify changing the critic. Audited inputs are finite; the largest normalized
channel p99 is 3.82 at update 100. Median impact speed first worsened from 78.74 to
81.44 m/s, then improved to 62.74 m/s. Updates 25/50 use matched seed 830025;
update 100 uses fresh seed 831099, so the latter is a distributional trend, not a
matched-start comparison. There were no attitude-limit violations in these audits.

A separate checkpoint-50 diagnostic (512 matched starts, seed 830050) forced engine
decisions while retaining the stochastic policy's other actions. Default-policy,
engines-off, and first-second-engines-on returns were -48.418, -48.337 and -50.416;
median impact speeds were 82.48, 85.56 and 70.07 m/s. All crashed. With fuel cost
removed only for evaluation, returns became -47.883, -48.116 and -48.167, with the
same aggregate trajectory outcomes. Fuel cost explains much of the short burn's
penalty, but removing it alone does not make that intervention outperform the policy.
Holding engines on throughout caused 509/512 attitude terminations. These are
interventions, not learned successes, and do not establish a reward fix. Keep the
baseline training unchanged: the historical recipe first produced landings around
250 updates, and braking is now improving without intervention. Update counts are
only a rough historical reference, not equal-data experiments.

Evidence lives in this run's `early-audit-{25,50}`, `learning-audits/model_99`, and
`reward-probe-50{,-no-fuel}` directories. Checkpoints 49/99, configurations and
reports were copied with verified hashes to the separate checkpoint volume. PR 67
now follows this ground run every 100 updates, using full diagnostic episodes with
ground-specific outcome labels; the retired catch publisher cannot overwrite it.

At update 150, the same audit seed as update 100 produced the first landing
(1/512); median downward impact speed fell to 6.96 m/s. Initial critic EV was
0.135 (RMSE 9.17) as episode returns became more variable; this is not evidence
that the earlier good critic fit transfers automatically into the new outcome range.
New ground-landing diagnostics sample terminal physics buffers and report each
contact limit separately. Among these 512 contacts, 491 exceeded downward speed,
415 horizontal speed, 55 tilt, 261 angular rate and 405 pad radius (overlapping
counts). Pad miss uses the body base and terminal fuel-dependent center of mass,
not center-of-mass position. Non-contact terminations are excluded from the counts.
The audit exports measurements and signed excesses for all episodes alongside the
existing trajectories. Evidence: `landing-constraints-150`; six simulator tests cover
successful touchdown and each failure limit with auto-reset enabled.

#### Ground rebuild: stochastic/deterministic gap and critic uncertainty

At update 200, a repeatability probe held 128 spawn states and initial values fixed
across eight action seeds. Within-start return variance was 347.42; estimated
between-start variance was 25.37. The estimated single-rollout EV ceiling was 0.068,
versus the critic's 0.059; EV against eight-repeat mean returns was 0.319. This is a
finite-repeat estimate with rare successes, not a proven ceiling or evidence that
the critic is optimal. It does not justify increasing critic learning rate from low
initial EV alone. Evidence: ground run `return-uncertainty-200`.

Matched checkpoint-225 evaluations (512 starts, seed 832225) landed 0 episodes with
deterministic actions and 71 with stochastic actions. Diagnostic hybrids sampled all
heads before replacing selected outputs: binary-only sampling landed 101; continuous-only
sampling landed 18. Both endpoint outcome arrays reproduced the original audits, and
all initial observations matched. The deterministic third engine stayed off, while
its stochastic duty was 0.122. This implicates binary switching in the mode gap for
this checkpoint, not a unique causal explanation or a validated deployment fix.
Evidence: `action-modes-225/hybrid`, including the reproduction script.

At update 300, another matched-start comparison (seed 831299) landed 407/512
stochastically and 22/512 deterministically. Ground-contact angular-rate violations
were 12 versus 370. Deterministic attitude normalization exceeded magnitude 10 in
17 episodes (9 crashes, 8 attitude terminations), with no initial outliers; 473
failures occurred without this outlier. Thus it does not explain most failures.
Deterministic evaluation subsequently reached 26.6% at update 325 and approximately
37% at 350 without changing the recipe. Stage 0 remains unsolved. Keep reporting
both action modes and keep the existing deterministic evaluation rule. Evidence:
`action-modes-300.json` and `learning-audits/model_299-deterministic`.

The bounded direct-catch baseline ended normally at update 7290: final independent
stochastic audit 373/512 caught, but 0/92 above 0.9 rad initial tilt. The large-batch
comparison ended at update 6720: 418/512 overall, 13/100 above 0.9 rad. These are
separate evaluations, not an equal-data causal comparison. Both final models,
configurations and reports are hash-verified on the separate checkpoint volume;
neither is end-to-end validated. The queued direct-catch extension was cancelled
to prioritize the ground-first rebuild. The ground audit watcher now automatically
backs up each 100-update checkpoint and report, in addition to passed-stage backups.

#### Ground rebuild: actuator response and deterministic student

The matched entropy experiment resumed checkpoint 499 for 150 updates with identical
512-environment batches and seeds. Final independent 512-start evaluations landed
288 deterministically / 502 stochastically for entropy coefficient 0.002, versus
430 / 504 for -0.01. This penalty affects all action heads, not just engines.
Both trials fluctuated substantially; the penalty candidate continues separately
with the standard ground curriculum and 2,048 environments. Evidence:
`landing_entropy_control/2026-11-16_02-22-35/comparison`.

At checkpoint 624 the mode gap has changed: on 512 matched starts, ordinary
deterministic inference landed 392, binary-only sampling 399, continuous-only
sampling 479, and full sampling 503. A deterministic engine probability accumulator
landed 401; repeating with a different action RNG produced identical traces.
Replacing continuous means by expected squashed commands landed 395 with sampled
binaries, while sampling just gimbals (plus binaries) landed 499. Thus the earlier
binary-switching diagnosis does not explain this later checkpoint; averaging the
squashed command alone does not recover the sampled behavior.

A diagnostic wrapper instead integrates each Gaussian gimbal command through the
existing ten actuator lag/rate-limit substeps, then inverts that mapping to match
the expected final gimbal angle with one legal constant command. Other heads use
ordinary deterministic outputs. It uses privileged actuator/filter state and is
not an unassisted neural policy. On the matched 512 starts it landed 501; changing
the action RNG gave identical action/status traces. A preregistered fresh 2,048-start
evaluation landed 1,998 (97.56%). Physics, termination and landing criteria are
unchanged. The 512-node quadrature passed the predeclared 0.0002 rad convergence
tolerance (maximum 256-vs-512 difference 0.0001501 rad on the fresh evaluation).
The lower-resolution 128-node version failed that tolerance and is not the accepted
result. Actual actuator state prediction agreed within 5.4e-8 rad; endpoint inversion
does not also match filtered target or integrated angle, which remain documented
limitations. Evidence: the ground run's `actuator-response-probe-625-q512` and
`actuator-response-heldout-625`; checkpoint, configuration, scripts and results are
hash-archived under `ground-response-625` on the separate checkpoint volume.

A separate 1,024-start collection (997 landings, seed 835900) supplies complete
recurrent episodes and soft source-policy targets, replacing only the two gimbal
targets with response-matched commands. Student run
`landing_response_distill_local/2026-11-16_02-50-51` freezes observation normalization
and fits the actor at learning rate 1e-5. All demonstrations, including failures,
remain in training. Unassisted deterministic development evaluations reached
248/256 at update 200 and 250/256 at update 1000. Checkpoint 200, the first scheduled
evaluation above 95%, was selected for a separate 2,048-start deterministic and
stochastic audit before inspecting those results. That fresh audit (seed 836900)
landed 1,988/2,048 deterministically (97.07%) but only 1,749/2,048 with the source
exploration noise (85.40%). Deterministic failures were mostly pad misses (58);
stochastic failures included 217 pad misses and 76 angular-rate violations, with
overlap. Initial critic EV was 0.042 / -0.006 and bias -8.01 / +16.91 respectively.
The inherited critic therefore needs calibration for the selected behavior before
PPO continuation. A matched noise probe changes only the two gimbal standard
deviations to 1, 0.5 or 0.25 times their inherited values; deterministic outputs
remain unchanged. These results concern only upright stage 0; cold starts,
belly-flops, randomized descents and catch transfer remain unvalidated.

#### Ground student: validated upright landings and guarded cold-start handoff

The matched gimbal-noise sweep landed 440/512 at inherited noise, 498/512 at half
noise, and 510/512 at quarter noise. A subsequent fresh 2,048-start audit of the
quarter-noise student landed 1,983 deterministically (96.83%) and 2,028 stochastically
(99.02%). These are normal neural-policy actions, without the actuator wrapper.
Actor-frozen critic calibration selected update 600 on episode-disjoint validation;
on the fresh stochastic audit, initial EV improved from 0.010 to 0.064 and bias from
-3.79 to -0.63. The improvement is modest, not evidence of an accurate critic on
every trajectory. Actor/normalizer weights and matched rollout observations, returns
and statuses remained bitwise unchanged. Evidence: the student's `quarter-heldout-200`
and `quarter-calibrated/heldout`; validated weights and reports are hash-archived in
`ground-response-student-20261116` on the separate checkpoint volume.

Run `landing_response_ppo_smoke/2026-11-16_02-56-24` then performed 50 fixed-stage-0
PPO updates: 512 environments, actor LR 1e-6, critic LR 1e-5, frozen observation
normalization, evaluation every five updates, and rollback after a 5% relative
deterministic regression. Two evaluations fell to about 92% and triggered rollback.
The final checkpoint therefore restores the earlier best actor (iteration 634),
not the uninterrupted result of 50 updates. A fresh matched 2,048-start comparison
landed 2,002/2,029 for the source and 1,999/2,026 for the restored candidate
(deterministic/stochastic). The candidate lost seven and gained four deterministic
landings, and lost eight/gained five stochastic landings. Both normalizers remained
bitwise unchanged; actual gimbal standard deviations stayed about 0.1066/0.1063.
This demonstrates recoverability through rollback, not stable PPO improvement.
Evidence: the smoke run's `retention/summary.json` and separate-volume backup.

The source already transfers to cold engines at spawn with 476/512 deterministic
and 464/512 stochastic landings. Run `landing_cold_retained_local/2026-11-16_03-04-53`
starts from the validated smoke candidate and fixes training at stage 1 for 100
updates. Every five updates it evaluates cold-start deterministic performance and
both action modes on stage 0; a stage-0 rate below 95% saves the rejected checkpoint
and restores the accepted best. Thus current-stage best selection cannot silently
discard earlier flight skills. The restoration path was exercised before training:
actor weights restored bitwise, counters preserved, correct stage/rates/frozen
normalization verified. Initial evaluations were 93.75% cold deterministic and
96.88%/98.83% retained stage 0. Promotion is disabled for this bounded experiment;
fresh 2,048-episode evaluations must reach 95% in both modes on both stages before
advancing. Belly-flop, randomized descent and catch success remain outstanding.

The initial braking diagnostic had retained the catch-relative height reference
when the catcher was disabled. Ground reports now use the fuel-dependent body base
above ground and label the reference explicitly; catch reports retain the pin-center
height above the support. Older ground braking-margin fields are superseded, while
trajectory outcomes and terminal landing measurements are unaffected. Four simulator
regression cases cover ground/catch targets, tilt and different header fills.
With corrected heights, the cold-transfer deterministic failures started at median
149.3 m and -57.9 m/s, versus 279.9 m and -48.9 m/s for landings. Failed trajectories
already requested about 2.94 engines and 97.3% throttle during their first two seconds.
These are observational associations, not a feasibility bound or proof of a controller
fix. Evidence: the student's `cold-transfer/failure-analysis`.

#### Cold-start critic mismatch and calibrated continuation

The first retained cold-start run completed its 100 updates without meeting the
promotion requirement. Its accepted best actor (iteration 744) validated on 2,048
fresh episodes per mode at 97.31%/99.51% for stage 0 and 93.26%/89.16% for stage 1
(deterministic/stochastic). Several updates were rejected or rolled back; the final
actor is the accepted earlier best, not an uninterrupted 100-update improvement.
Weights, configuration, reproduction script and validation summary are archived under
`ground-cold-retained` on the separate checkpoint volume. PR 67 now follows cold-start
training, showing complete fixed-seed episodes every 25 updates and at completion.
The publisher uses saved transition counters across batch-size changes and labels
the displayed rate as the latest training evaluation; it can precede a rollback.

A matched 512-start stochastic audit of checkpoint 734 showed only 466 landings
versus 464 for the pre-cold source. Initial critic bias remained -39.30, from -45.87
at transfer; RMSE was 54.83 versus 60.46. Source failures already requested nearly
full available thrust, and the critic strongly mispriced initial cold states. This
supports testing calibration before changing the controller or reward, without
establishing that critic error is the sole cause of failed landings.

Actor-frozen calibration used 512 complete stochastic episodes from each of stages
0 and 1, with episode-disjoint validation. The first fit assigned half its loss to
the first 20 steps: it improved cold initial RMSE from 57.79 to 38.48 but worsened
warm initial RMSE from 5.08 to 7.74, failing the preset retention limit. It was rejected.
A controlled second fit changed that window to the initial state only, retaining
the dataset, seed, 2,000 updates and 1e-4 critic learning rate. Warm initial RMSE was
5.28; cold initial RMSE was 21.59 and EV 0.699, passing the same acceptance rule.

Independent validation replayed the critic over the previously separate 2,048-episode
stochastic cohorts for each stage, checking the source replay against recorded values
within 0.001. Cold initial bias improved from -41.62 to -0.30, RMSE from 59.15 to
27.48, and EV from 0.214 to 0.664. Warm initial/all-state errors passed the preset
limit of at most max(10%, 2 return units) deterioration. Actor weights and both
normalizers remained bitwise unchanged. Evidence: the cold run's
`joint-critic-calibration`, `joint-critic-initial` and `learning-audit-735`; selected
weights and independent report are archived under `ground-cold-critic-initial`.

Run `landing_cold_calibrated_retained_local/2026-11-16_03-22-08` now tests another
100 cold-stage PPO updates with that critic and the same retention guard. Its source
metadata carries the completed parent's iteration 774 and true transition count,
while weights come from the selected calibrated checkpoint. It remains fixed at
stage 1 and requires fresh >=95% validation in both modes on both stages before
advancing. This is a continuation experiment, not a matched causal comparison of
calibrated and uncalibrated training.

#### Clearance bridge and gradual tilt frontier

The calibrated cold continuation ended without passing the added cold gate: fresh
2,048-episode results were 97.22%/99.22% on warm starts and 91.60%/88.13% on the
original cold distribution (deterministic/stochastic). Better initial critic accuracy
alone did not establish successful cold-stage learning. A separate matched intervention
forced all engines on at near-full throttle for one or two seconds: 472 and 476 of
512 landed, versus 476 normally. It rescued some low starts but lost other landings;
no forced-burn controller was adopted.

The outside review identified a tight synthetic starting envelope. Spawn altitude is
COM altitude; initial COM is 23.82 m above the base, making the minimum upright base
height about 126 m. Cold throttle starts at 0.4 and takes 1.5 s to reach full throttle,
in addition to the 2 s spool ramp. An upright maximum-burn reference using actual
engine ramps, fuel-dependent mass/COM and sampled axial aerodynamics had negative
stopping clearance for 23/512 starts, all policy failures; a 40,000-cell height/speed
grid gave 4.95% reference failures. This is **not a global reachability bound**:
tilting, non-axial aerodynamics, clearance changes and lateral control are not fully
represented. It cannot certify that those starts are impossible.

A matched counterfactual changed only nominal COM altitude from 300 to 400 m,
retaining the same ±150 m spread, speeds, orientation, rates, tanks, actuators and
mirror flags. Initial non-height state fields matched bitwise; the height shift was
100 m within 3.1e-5 m. The unchanged source landed 945/925 of 1,024 on the original
profile and 1,006/1,016 on the raised profile (deterministic/stochastic). Fresh raised
profile validation landed 2,014/2,025 of 2,048. This establishes a useful clearance
bridge, not impossibility or mastery of the original cold profile.

The extra >=95% requirement on the original intermediate cold distribution is
therefore being replaced by the validated clearance bridge. The original profile
remains an unfiltered, separately reported stress set. Final randomized-ground
acceptance remains >=95% in both modes, and end-to-end catch validation/LFS/merge
requirements are unchanged. No physics, actuator or terminal-success rule was relaxed.

| Current profile | Evidence/status |
| --- | --- |
| Warm upright starts | Reliable retained source; checked throughout adaptation |
| Raised cold, 400±150 m, pitch ±0.1 rad | Unchanged ground student passed fresh 2,048-per-mode bridge validation |
| Raised cold, pitch ±0.2 rad | Coherent calibrated source: 497/506 of 512; new band 246/252 of 257 |
| Raised cold, pitch ±0.3 rad | Aggregate retained; narrow 0.2–0.3 band passed one cohort but failed the next with identical weights |
| Raised cold, pitch ±0.4 rad | Bounded 200-update expansion running; initial new-band deterministic success 78.68% |
| Original low-clearance cold starts | Reported stress set; not claimed solved |
| Full belly-flop, randomized descent, catch | Not yet validated |

Direct transfer to the standard 900–1,500 m belly-flop stage landed 0/512 in both
modes, with 511/507 attitude terminations. That jump changes several conditions
simultaneously. A first-step LSTM audit also found normalized feature p99 magnitudes
around 70–76; input/forget/output gate saturation fractions were 18.8%/13.1%/14.8%
at absolute-logit threshold 8, and candidate saturation was 44.5% at threshold 4.
Warm and raised-cold starts had approximately zero saturation by these definitions.
Reconstructed hidden outputs matched the actual LSTM within 2e-7. This supports a
gradual transition, not a claim that normalization alone explains every failure.

Run `landing_tilt_frontier_local/2026-11-16_03-50-52` holds raised-cold height/speed
fixed and widens only pitch noise to ±0.3 rad after the ±0.2 screen succeeded. It
uses the coherent accepted actor/critic, frozen statistics and conservative learning
rates for 100 updates. Best selection uses the minimum of aggregate success and
success in the new 0.2–0.3 rad band. Warm and raised-cold ±0.1 retention are checked
in both modes every five updates; any rate below 95% restores the accepted checkpoint.
Frontier regression uses a separate 10% relative, two-evaluation guard. Fresh final
validation checks both retained profiles, the frontier and its new band; the original
cold profile is reported separately. Later height, speed and tilt extensions will be
chosen from measured transfer, with controlled normalization refreshes as needed.

Evidence: ground student's `cold-early-burn`, `cold-clearance-probe` and
`clearance-next-stage`; the retained cold run's `tilt-frontier-screen` and
`tilt-frontier-next`. The burn/clearance experiments and explicit reference assumptions
are archived on the separate checkpoint volume. PR 67 now follows the tilt frontier.

#### Tilt retention and throttle regression

The first 100-update frontier run selected iteration 924. Fresh validation retained
warm landings at 97.36%/99.22% and raised-cold landings at 99.02%/99.41%
(deterministic/stochastic, 1,024 each). Aggregate frontier success was 95.21%/97.95%
on 2,048 episodes, but the new 0.2–0.3 rad band reached only 609/643 of 677
(89.96%/94.98%). It therefore did not pass. The original low-clearance stress set
scored 92.97%/90.63%. A bounded 200-update continuation starts from this accepted
checkpoint, retaining the same guards and frozen normalization; final validation
uses fresh seeds. It is not a promotion to belly-flop or catch training.

A matched 512-start audit compares the pre-frontier source, accepted iteration 924,
and rejected continuation iteration 1029, with identical initial observations and
stochastic action seeds. In the new tilt band, accepted success changed from
160/174 to 163/177 of 188: five gains and two losses in each action mode. This is
small evidence of improvement, not convincing mastery. Warm deterministic success
was 501/512 before frontier training, 498/512 accepted, and 409/512 rejected;
rejected stochastic success remained 495/512. Normalizer state matched exactly and
continuous action standard deviations changed by less than 0.0001 in absolute units.
The regression was not a changed observation scale or a large exploration-noise jump.

The rejected warm policy exceeded touchdown downward-speed limits in 82 episodes,
versus zero accepted; median terminal vertical velocity changed from -1.397 to
-1.716 m/s. Neither had tilt or angular-rate violations. Initial critic bias on
stochastic trajectories stayed modest (-2.68 accepted, -2.08 rejected), while on
deterministic trajectories it changed from -7.06 to +14.22. Training explained
variance alone therefore does not establish deterministic-controller reliability.

A diagnostic command intervention ran both actors on the same complete recurrent
observation history, keeping all physics and terminal rules fixed:

| Deterministic warm controller, 512 matched starts | Landed | Downward-speed violations |
| --- | ---: | ---: |
| Rejected actor, reproduced baseline | 409 | 82 |
| Rejected actor with accepted throttle command only | 497 | 0 |
| Rejected actor with accepted engine switches and throttle | 498 | 0 |
| Accepted actor with rejected throttle command only | 431 | 58 |

Replacing throttle alone rescued 90 episodes and lost two against the rejected
baseline; the reverse intervention lost 67 against the accepted baseline. This
isolates throttle behavior as a major contributor on this cohort, without claiming
that only final-layer throttle weights caused it. These two-actor interventions are
diagnostics, not deployed policies. The continuation remains bounded; repeated
regression motivates actor-only retention on complete successful warm/raised-cold
histories, with special attention to throttle behavior. The prepared demonstration
bank is not on-policy PPO data and has not yet been enabled as a training loss.

A separate belly-flop counterfactual disabled attitude-limit termination for the
unchanged ground source. Both modes still landed 0/512; all episodes eventually
crashed, and trajectories matched exactly before the original cutoff. Removing the
cutoff alone does not rescue this source, so no termination-rule change was adopted.

Evidence: `landing_tilt_frontier_continued_local/2026-11-16_04-08-53/` contains
`matched-regression-audit` and `throttle-regression-probe`, including source hashes,
reproduction scripts, complete trajectories and paired outcomes. Both are archived
under the corresponding `tilt-matched-regression` and `tilt-throttle-probe` directories
on the checkpoint volume. The belly-flop counterfactual is archived under
`flip-recovery-probe/2026-11-16_02-50-51`. PR 67 follows the continuation with complete
first-episode GIFs; it remains draft pending end-to-end validation.

#### Throttle retention pilot and rollout credit

The 200-update continuation did not solve the frontier. Its selected checkpoint's
fresh harder-band results were 597/672 deterministic and 620/672 stochastic
(88.84%/92.26%); warm and raised-cold retention still exceeded 95% in both modes.
Most deterministic angular-rate failures were longitudinal roll: 37 of 38 in the
harder band, alongside 44 pad misses (overlapping counts). Initial lean direction
mattered: negative/positive `up_body_3` groups landed 335/343 and 262/329 respectively.
Initial LSTM saturation was zero under the diagnostic thresholds; normalized channel
p99 magnitudes were at most about 9. This does not assess later recurrent saturation.

Exceeding a touchdown rate limit during flight is not itself an error. Of 597
successful harder-band episodes, 343 crossed the longitudinal touchdown-rate limit;
the median time above it was 1.3 seconds, versus 17 seconds among angular failures.
Simple controller interventions did not solve this: a fresh 512-start baseline landed
472; symmetric flap commands for the first five seconds landed 462, symmetry for the
whole flight landed 65, and proportional RCS roll damping landed 470. No such controller
was adopted. Full trajectories and reproduction scripts are in the continued run's
`roll-control-probe` and `frontier-failure-breakdown`, archived on the checkpoint volume.

A matched 50-update pilot compared unchanged PPO with four actor-only throttle replay
steps after every PPO update. Replay used 16 warm and 16 raised-cold complete histories
per step, explicitly equal profile loss weights, and an iteration-seeded local sampling
generator. It used the same Adam optimizer, including its existing momentum; the added
steps are part of the intervention, not a pure projection. Critic parameters received
no replay gradients. Diagnostics measured collateral changes to all action heads and
frontier histories. Correctness preflight verified unchanged critic/normalization/
recurrent rollout state and exact actor/optimizer restoration. Replay-bank episodes
are training data, never independent validation or fresh on-policy PPO trajectories.

Both arms started from the same checkpoint and seed, with equal added transitions.
Fresh paired evaluations verified bitwise-identical initial observations. After all
50 updates, warm deterministic success improved from 479/512 to 496/512, raised-cold
from 502/512 to 510/512, and harder-band deterministic success from 280/335 to 295/335
(control/replay). Harder-band stochastic success changed from 304/335 to 310/335.
However, the checkpoints selected by training evaluation did not show the same gain:
harder-band deterministic success was 601/676 control versus 591/676 replay, and
stochastic success 633/676 versus 632/676. This single-seed pilot supports a stability
benefit at the fixed training budget, not superior peak performance or frontier mastery.
The paired bootstrap intervals in the reports quantify episode uncertainty conditional
on these policies; they do not quantify variability across training seeds.

Credit assignment is the next bounded test. In 35/38 angular failures, the first
observed longitudinal rate crossing preceded touchdown by at least the 128-step
rollout; median lag was 173.5 control steps (17.35 seconds). Successful episodes also
cross early, so this does not identify the decisive wrong action. GAE stops at rollout
boundaries and bootstraps the critic. Increasing lambda alone cannot bridge that
boundary; conditional critic overprediction on failed episodes is not by itself
proof of miscalibration.

Run `landing_long_rollout_local/2026-11-16_04-41-14` changes 512 environments ×128 steps
to 256×256, keeping 65,536 transitions/update, 50 updates, lambda 0.98, rewards, learning
rates, throttle replay, source checkpoint, seed and retention guards unchanged. This
changes temporal context, recurrent backpropagation length and environment diversity
together. Both selected and final checkpoints are compared against the 128-step replay
arm on the same evaluation cohorts; promotion would still require fresh validation.
Evidence: the throttle replay run's `paired-comparison` and `final-paired-comparison`,
archived under `throttle-retention-pair`; credit diagnostics under the continued tilt
run's `frontier-failure-breakdown`. PR 67 follows the longer-rollout run and remains draft.


#### Longer rollout result and trace extension

The matched 256-step rollout result was mixed. Selected-checkpoint harder-band
success was 609/676 deterministic and 630/676 stochastic, versus 591/676 and 632/676
for 128-step throttle replay. The deterministic comparison gained 19 episodes and
lost one. At the equal 50-update endpoint, however, 256 steps scored 286/335 and
307/335 versus 295/335 and 310/335; deterministic gains/losses were two/eleven.
Both modes retained warm and raised-cold skills, but neither checkpoint passed the
frontier gate. Longer rollouts alone are therefore not established as an improvement.
These are single-seed, paired policy comparisons, not independent training replications.

A fresh 512-start gimbal-response screen tested the earlier expected-actuator-endpoint
method against the selected longer-rollout actor. Baseline landed 488, the diagnostic
wrapper 489 (four gains, three losses); the harder band changed from 148/167 to 149/167.
Actuator execution error was below 6.4e-8 rad and 256/512-node quadrature discrepancy
below 2.6e-5 rad. The endpoint match does not also match the actuator filter state or
integrated angle, and the wrapper uses privileged filtered state. It offers no useful
new teacher on this cohort and was not adopted. Evidence is in the long-rollout run's
`actuator-response-screen`, archived under `frontier-actuator-response`.

The next bounded run, `landing_long_trace_local`, keeps 256 environments ×256 steps,
throttle replay, the original comparison source/seed, 50 updates and all retention
guards fixed, changing only GAE lambda from 0.98 to 0.995. This tests trace attenuation
within the longer rollout; it does not claim that lambda extends GAE across rollout
boundaries. Selected and final checkpoints use paired evaluation cohorts, with
initial-direction subsets. Fresh validation is still required before promotion.
The existing `long-rollout-comparison` archive contains the completed comparison;
PR 67 now follows the trace-extension run.

#### Sustained training and interpreting feature rank

Run `landing_tilt_sustained_local/2026-11-16_04-57-16` continues the selected trace
checkpoint for 500 updates. Selection uses fixed 1,536-episode frontier and 512-episode
retention panels; separate fresh cohorts evaluate saved selections every 100 updates.
Reset preflight reproduced all physical state fields, observations and mirror flags
on CPU and GPU. Wind has a separate RNG stream and is reset alongside spawn/mirroring.
The first 100-update snapshot retained warm landings at 97.46%/98.93% and raised-cold
landings at 99.90%/99.32%; the harder band scored 601/635 of 666 (90.24%/95.35%).
Deterministic performance remains below the advancement criterion.

The diagnostic's low raw critic-rank warning prompted a full-history probe of that
snapshot: 26,336 valid states from the first 128 held-out stochastic frontier episodes,
without selecting episodes by outcome. Critic raw rank was 14/128, centered rank
48/128, and centered variation accounted for 2.25% of feature energy. Actor ranks
were 86/95 respectively. Replayed critic values matched the audit exactly. These
ranks use 99% of singular-value mass, not squared energy. The large common critic
feature offset makes uncentered rank alone insufficient evidence of collapse.

Diagnostics now log centered rank and the fraction of energy in feature variation.
Warnings use centered rank when available; legacy raw-only logs explicitly request
the missing check rather than asserting collapse. Zero features have rank zero, and
nonzero rank remains invariant to feature scale. Low centered rank still warrants
investigation, but neither a high rank nor a large rank increase proves good value
calibration. The probe is archived under `critic-feature-rank` on the checkpoint volume.

### Sustained tilt results and the next expansion (2026-11-16)

The 500-update sustained run finished. Its selected iteration 1349 passed one fresh
cohort in the 0.2–0.3 rad band (656/690 deterministic, 657/690 stochastic), but the
**same checkpoint** scored 631/672 and 630/672 on the next cohort. This is not robust
mastery. The original broad rebuild also finished: its final stage-2 policy landed
0/512 episodes in either mode. Neither run is automatically extended.

A matched 100-update pilot compared throttle-only retention with throttle retention
plus the other seven continuous action means. Both began from coherent selected
iteration 1074, with the same training seed and fixed evaluation panels; the replay
teacher remained the older warm/cold policy. Both used four shared-Adam replay steps
per PPO update. Adding losses preserves the throttle loss coefficient, not its exact
optimizer update. The broader anchor's selected actor remained bitwise identical to
its starting actor; throttle-only selected an updated actor.

On fresh paired starts, selected policies scored 657/696 versus 647/696 deterministic
hard-band landings (throttle-only versus broader retention); stochastic scores were
653/696 versus 654/696. At the final checkpoint, deterministic hard-band scores tied
at 316/344, while stochastic scores were 318/344 versus 320/344. Broader retention did
improve final deterministic raised-cold landings, 509/512 versus 501/512; warm scores
were 492/512 versus 490/512. Thus this single-seed pilot suggests a retention benefit
on cold starts but does not establish better frontier learning. Selected and final
paired reports, checkpoint hashes, scripts and configs are backed up under
`control-retention-pair`. A variable shadowing bug interrupted audit backup after the
first completed audit; it was fixed, that backup was hash-verified, and all remaining
audits completed without changing their prescribed seeds.

The next bounded run, `landing_tilt_expansion_local/2026-11-16_05-31-25`, uses
throttle-only replay and expands pitch noise from ±0.3 to ±0.4 rad. It starts from
coherent sustained selected iteration 1284 (`model_selected_1324.pt`), chosen for its
warm retention margin; only accounting counters carry forward from completed 1524.
Preflight on the fixed panels scored 78.68% deterministic in the new 0.3–0.4 band,
with warm 98.05%/98.44%, raised-cold 99.41%/98.44%, and old-tilt aggregate
96.78%/97.85%. Initial retention and exact actor/counter restoration checks passed.

The run is limited to 200 updates, with independent audits and checkpoint backups at
100 and 200 updates. Warm, raised-cold and old ±0.3 aggregate retention must each
remain at least 95% in both modes. Audits separately report the unfinished old
0.2–0.3 band and the new 0.3–0.4 band. Requiring 95% in every narrow intermediate
band was an experimental gate, not a final task requirement; expanding coverage does
not declare the old band solved. Physics, contact success and termination rules are
unchanged. Randomized ground landing and end-to-end catch validation remain required.
