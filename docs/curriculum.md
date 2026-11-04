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
rate. It expands to 209 stages and has a 2,000-update default budget; promotion requires at least 45%
of 256 deterministic evaluation episodes twice in succession. Compact numeric interpolation blocks
keep the long braking tail reviewable while limiting each post-60 transition to 1.25 m/s vertical
speed, 0.75 m/s horizontal speed, 12.5 m pad radius, 5° tilt, and 10°/s angular rate. The final stage
additionally requires exactly one inverted-to-upright recovery with no re-flip and penalizes axial
roll and rotation beyond 360°.

Current campaign: stages 0--166 are solved. The unified policy completed the 51-step two-engine
braking/pad ladder, restored the exact 35 m/s vertical, 22 m/s horizontal and 450 m pad waypoint,
passed the 34.583 m/s vertical half-step after eight more iterations, passed its pad/horizontal
half-step after another 13, cleared the original 34.167 m/s boundary after five, and reached stage
154. It then completed all five steps to 30 m/s and all four to 25 m/s; the exact 25 m/s endpoint
needed 51 stage-specific iterations and promoted to stage 163, the first of nine steps toward
20 m/s. The first three steps toward 20 m/s then passed after 74, 9 and 23 iterations. Stage 166,
step four of nine, initially regressed; its protected best lands 41.6% over 512 episodes, with 66% of
crashes over the horizontal-speed limit, 63% missing the pad and only 7% over the vertical-speed
limit. A second protected rollback briefly restored 48% deterministic landing but exposed that
checkpoint loading reset the optimizer from the intended 1e-6 post-warmup rate to its 3e-5
bootstrap rate. That resume bug is now regression-tested and fixed. Recovery from the protected
stage-166 checkpoint runs at a TensorBoard-confirmed 1e-6, recovered live rollout landing from zero
to 31% in four iterations, and passed the boundary after 16 iterations on consecutive 47% and 50%
evaluations. Stage 167, step five of nine toward 20 m/s, inherited a 46.9% protected peak. Its first
trajectory declined to 15--22% live landing, and a post-fix rollback correctly retained the 1e-6
rate; the 100-update retry nevertheless reproduced 27--35% evaluation landing without a new best.
That redundant control is retired. A protected restart now keeps the actor and healthy 0.94
explained-variance critic, clears only stale actor Adam moments and halves the touchdown-adapter rate
to 5e-7. It also declined and rolled back after 135 updates, so optimizer state and rate are ruled
out as the boundary's cause. A nonuniform interpolation now inserts only the measured midpoint at
325 m pad radius, 22.5 m/s vertical speed and 13.5 m/s horizontal speed. Saved stages 0--166 are
bit-identical, and the former stage 167 remains exact at stage 168. The protected midpoint run opens
at 39--40% deterministic landing and broad-probes at 40.2%. A matched 1,024-episode stochastic
sweep on that policy raises landing from 27.0% at the existing 0.10 throttle-noise cap to 29.2% at
0.05 and 30.8% at 0.03, while return rises from 105.7 to 114.4. The first midpoint continuation used
that measured 0.03 cap so the rollout distribution supplied more successful terminal examples.
A second matched sweep across every continuous head further raises stochastic landing from 30.3%
at 0.03 to 31.7% at 0.02, 32.0% at 0.01 and 33.4% at the configured 0.005 minimum; return is also
highest at 0.005. The fixed-0.005 continuation remains stable through 270 updates at 30--36% live
landing, sets an isolated 47% deterministic best with actor rank 72/128 and critic rank 46/128, and
has no rollback, although it does not produce consecutive gate crossings in a complete 500-update
comparison and ends at 32% evaluation landing. Its latest protected
best lands 38.9% over a fresh 1,024-episode probe. Among crashes, 56% exceed horizontal speed, 51%
miss the pad and 27% exceed vertical speed: compared with the initial midpoint policy, pad-miss
failures improve from 59% to 51% while horizontal-limit failures worsen from 49% to 56%. A protected
parallel branch therefore doubles only horizontal-speed shaping and trains a fresh critic for 100
updates before using the same 5e-7 touchdown-adapter rate and 0.005 exploration. It passes the
inserted midpoint at iteration 16,097 after about 24 actor updates on consecutive 45% evaluations
and reaches the original 5/9 boundary at stage 168. Its protected best lands 40.7% over 1,024 fixed
stage-167 seeds and 38.7% on stage 168, versus 40.2% and 37.1% for the earlier baselines. Stage-168
crashes are 65% horizontal-limit, 50% pad-miss and 17% vertical-limit failures. The live actor then
declines to 24--36% as the fresh critic compresses to rank 4--5/128 despite about 0.71 explained
variance, so the 1,024-environment process is retired after about 150 actor updates with its better
checkpoint protected. The 4,096-environment slot resumed from that stage-168 best with a gentler
1e-5 fresh-critic warmup and a 40% intermediate promotion gate. It passed stage 168 on startup and
began stage 169 before any actor update. Its fresh critic initially retained rank 44--66/128, but
explained variance was only about 0.05 after 50 updates. Extending the warmup to roughly 100 total
updates raised explained variance to 0.19 while compressing rank to 10/128, so that critic was
retired before it could train the actor. A verified transplant now combines all 35 actor tensors
from the promoted stage-169 checkpoint with all 14 critic tensors and the critic optimizer from the
healthy fixed-noise control; only stale actor optimizer moments are cleared. Its 100-update 1e-6
critic warmup retains rank 42--44/128 and reaches about 0.95 explained variance. After switching to
the 5e-7 touchdown-adapter rate, actor and critic rank are 73/128 and 43/128 respectively and the
adapter has nonzero updates. Its first 100 actor updates average 33.5% deterministic evaluation
landing with no rollback; a 1,024-episode probe is statistically flat at 32.8% versus 33.4% for the
source actor. Horizontal-limit crash failures improve from 62% to 55%, while vertical-limit and pad
miss failures move from 24% and 49% to 30% and 52%. By 500 actor updates, full-window evaluation
landing averages 32.5% and latest-100 landing declines to 30.4%. The fixed-seed broad probe falls to
29.1%; pad-miss failures improve to 40%, but vertical failures worsen to 33% and horizontal failures
return to 62%. The healthy run then completes the promised 1,500 actor updates without promotion.
It has one protected rollback near update 886, averages 30.7% deterministic landing across the full
window, and declines to 30.5%, 28.0% and 27.4% over its latest 500, 100 and 20 evaluations. Actor and
critic ranks remain healthy at 72/128 and 43/128 with 0.959 recent explained variance, so this is not
critic collapse or lost adapter plasticity. Its final fixed 1,024-seed policy lands only 27.3% versus
33.4% for the protected source; 46% of crashes exceed vertical speed, 59% exceed horizontal speed and
39% miss the pad. The trained touchdown-only actor is therefore rejected. The curriculum now inserts
only fraction 11/18 between the preserved 5/9 and 2/3
boundaries, making stages 0--168 bit-identical and moving the original blocked boundary unchanged to
stage 170. The unchanged source actor passes that midpoint during critic-only adaptation on
consecutive 41% evaluations. Its 512-environment critic then falls to rank 26/128 after 44 warmup
updates, so the branch is retired before any actor update. A second verified restart combines the
source actor with the long control's healthy, original-boundary-adapted critic and optimizer. It
keeps feature ranks at 59/128 actor and 39/128 critic through 100 direct actor updates, but broad
landing rises only from 33.4% to 34.2% at update 50 and falls to 33.0% at update 100; a critic output
update-ratio warning retires it. The midpoint pass is retained, neither small-branch actor replaces
the protected source, and full GPU throughput returns to the 1,500-update control. A static
architecture sweep then tests whether the touchdown residual merely begins too late. Applying its
learned output below 1,200, 1,500 or 2,000 m yields 0/1,024 landings at every setting, versus 33.4%
at the trained 1,000 m gate. The respective crash means are 24/40 m/s and 558 m miss, 52/47 m/s and
1,054 m miss, and 95/25 m/s and 435 m miss, demonstrating severe out-of-distribution behavior rather
than a useful earlier correction. The selected intervention keeps that gate unchanged, restores all
35 actor tensors from the protected source, and combines them with all 14 critic tensors plus the
critic optimizer from the completed long control. At the unchanged stage-170 boundary it trains only
the already phase-gated post-loop residual, which can correct the descent earlier while remaining
exactly off throughout the learned salto. A checkpoint audit at update 100 confirms that only its six
parameter tensors change; every other actor tensor remains bit-identical. The fixed broad probes land
33.6% at update 100 and 29.8% at update 250, then fall to 22.3% by update 500. Its deterministic mean
declines from 29.9% over the full 500-update window to 27.5%, 25.1% and 25.2% over the latest 250, 100
and 20 evaluations. By update 500, vertical-limit crash failures worsen to 49%, horizontal failures
are 54% and pad misses are 41%; a critic output-layer update-ratio warning joins the negative broad
trend even though actor/critic ranks remain 71/128 and 43/128. The full residual is therefore retired
at the planned checkpoint rather than allowed to trade away the protected engine/throttle braking
schedule. A new `post_loop_lateral_head` update scope freezes the adapter's feature layers and masks
the final layer's three engine-logit and throttle rows. Only the two gimbals, four flaps and roll RCS
can change after the salto; a regression test verifies the protected rows and all shared features are
bit-identical after an optimizer step. The verified 4,096-environment successor opens at 37% landing
from the source actor and healthy boundary critic. Its first 100 evaluations average 33.7%, latest-20
landing is 34.1%, and its fixed 1,024-seed probe lands 34.1% versus 33.4% for the source. Horizontal-
limit failures improve from 62% to 57% while vertical failures stay near baseline at 27%. A tensor
audit confirms that only the final post-loop weight and bias change and that their engine/throttle
rows remain exact. The run then produces the required consecutive 40% evaluations and passes the
formerly blocked stage 170 after about 126 actor updates. The next-stage distribution briefly reads
31/128 critic feature rank, but explained variance remains 0.93--0.97 and the following diagnostic
recovers to 40/128. Training resumes from the protected stage-171 best; its fresh diagnostic is
warning-free at 67/128 actor and 43/128 critic rank. The continuation then completes 500 updates
without promotion or rollback. Deterministic landing averages 30.4% over the full window and 30.7%,
29.9% and 30.0% over its latest 250, 100 and 20 evaluations. Fixed 1,024-seed landing is 30.1% at
update 116, 31.1% at update 266 and 29.8% at update 516: flat within sampling noise rather than the
unrestricted controller's collapse. The latest broad crash failures are 37% vertical speed, 60%
horizontal speed and 49% pad miss. Actor/critic ranks remain 72/128 and 43/128 with 0.963 recent
explained variance. A critic output gradient warning compares the stable current norm with an
exceptionally large first-update spike; representation and value fit remain healthy, so the run
continues toward the requested 1,500-update horizon. At update 1,000, full-window landing is 30.3%
and latest-500, 250, 100 and 20 landing are 30.2%, 30.6%, 30.6% and 31.1%. The fixed checkpoint near
update 1,016 remains at 29.8% broadly, matching update 516; vertical-limit failures improve from 37%
to 29%, horizontal-limit failures worsen from 60% to 64%, and pad misses remain 49%. Actor/critic
ranks are still 72/128 and 43/128 with no rollback or actionable diagnostic warning. This is a flat
healthy long control rather than a collapse, so it continues through 1,500. The 40% gate is only for
intermediate traversal. The full 1,500-update result never promotes or rolls back: deterministic
landing averages 30.1% overall and declines gently to 29.9%, 29.6%, 29.1%, 29.6% and 28.9% over the
latest 1,000, 500, 250, 100 and 20 evaluations. Its fixed checkpoint near update 1,516 lands 27.7%;
32% of crashes exceed vertical speed, 63% exceed horizontal speed and 51% miss the pad. Actor and
critic ranks remain healthy at 73/128 and 44/128, so the constrained final output rows have exhausted
their capacity rather than collapsed. The stalled 2/3 to 7/9 transition is twice the size of the two
preceding measured half-steps. Fraction 13/18 now inserts only its midpoint at 313.89 m pad radius,
21.39 m/s vertical speed and 12.83 m/s horizontal speed; stages 0--170 remain config-identical and the
original stage-171 boundary moves unchanged to stage 172. A verified restart carries the exact solved
stage-170 actor, critic and optimizer into that midpoint. It opens at 35% deterministic landing and
its first diagnostic is warning-free at 67/128 actor and 42/128 critic rank. Fixed checkpoints near
updates 124, 224, 424 and 624 land 32.7%, 33.2%, 34.8% and 34.9% over 1,024 seeds. At update 624,
30% of crashes exceed vertical speed, 57% exceed horizontal speed and 54% miss the pad, while
actor/critic rank remains healthy at 73/42. Landed episodes complete exactly one inverted-to-upright
recovery at this still-loose midpoint, although the later strict cleanup stages must still remove the
subsequent re-flip. The long run passes the midpoint after about 711 actor updates on consecutive 40%
evaluations, saving `model_stage171_passed.pt`, and advances into the unchanged original 7/9 boundary
at stage 172. That boundary opens at 40% and averages 32.4% through its first 59 evaluations; training
continues without rollback. Its fixed checkpoints near stage-specific updates 112, 262 and 513 land
31.4%, 31.8% and 31.8% over 1,024 seeds. At update 513, 33% of crashes exceed vertical speed, 57%
exceed horizontal speed and 51% miss the pad, while actor/critic rank remains healthy at 72/43. The
same fixed probe near update 1,013 lands 29.1%; crash failures shift to 48% vertical speed, 50%
horizontal speed and 48% pad miss, while actor/critic rank remains healthy at 72/44. Landed episodes
still average one re-flip and 944 degrees of roll travel, as expected before the strict cleanup
stages. The full 1,500-update run never promotes or rolls back: deterministic landing averages
30.9% overall and 30.6%, 30.0%, 30.1%, 30.3% and 29.9% over its latest 1,000, 500, 250, 100 and
20 evaluations. Its fixed checkpoint near update 1,513 lands 29.8%; 41% of crashes exceed vertical
speed, 56% exceed horizontal speed and 48% miss the pad, with actor/critic rank still healthy at
72/44. This is another measured healthy plateau. Fraction 3/4 now inserts only the midpoint at
312.5 m pad radius, 21.25 m/s vertical speed and 12.75 m/s horizontal speed; stages 0--171 remain
config-identical and the unchanged original 7/9 boundary moves to stage 173. A verified restart
carries the exact update-1,513 actor, critic and optimizer into that midpoint. Its first three
deterministic evaluations are 34%, 31% and 30%. Fixed 1,024-seed checkpoints near midpoint updates
100 and 500 land 31.0% and 30.1%. At update 500, 37% of crashes exceed vertical speed, 60% exceed
horizontal speed and 47% miss the pad, while actor/critic rank remains healthy at 73/44. Through
514 live evaluations the midpoint averages 31.9% overall and over its latest 500, 31.8% latest-250,
31.2% latest-100 and 31.7% latest-20 without promotion or rollback. This is a healthy but broadly
flat opening. The fixed checkpoint near update 1,000 likewise lands 30.3%; 35% of crashes exceed
vertical speed, 62% exceed horizontal speed and 48% miss the pad, with actor/critic rank still
healthy at 73/44. Through 1,013 live evaluations the midpoint averages 31.9% overall and over its
latest 1,000, 32.0% latest-500, 32.5% latest-250, 32.3% latest-100 and 32.4% latest-20. The live
window is slightly stronger but the broad fixed result remains flat, so the same uninterrupted run
continues to 1,500 updates. The completed run still does not promote or roll back: deterministic
landing averages 32.0% overall and 32.0%, 32.0%, 31.2%, 31.1% and 31.0% over its latest 1,000, 500,
250, 100 and 20 evaluations. Its fixed update-1,500 probe lands 30.4%; 29% of crashes exceed vertical
speed, 67% exceed horizontal speed and 51% miss the pad, with actor/critic rank still healthy at
72/44. The unchanged fixed landing result at updates 100, 500, 1,000 and 1,500 rules out insufficient
training time and makes lateral braking the measured bottleneck. The geometric midpoint is retired.
A checkpoint-safe private post-loop steering adapter was then tested as a matched capacity A/B. It
keeps engine logits and throttle exact while learning its own gimbal, flap and RCS features, but the
live result falls to 26% over its latest 20 evaluations by update 129. Its fixed update-124 probe
lands only 25.8% versus 33.2% for the constrained-head control, and horizontal-speed crash failures
rise from 58% to 72% despite healthy 64/128 actor and 41/128 critic rank. The branch is retired: extra
lateral feature capacity destabilizes the inherited steering schedule rather than solving the measured
midpoint.
A matched reward intervention now doubles horizontal-speed shaping from 2 to 4 while preserving the
exact midpoint update-1,500 actor, critic and optimizer. Its first 100 updates train only the inherited
critic at 1e-4 for the changed returns; the actor stays bit-exact, then only the protected post-loop
gimbal, flap and roll-RCS output rows unfreeze at 1e-6. This avoids the critic collapse seen after
earlier fresh-critic experiments while directly targeting the dominant failure. The warm-up completes
with all 35 actor tensors bit-exact and all 14 critic tensors changed, 0.971 explained variance and
healthy 72/38 actor/critic rank. The first post-warm-up diagnostic confirms the automatic 1e-6 rate
and a small 6.6e-5 update ratio confined to the protected steering output. The matched actor run then
completes 500 updates without promotion or rollback. Deterministic landing averages 30.3% overall
and 30.0%, 30.3% and 29.8% over its latest 250, 100 and 20 evaluations. Fixed 1,024-seed probes land
30.7%, 30.0% and 29.1% at actor updates 100, 250 and 500 versus 30.4% for the source. Horizontal-speed
crash failures improve monotonically from the source's 67% to 59%, 57% and 49%, but vertical-speed
failures worsen from 29% to 37%, 40% and 51%; pad-miss failures finish at 49%. Actor/critic rank
remains healthy at 73/40. The branch is retired: globally doubling the term learns lateral braking
but trades away vertical braking rather than increasing landings.
The next matched reward A/B retains global horizontal-speed weight 2 and adds only weight 2 faded
exponentially below 500 m. It therefore reaches the same total weight 4 at touchdown without
discouraging cross-range correction around the 1,230 m flip. The exact source actor, critic and
optimizer restart with the same 100-update actor-preserving critic adaptation, followed by protected
steering-only training at 1e-6. The unused private lateral path is also safer for a later capacity A/B:
when introduced into an old checkpoint, its hidden layers now inherit the trained post-loop features
while its output remains exactly zero. The localized critic warm-up preserves all 35 actor tensors,
adapts all 14 critic tensors, reaches 0.963 explained variance and retains healthy 72/44 rank. Fixed
1,024-seed landing is 30.6%, 30.5% and 31.8% at actor updates 100, 250 and 500. The update-500 probe
beats both the 30.4% source and 29.1% global-weight branch; horizontal-speed failures improve from
67% to 60% and pad misses from 51% to 50%, while vertical-speed failures rise moderately from 29%
to 37% rather than the global branch's 51%. Deterministic landing averages 31.5% over all 500 live
evaluations and 31.8%, 31.7% and 32.4% over its latest 250, 100 and 20. Actor/critic rank remains
healthy at 73/44. The same fixed probe at actor update 1,000 lands 29.9%: horizontal-speed failures
continue improving to 54% and pad misses to 48%, but vertical-speed failures worsen to 46%. Live
landing still averages 31.9% across all 1,000 updates and 32.4%, 32.5%, 32.0% and 31.6% over the
latest 500, 250, 100 and 20. Actor/critic rank remains healthy at 72/43. The update-500 gain was
not sustained at update 1,000, but the exact update-1,500 endpoint recovers to 31.4% fixed-seed
landing. Horizontal-speed failures remain improved at 55% and pad misses at 49%, while vertical-
speed failures remain elevated at 46%; actor/critic rank is healthy at 73/43. Across the complete
1,500-update run, live landing averages 31.8% overall and 31.7%, 31.7%, 31.7% and 32.7% over the
latest 500, 250, 100 and 20. Localized shaping is therefore modestly positive but still delays
rather than removes the same opposing horizontal/vertical objective pressure seen in the global-
weight control. A direct checkpoint audit confirms that the protected engine-logit and throttle
rows stay bit-exact throughout this run. A separate resumed-optimizer regression exposed that
masked gradients alone do not protect rows carrying old Adam moments; masked rows and their moments
are now restored exactly after every update. The next matched continuation resumes the 31.8%
update-500 best, keeps all engine-count logits and shared features exact, and unlocks only throttle
with steering so vertical thrust can compensate for lateral correction. Its update-50 tensor audit
shows only the final post-loop weight and bias changing: every engine-logit row remains exact while
both throttle and steering move. The fixed update-100 probe lands 31.6%, essentially preserving the
31.8% handoff best and remaining above the 30.4% source. Horizontal-speed failures improve from
60.3% at handoff to 58.9%; vertical-speed failures remain nearby at 38.6% versus 37.4%, rather than
drifting toward the steering-only endpoint's 46.2%. Actor/critic rank is healthy at 72/44 with zero
rollback. The fixed update-250 probe remains at 31.6%; horizontal-speed failures improve further to
53.7%, but vertical-speed failures rise to 44.4% and pad misses are 50.7%. Live landing averages
32.7% across the first 250 updates and 33.4%, 32.8% and 33.6% over the latest 50, 100 and 20. The
extra throttle authority initially slows but does not remove the opposing failure tradeoff. By
update 500, however, the fixed probe reaches 32.8%: the best checkpoint in this stage-172 study,
2.4 points above the 30.4% source and 1.0 point above the 31.8% handoff. Horizontal-speed failures
are 56.8%, vertical-speed failures 41.3% and pad misses 50.7%; actor/critic rank remains healthy at
73/43 with zero rollback. Live landing averages 33.3% across all 500 updates and 34.0%, 33.8% and
34.9% over the latest 250, 100 and 20. The broad gain and strengthening live windows support the
requested long horizon, so the same branch continues uninterrupted through updates 1,000 and 1,500
rather than activating the prepared localized-vertical fallback. Final broad strict-target
acceptance remains unchanged. The longer horizon then produces the first real promotion in this
stage-172 study: consecutive 41% deterministic evaluations at continuation update 651 pass the
inserted 3/4 boundary and advance to the unchanged 7/9 boundary at stage 173. The exact passed
checkpoint lands 33.1% over 1,024 fixed stage-172 scenarios, with 38% vertical-speed, 60%
horizontal-speed and 51% pad-miss failures; actor/critic rank is healthy at 73/44 with zero
rollback. It also lands 31.9% broadly on stage 173 before any stage-specific update. The first 62
stage-173 evaluations average 33.3% and 33.8% over the latest 40, so training continues unchanged
instead of activating the fallback. The passed checkpoint still averages one later re-flip among
landed episodes; the downstream cleanup stages remain responsible for the exact single-salto
invariant. At 250 stage-specific updates, fixed stage-173 landing improves slightly to 32.3% and
the live window averages 33.1% overall, 32.9% latest-100, 32.7% latest-50 and 32.4% latest-20.
Horizontal-speed failures improve from 59.1% at entry to 51.1%, while vertical-speed failures rise
from 39.7% to 48.3% and pad misses remain near 52%. Healthy 73/43 actor/critic rank and the modest
broad landing gain support continuing this recently promoted branch through the same 500--650
update horizon before activating the prepared vertical fallback.
The localized update-500 seed-3
[41-second chase recording](assets/salto-stage172-touchdown-u500.mp4) and
[nine-frame sequence](assets/salto-stage172-touchdown-u500-sequence.jpg) show the powered salto,
recovery, descent and a representative stage-172 landing. It contacts at 20.02 m/s vertical speed,
1.67 m/s horizontal speed and about 243 m miss under the interim 21.25 m/s, 12.75 m/s and 312.5 m
limits. This is evidence for the localized shaping result, not final strict acceptance: the broad
policy must still demonstrate exactly one recovery, no re-flip and a safe landing before merge.
The original stage-172 boundary's update-1,013
[40-second chase recording](assets/salto-stage172-u1013.mp4) and
[nine-frame sequence](assets/salto-stage172-u1013-sequence.jpg) show seed 3 completing the powered
salto and landing under the unchanged 7/9 boundary. It contacts at 19.58 m/s vertical speed,
4.65 m/s horizontal speed and about 233 m miss under the interim 21.11 m/s, 12.67 m/s and 311.11 m
limits. The later re-flip remains visible, so this is current curriculum evidence rather than a
clean-single-salto or safe-target claim. The measured stage-171 control's update-324
[40-second chase recording](assets/salto-stage171-midpoint-u324.mp4) and
[nine-frame sequence](assets/salto-stage171-midpoint-u324-sequence.jpg) show seed 3 completing the
powered salto and landing under the tighter midpoint. It contacts at 18.51 m/s vertical speed,
4.32 m/s horizontal speed and 234 m miss under the interim 21.39 m/s, 12.83 m/s and 313.89 m
limits. This remains an intermediate curriculum landing with a later re-flip, not the final clean
safe-target claim. The earlier protected midpoint best's
[40-second chase recording](assets/salto-stage167-midpoint-live.mp4) and
[nine-frame sequence](assets/salto-stage167-midpoint-live-sequence.jpg) show seed 3 completing one
powered salto, recovery and the two-engine descent. It contacts at 19.13 m/s vertical speed,
4.89 m/s horizontal speed and 237 m miss under the interim 22.5 m/s, 13.5 m/s and 325 m limits;
this is closer than the stage-131 recording but is still not the safe-target claim. This is a
balanced terminal-control problem, not another single-limit failure. The historical path to that
boundary starts with
`salto_stage60_lateral_pad_v1`: its
0.625 m/s braking ladder reached and solved the 80 m/s boundary (stage 34) after about 1,000 stage
updates. Tightening all five landing constraints
directly from 80 to 70 m/s then dropped deterministic landings to zero. Interpolating that transition
over 16 stages restored a learnable boundary: the first 79.375 m/s stage passed in about 90 updates,
78.75 m/s passed after about 270 updates, and 78.125 m/s passed after a critic refresh restored its
collapsed feature rank from 31/128 to 76/128. The next 77.5 m/s stage then stalled at a 44.1% best
through 271 updates; increasing rollback patience produced longer trajectories but no better policy.
A measured 77.8125 m/s midpoint reached 46.1%, passed in about 140 updates, and transferred into the
original 77.5 m/s stage, which passed after about 174 updates. The next 0.625 m/s transition again
collapsed to 21-32%, so the complete 77.5 to 70 m/s segment now uses 0.3125 m/s interpolation across
all five constraints. The first 77.1875 m/s attempt plateaued at 44.1% for 273 updates and 13
protected retries. A 77.34375 m/s bridge crossed the gate once but did not produce consecutive
passes in 140 short-patience plus 140 long-patience updates. Its 77.421875 m/s bisection repeatedly
reached 43-44% and crossed the gate in isolation, but matched short- and long-patience runs again
failed the consecutive gate. A final 77.4609375 m/s half-step was inserted. Aligning stochastic
engine exploration with the deterministic expected-count decoder and updating only the continuous
action rows then broke through that boundary without disturbing the learned engine sequence. The run
passed stages 40 through 52 with no rollback; the 75 m/s and 74.6875 m/s stages needed 112 and 132
updates despite several earlier isolated gate crossings. Stage 53 then plateaued at a 41.0% best over
153 updates; a probe showed horizontal-speed and pad misses dominated while mean crash vertical speed
was already below its limit. A coupled 74.21875 m/s midpoint was inserted before retrying the original
74.0625 m/s boundary, which then passed in 46 updates. The next 73.4375 m/s stage plateaued at a
39.5% 512-episode probe over 155 updates with the same horizontal-speed and pad-miss pattern, so a
73.59375 m/s midpoint was inserted. This is not yet a safe-pad result.
The midpoint then remained at 39--41% for 155 updates at 1e-6. Doubling the post-warmup actor rate
did not improve it over 100 updates and caused 39% of recent updates to hit the KL limit within two
optimizer steps. Because the same probe attributed 70% of crashes to horizontal speed while vertical
impact speed was already inside the limit, stage 56 now raises horizontal-speed shaping from 0.25 to
0.5, matching the strength that unlocked the general landing curriculum, while retaining the strict
45% promotion gate. After more than 300 continuous-head updates, however, the recent-100 mean was
still 38.8%. The boundary is therefore split into a horizontal-only half-step and then a pad-radius
half-step before the coupled 73.59375 m/s limits; vertical speed, attitude, and angular rate remain
at their solved stage-55 values during both bridges. The horizontal-only bridge likewise stayed at
40.0% through 225 actor updates; a 1,024-episode probe showed 40.5% landed and unchanged horizontal
failure statistics. The final linear continuous output rows therefore lacked useful capacity for
this correction. `ContinuousAdapterRNNModel` adds a zero-initialized nonlinear residual fed by the
frozen recurrent latent. Existing checkpoints load with bit-identical initial actions, and the
adapter has no computation path to engine logits.
The 64-unit adapter at 1e-5 passed the horizontal-only stage in 28 actor updates and the pad-only
stage in another 63, then reached a 48.8% best on the formerly blocked coupled boundary. Continued
updates at that rate degraded the recent window to 29--40%, so training resumed from the preserved
stage-58 best with a refreshed critic and a 2e-6 adapter rate. That checkpoint passed stages 58 and
59. The following 73.125 m/s stage reached a 49.2% best but failed to make that behavior reliable
before two protected rollbacks, so a 73.28125 m/s coupled midpoint now precedes it. Its best policy
landed 40.9% over 1,024 episodes and reduced horizontal-limit failures from about 71% to 61%, but
42% of crashes still missed the pad and another 289 actor updates stayed flat. Stage 60 therefore
raises distance shaping from 0.02 to the general landing curriculum's 0.05 while preserving that
actor and refreshing its critic. The historical landing ablation actually attributed its 46% to
81% improvement to a separate near-ground lateral-position potential, not to total 3D distance.
Total distance is dominated by altitude during the salto descent, so stage 60 now also applies the
same 0.05-per-metre horizontal potential with a 500 m exponential fade. This remains
policy-invariant shaping and directly distinguishes pad-centred approaches from the measured miss
mode without changing the touchdown gate. The total-distance-only control was then allowed 1,500
actor updates rather than being judged from a short window. It remained flat at about 39% and its
protected best still landed 40.9% over a final 1,024-episode probe: 61% of crashes broke the
horizontal-speed limit, 42% missed the pad, and neither attitude nor angular rate was binding. The
actor adapter remained plastic, but critic feature rank collapsed from 22/128 to 4/128. The lateral
run therefore started from the protected actor with a fresh critic and optimizer. The dedicated
lateral term passed stage 60 after about 715 actor updates on consecutive 47% and 46% evaluations;
retain horizontal-speed shaping at 0.5 because the measured intervention solved the boundary without
the queued increase to 1.0. Stage 61 passed after another 292 updates. The following 0.3125 m/s step
then stayed at a 36.4% recent-300 mean through 505 updates, so a 72.96875 m/s midpoint now precedes
the original 72.8125 m/s boundary. That midpoint also stayed flat: its recent-500 landing mean was
36.8%, with only seven noisy gate-level evaluations and no consecutive passes after 501 actor updates.
The adapter was still updating; critic feature rank declined from about 60/128 to 39/128 but did not
repeat the prior collapse to 4/128. A 73.046875 m/s bridge now halves the transition from the solved
stage-61 policy, with another fresh critic, before retrying both stricter boundaries. The easier
bridge nevertheless stayed at a 37.2% recent-500 mean with nine isolated gate-level evaluations and
no promotion. Further subdivision would only train against the same marginal stage-61 behavior, so
stage 62 now raises horizontal-speed shaping from 0.5 to 1.0, directly targeting the dominant 61%
horizontal-limit crash mode measured by the last 1,024-episode probe. A full 2,500-evaluation run
with that stronger shaping still ended at a 36.1% recent-500 mean after two protected rollbacks. Its
512-episode protected-best probe landed 39.6%; 63% of crashes broke horizontal speed, 41% broke
vertical speed, and 39% missed the pad. The policy used exactly one engine throughout, with throttle
well below saturation. Continuous-only adaptation therefore cannot discover the extra braking
authority needed by the remaining curriculum. The actor now supports a separate zero-initialized
engine-logit adapter: it preserves the checkpoint bit-for-bit at load, but a hybrid-adapter update
scope can learn engine count while the recurrent controller and original action head remain frozen.
Per-step count sampling gave that adapter no successful rollout landings in its first 500-update
control. Holding each sampled count for 20 control steps restored a useful signal: rollout landing
rose to 5--7%, and a 512-episode matched probe found successful trajectories used 1.123 engines on
average versus 1.079 for crashes. Deterministic landing nevertheless remained near 37% through the
first 500 actor updates, while critic feature rank declined to 31/128. An offline persistence sweep
on the same policy measured 4.7% stochastic landing at 20 steps, 11.0% at 60, 16.2% at 100, and
24.8% at 150 steps across four seeds. A once-per-phase variant (one count for the salto, resampled
once at recovery and then held through landing) retains about 20% successful exploratory episodes
without interrupting the two-second ignition transient. The actor now also has a zero-initialized
post-loop observation adapter and explicit recovery-handoff resampling, both exactly gated off during
the learned salto. The 20-step control later climbed to 12--13% exploratory landing, but deterministic
landing declined until the protected rollback fired after about 805 actor updates. The held count had
been charged a fresh categorical log-probability at every timestep even though it was sampled only
once, and the once-per-phase variant would also have sampled an untrainable random count before the
post-loop gate. Ranked-count rollouts now use the deployment count during the salto, sample only at
the recovery handoff, and apply categorical probability, entropy, and KL credit only at that actual
decision. The post-loop-only scope also freezes shared continuous noise. This corrected phase-coherent
controller is the next long-run intervention. The completed 1,500-actor-update control never
promoted: its final recent-500 deterministic mean was 35.6%, its exploratory landing mean was 5.8%
(8.2% over the final 100), and its protected 39.6% best was never replaced. Actor feature rank stayed
healthy at 91/128 while critic rank ended at 26/128; the run had one protected rollback.
The corrected run then trained a fresh critic for 200 updates on a stable 30--31% exploratory
landing distribution (explained variance reached 0.81), while keeping the protected salto actor
exactly frozen. Once enabled, the direct-observation post-loop adapter raised rollout landing to
38--41% and produced consecutive 48% deterministic evaluations, passing stage 62 after only about
65 actor updates. The same policy then passed every remaining fine braking boundary through 70 m/s
(stages 63--73) without a rollback or restart, reaching the 60 m/s stage 74 cliff after only about
175 actor updates in total. This clears the complete segment that had blocked all continuous-only and
20-step count controls. The following direct 70 to 60 m/s transition then produced zero train and
deterministic landings for 311 updates while critic rank fell to 13/128. The protected stage-73 actor
then resumed with a fresh critic and passed the new 67.5 m/s bridge after 36 actor updates and the
65 m/s bridge after another 144, without rollback. The 62.5 m/s bridge then stayed near 20--29%
deterministic landing for its first 340 updates, but the healthy actor continued improving: rollout
landing rose into the 30--38% range, the protected deterministic best reached 43.4%, and the run
finally produced the consecutive gate-level evaluations needed to pass after 497 stage-specific
updates. Actor feature rank remained about 63--66/128, the post-loop adapter retained nonzero
gradients, and the run had no rollback. This validates the long horizon instead of a premature
midpoint restart. The original 60 m/s boundary now starts with 27--34% deterministic landing rather
than the zero signal measured by the failed direct 70 to 60 m/s trial; training continues from the
promoted actor. The original schedule after 60 m/s jumped directly to 50 m/s while shrinking the pad
by 75 m. All solved indices through stage 77 are now preserved verbatim for checkpoint compatibility,
while the remaining coarse waypoints expand into 65 bounded landing stages before the seven exact-salto
cleanup stages. Stage 77 passed after 136 actor updates with no rollback. The obsolete process then
confirmed the predicted cliff at the old 50 m/s stage with zero train and deterministic landings, so it
was stopped after eight updates. A fresh-critic restart from the protected stage-77 actor now resumes at
the new 58.75 m/s, 640.625 m pad boundary. Its first critic-warmup evaluations retain 35--37% landing
and its first complete exploratory rollout lands 28%, demonstrating that stage 78 is learnable before
any actor update. After the full 200-update critic warmup reached about 0.59 explained variance, the
post-loop adapter raised exploratory landing from about 27% to 40% and passed stage 78 in 40 actor
updates with no rollback or KL stop. Stage 79 inherited 31--36% deterministic landing and passed in
another 63 actor updates. Stage 80 (56.25 m/s vertical speed, 621.875 m pad) then inherited a usable
policy, climbed from 31--44% to consecutive gate-level evaluations, and passed after 83 actor updates.
Stage 81 (55 m/s, 612.5 m pad) recovered from an initial 24--38% range, reached 39--42% exploratory
landing, and passed after 142 actor updates. Stage 82 (53.75 m/s, 603.125 m pad) then spent roughly
200 updates in a 26--38% trough before recovering and passing after 322 updates. That validates the
long horizon again. Stages 83--85 subsequently passed in 149, 147, and 178 updates, protecting the
50 m/s policy, and the first 40 m/s bridge passed after another 67 updates. None of these transitions
rolled back or hit a KL stop. The exact protected stage-85 actor landed seed 3 in 32.0 s; its
[chase-camera recording](assets/salto-stage85-50ms.mp4) and
[nine-frame sequence](assets/salto-stage85-50ms-sequence.jpg) show the full single salto, recovery,
and deliberately still-loose 50 m/s curriculum touchdown. Stage 87 (47.5 m/s) inherited 36--39%
deterministic landing and passed after 81 updates. Stages 88--91 then passed in 79, 81, 49, and 81
updates, reaching the 42.5 m/s boundary without a rollback. Stage 92 (41.25 m/s) exposed a
train/deployment mismatch: the protected policy landed 26.2% over 512 deterministic episodes, but
the original std-0.10 run collapsed to 0% deterministic landing while stochastic rollouts remained
near 30%. Reducing the actor rate fivefold only delayed the same collapse. A matched 512-episode
exploration sweep improved expected landing from 28.5% at std 0.10 to 32.4% at std 0.03. The current
protected restart therefore combines the untouched actor with its adapted critic, clears the actor's
Adam moments, caps continuous std at 0.03, and uses a 5e-7 post-loop-adapter rate. It improved from a
27.6% first-20 evaluation mean to 34.7% over the latest 20 and set a 42.2% protected best by update
134, with critic explained variance near 0.93 and no rollback or KL stop. A 512-episode probe
attributes about 89% of remaining crashes to the 41.25 m/s vertical-speed limit; attitude and angular
rate are non-binding. A second matched probe found that deterministic decoding briefly dropped from
one to zero engines after the recovery handoff in 35% of episodes, despite persistent-count training.
Holding the deterministic handoff count through landing raised success from 35.9% to 40.4%, kept the
mean count exactly one, reduced crash vertical speed from 45.64 to 45.26 m/s, and is now implemented
for both evaluation and TorchScript export. The old-decoder control completed 500 updates: exploratory
landing reached 40--43% while its flickering deterministic evaluation ended at 24--27%. Re-evaluating
its saved actors with the corrected decoder instead improved monotonically; the final checkpoint
landed 51.6% on the 256 curriculum scenarios and 46.9% over 1,024 episodes, with mean crash vertical
speed down to 44.22--44.47 m/s. Restarting its matching model, critic, and optimizer under corrected
semantics passed stage 92 after six updates. Stage 93, the final 40 m/s boundary, inherited 44--49%
landing and passed after another 86 updates without rollback or KL stop. Stage 94 starts the 35 m/s
ladder with 31--44% deterministic landing instead of a zero-signal curriculum cliff. Its protected
best first crossed the 45% gate at update 118 (47.7%), but the following evaluation fell below the
gate, so the unchanged control remains in the promised long training window. A 512-episode probe
landed 37.5% and found that 82.8% of crashes exceeded only the vertical-speed limit. At the observed
landing mass, one engine supplies about 3.0 m/s² net deceleration at full throttle while the dense
speed reward assumes the general landing policy's three-engine 10 m/s² envelope. A matched
post-salto throttle-logit sweep raised landing from 37.9% to 45.3--48.1%; a protected parallel run is
therefore testing a 2 m/s² speed profile after a fresh-critic warmup without interrupting the control.
The shared 0.03 continuous exploration cap also made the measured +0.3 throttle improvement a
ten-sigma event. Exploration ranges can now be overridden by actuator without changing old
checkpoints: widening only throttle to 0.10 reduced a matched 512-episode exploratory rollout by
0.6 percentage points, versus 6.1 points when gimbal, flaps and roll control were widened as well.
The corrected 2 m/s² braking reward then trained the critic for 200 updates while leaving the actor
bit-identical. Explained variance reached 0.729, but feature rank remained 13/128, so a further 100
critic-only updates raised explained variance to 0.777 while rank still ended at 12/128. That warning
has therefore been exercised rather than ignored; extending the same fit no longer has supporting
evidence. A checkpoint-compatible +0.8 offset to the trainable post-loop throttle residual improves
the unchanged policy from 38.4% to 51.6% in a matched sweep and scores 55.3%, 47.9%, and 43.1% over
1,024 deterministic episodes at stages 94--96. It clears only the shifted bias's optimizer moments
and preserves every other actor and critic moment. Starting the live continuation from the exact
100-update critic checkpoint passed stage 94 immediately at 54%, stage 95 after two actor updates,
and stage 96 after another 13. Stage 97 inherited 36--39% deterministic landing. The unchanged
control vindicated the requested long horizon by eventually passing stage 94 after 539 evaluations,
without rollback, and moving to stage 95. It continues in parallel as an independent comparison.
Over the first 120 stage-97 updates, the protected best rose to 42.2% and the latest 20-evaluation
mean recovered to 36.3% after an intervening dip to 29.9%. A 1,024-episode probe of the newer best
lands 37.1%; 82.8% of crashes
exceed the vertical-speed limit and 48.3% miss the pad. Repeating the throttle-residual sweep on that
newer checkpoint finds that the actor has already learned the earlier braking offset: a further +0.1
changes the matched 256-scenario rate only from 41.4% to 41.8%, and larger offsets trade vertical
speed for worse pad accuracy. The live optimizer therefore continues instead of restarting from the
older sweep. The longer window is now paying off: by 500 stage-97 evaluations the latest 50 mean rose
to 38.1%, six evaluations since update 250 crossed the 45% gate, and the protected best reached
47.3%. A 1,024-episode probe at update 300 improved from 37.1% to 39.5%; current explained variance
is near 0.74, actor rank is 66/128, and rollback count remains zero. The crossings are not yet
consecutive. By update 750 the best had not improved for roughly 275 updates and the latest 50 mean
had fallen to 32.3%, despite healthy 0.79 explained variance. A protected restart from the 47.3%
best preserved the actor, critic, optimizer, reward and exploration settings. Its four first
25-update means are 36.1%, 37.4%, 37.4%, and 37.5%, with two isolated gate crossings, 0.78 explained
variance, actor rank 67/128, and no rollback. Stage 97 therefore remains active toward a combined
1,500-update horizon. A fresh 1,024-episode probe lands 38.5%; 90% of crashes violate vertical speed
and 42% miss the pad. Since the final 36.25 to 35 m/s transition has remained below the consecutive
gate for more than 900 combined actor updates, stage 97 now bisects only that blocked transition at
35.625 m/s while preserving the three already-solved 35 m/s interpolation stages exactly. The
original 35 m/s boundary moves to stage 98 and remains unchanged. A protected parallel run tests
this curriculum bridge while the long stage-97 optimizer continues untouched. Its early protected
best lands 41.7% over 1,024 episodes; the midpoint passed after 101 updates on consecutive 48%
evaluations, and the original unchanged 35 m/s boundary inherited a 46% first evaluation and passed
after another 26 updates. The original first 30 m/s interpolation step plateaued at a 39.5%
protected best over its first 80 evaluations; a 1,024-episode probe lands 36.0%, with 89% of crashes
breaking vertical speed and 59% missing the pad. A protected 34.583 m/s midpoint now precedes that
original stage while preserving all solved and downstream waypoints exactly. Its first protected best
lands 38.0% over 1,024 episodes versus 36.0% at the original step, but the coupled midpoint averaged
37.25% over 100 evaluations, declined to 35.4% over its latest 20, and crossed the 45% gate only in
isolation despite actor rank remaining healthy. Following the earlier general-landing breakthrough,
the first stage-99 intervention isolated the vertical-speed half-step while retaining the solved
450 m pad and 22 m/s horizontal limit; a lateral half-step then preceded the original boundary.
The coupled run remains an untouched control while matched fresh-critic runs compare the split. A
fresh critic on the vertical-only stage restores feature rank from 15/128 to 67/128 while keeping
the protected actor bit-identical for its first 100 updates. A second zero-initialized touchdown
adapter now provides a direct-observation residual only after the salto and below 500 m. It preserves
the stunt, recovery and high-altitude descent exactly at initialization, then lets the remaining
strict stages learn near-ground braking and pad corrections without moving those solved behaviors.
The 500 m specialist completed 100 live actor updates after its fresh-critic warmup, but both its
40-update and 90-update checkpoints remained exactly at the protected 39.1% over 1,024 fixed-seed
episodes. Mean crash speed moved only from 41.1 to 40.9 m/s; 87% of crashes still violated vertical
speed and 52% missed the pad. It was therefore retired in favor of a matched 1,000 m-gated specialist
that has more altitude to alter the braking trajectory while keeping the completed salto protected.
That wider gate also finishes at the inherited 39--40% training rate: its 40- and 90-update
checkpoints both land 38.7% broadly, while pad-miss failures rise to 63--65%. It is retired after
101 actor updates; the adapter remains under test only in combination with the corrected
vertical-only objective.
A matched 256-episode sweep also rejects a constant post-loop throttle warm start at this boundary:
the unshifted actor lands 39.1%, offsets from +0.1 through +0.8 fall from 37.5% to 21.1%, and negative
offsets of -0.1 and -0.2 likewise regress to 37.5% and 35.2%.
Doubling vertical-speed shaping to 20 also fails as a standalone intervention: after 105 actor
updates its update-102 mean is 38.82%, latest-20 mean is 37.65%, and its 90-update checkpoint lands
38.5% broadly. Although mean crash speed falls to 40.5 m/s and vertical-limit failures fall from 87%
to 84%, pad-miss failures rise from 52% to 58%, so that branch is retired. Inspection then exposed
the objective mismatch behind that trade: the dense `speed_weight` potential penalizes total 3-D
speed even though the hard gate is vertical speed and `horizontal_speed_weight` already shapes the
lateral component. A backward-compatible `vertical_speed_weight` now targets descent speed only;
stage 99 switches from total-speed weight 10 to vertical-only weight 10 while all solved stages
remain unchanged. The legacy-objective 1.0 m/s² timing control completed 100 actor updates, declined
to a 36.7% latest-20 mean, and its 90-update checkpoint lands 37.8% broadly while 64% of crashes miss
the pad; it is therefore retired. The 0.5 m/s² run is likewise superseded after 61 critic-only
updates. Matched vertical-only-objective runs now compare the original 2.0 m/s² profile with 1.0
m/s² while preserving the same protected actor, fresh-critic warmup, and optimizer settings.
The corrected 2.0 m/s² run is also retired after 101 actor updates: its 90-update checkpoint lowers
mean crash speed to 39.9 m/s and vertical-limit failures to 81%, but broad landing falls to 37.8% as
pad-miss failures climb to 65%. The corrected 1.0 m/s² run initially avoids that trade at update 40,
preserving the 39.1% broad baseline and 52% pad-miss rate, but by update 90 it falls to 33.6% landing
as pad misses reach 70% despite vertical failures improving to 74%. That control is retired after 106
actor updates. A matched 2e-6 actor-rate run also regresses to 34.0% broad landing by update 90,
with 82% vertical-limit and 68% pad-miss failures, and is retired after 124 actor updates. Doubling
near-ground pad-centering weight from 0.05 to 0.10 retains 37.1% broad landing at update 90 but still
reaches the same 68% pad-miss rate, so that 4,096-environment branch is retired after 114 actor
updates. The matched 0.20 branch is steadier but does not improve: its update-40 and update-90 broad
probes land 38.5% and 38.8%, with pad-miss failures at 56–57% and vertical-limit failures fixed at
84%. It is retired after completing 106 actor updates.
The combined vertical-only/1,000 m touchdown adapter is steadier in its training window but likewise
fails to improve broadly: its 40- and 90-update checkpoints land 38.9% and 38.7%, with 63% and 61%
pad-miss failures. It is retired after 102 actor updates. Because every branch that begins to reduce
vertical-speed violations simultaneously drifts outside the 450 m pad, two long-horizon runs now
separate those objectives more cleanly with a temporary 500 m stage-99 pad gate. Both preserve the
same protected actor, vertical-only 1.0 m/s² objective, fresh-critic warmup and 5e-7 actor rate; the
4,096-environment run keeps the baseline 0.05 centering weight and the matched 1,024-environment run
uses 0.10. Their resolved 153-stage configurations were checked before actor training. Both passed
the relaxed pad gate during critic-only warmup, but their first stage-100 actor checkpoints land only
38.9% and 37.9% broadly, retain exactly one engine and leave 88% of crashes above the vertical limit.
They are retired at update 40; neither result is counted as new braking behavior.
A trajectory audit then identified a physical limit hidden by the aggregate crash metrics: the
one-engine controller is already at essentially full throttle from 1,000 m to touchdown. Failed
episodes cross 500 m at 62--69 m/s and still contact at 36--48 m/s, so reward tuning cannot create
the missing braking authority. A whole-descent two-engine warm start found a learnable loose gate but
was superseded during critic-only warmup by a more localized controller. The existing touchdown
adapter can now reselect and persist a new engine count exactly when its altitude gate activates,
with matching training, deterministic deployment and TorchScript semantics. A measured 1,000 m
handoff uses a +1.24 engine-logit and -3.0 throttle residual while leaving the salto and high-altitude
one-engine descent bit-exact. On a 700 m pad, 95 m/s vertical and 60 m/s horizontal bridge it lands
34.4% over 256 fixed-seed episodes and uses 1.18 engines on average; the remaining crashes are 57%
vertical-limit, 40% horizontal-limit and only 10% pad-miss failures. A fresh critic now warms up on
that bridge before the altitude-gated adapter begins actor updates; a matched seed-1 run uses 4,096
environments and a 2x adapter rate to test the same handoff with higher-throughput experience. A
probe bug briefly overstated both policies because its confidence-only stochastic forward cleared
the deterministic touchdown-count latch before every action; confidence now runs through a separate
policy copy. Corrected 1,024-seed probes at actor updates 40 and 90 are 32.8%/33.9% for the
conservative branch and 35.2%/39.4% for the high-throughput branch, against a 34.8% baseline. The
conservative branch is retired after 149 actor updates. The loose high-throughput branch is retired
after 200 actor updates: its update-190 probe lands 38.7%, averages 1.003 engines and leaves 99% of
crashes above the 95 m/s vertical limit, confirming that the loose success gate rewards a retreat to
the inherited one-engine schedule. A stricter 90 m/s branch instead lands 12.3% at actor update 29;
its successful episodes retain 1.424 engines versus 1.159 on crashes. Its 4,096-environment
continuation improves clean 1,024-seed landing from 16.2% at actor update 50 to 20.9% at update 150
and 35.4% at update 200. Mean crash horizontal speed falls from 67.4 to 62.8 m/s against the 60 m/s
gate while vertical-limit failures stay at 2% and engine count stays near 1.41. The base run remains
untouched for the long horizon. After one protected rollback before solving the bridge, it passes the
90 m/s gate at iteration 14,523 and then clears all 51 cumulative braking/pad steps in another 865
iterations without a further rollback, reaching stage 151. Actor feature rank is 61/128 and the
touchdown adapter retains strong nonzero updates. Critic rank oscillates from 8--45/128 and briefly
triggers the collapse warning, but its latest diagnostic is 34/128 while explained variance
remains 0.84--0.92; retain it while promotions continue. A matched 5x
horizontal-speed-shaping branch is retired during critic-only warmup because the base policy solves
the measured lateral bottleneck before that A/B reaches actor training. The two-engine policy
must subsequently traverse a dense braking ladder and re-tighten the pad before the strict 15 m
target.
A checkpoint-safe post-loop engine-logit warm start then tested whether the fixed one-engine landing
schedule lacked braking
authority. On matched 256-scenario evaluations, logit shifts through +0.60 retain one engine and
45.3% landing; the first state-dependent two-engine mixtures at +0.62, +0.64, +0.66 and +0.68 fall
to 37.9%, 23.4%, 10.5% and 0.8%, respectively, while +0.70 and above land 0%. A static whole-descent
count shift is therefore rejected; the altitude-gated bridge above instead adds authority only for
the landing burn and jointly adapts throttle, gimbal and count. A new
[36.3-second stage-131 chase recording](assets/salto-stage131-live.mp4) and
[nine-frame sequence](assets/salto-stage131-live-sequence.jpg) show immutable checkpoint
`policy_15099.pt` completing one salto and the two-engine descent at step 32/51. It contacts at
42.7 m/s vertical speed, 26.5 m/s horizontal speed and 428 m pad miss under the intermediate
55.5 m/s, 36.2 m/s and 543 m curriculum limits. That improves the stage-110 recording's
67.8 m/s, 43.8 m/s and 519 m, but is still not the safe-landing claim. The earlier
[34.2-second stage-110 chase recording](assets/salto-stage110-live.mp4) and
[nine-frame sequence](assets/salto-stage110-live-sequence.jpg) show seed 1 completing one salto and
the two-engine descent at step 11/51. It contacts at 67.8 m/s vertical speed, 43.8 m/s horizontal
speed and 519 m pad miss under the still-loose 78.1 m/s, 51.8 m/s and 646 m curriculum limits; the
visibly rough touchdown is transparent intermediate evidence, not a safe-landing claim. The earlier
[27.8-second 90 m/s two-engine recording](assets/salto-stage99-v90-u50.mp4) and
[nine-frame sequence](assets/salto-stage99-v90-u50-sequence.jpg) show the actor-update-50 bridge
policy completing one salto and landing with a persistent two-engine burn. This is live training
evidence, not the final safe-pad claim. A
[39.8-second stage-97 chase recording](assets/salto-stage97-35ms.mp4) and
[nine-frame sequence](assets/salto-stage97-35ms-sequence.jpg) show a landed 35 m/s curriculum
episode. They are an interim stricter-boundary artifact, not the final 15 m safe-pad claim.

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
- Tightening vertical speed, horizontal speed, pad radius, tilt and angular rate together from the
  80 m/s boundary to 70 m/s was another curriculum cliff: the inherited policy went from a solved
  stage to 0% landings. Sixteen interpolated steps across all five constraints made the first boundary
  learnable immediately and preserved the same strict promotion gate.
- At 78.125 m/s the actor remained healthy while the critic feature rank fell to 31/128. Continuing
  from the best actor with only the critic and its optimizer moments reinitialized restored critic rank
  to 76/128, improved the deterministic best from 40.2% to 49.6%, and passed the stage. Preserve the
  actor when value-network collapse, rather than policy plasticity, is the measured bottleneck.
- At the next 77.5 m/s boundary, 271 updates with short rollback patience and another 100 with long
  patience both stayed below promotion despite healthy ranks. Inserting one evidence-driven half-step
  at 77.8125 m/s produced repeatable gate-level evaluations, passed in about 140 updates, and entered
  77.5 m/s near the gate; that original boundary passed after about 174 updates. The following 0.625
  m/s step again collapsed to 21-32%, so the remaining ladder through 70 m/s was densified to 0.3125
  m/s rather than rediscovering the same cliff at every stage. Use local midpoint stages when
  optimizer changes do not move a healthy near-gate policy; do not weaken the promotion threshold.
- Cumulative absolute horizontal rotation was not enough to define a single salto. A deterministic
  probe of an otherwise promising stage-18 controller averaged about 930° of horizontal travel and
  repeated re-flips. A later stage-50 probe still averaged 1.22 re-flips and roughly 1,000° of axial
  roll travel. Final touchdown checks exactly one inverted-to-upright recovery and zero re-flips;
  six preceding cleanup stages now ramp axial-roll, overshoot and re-flip costs before enforcing that
  sparse invariant instead of introducing every constraint at once.
- Independent Bernoulli exploration gave the continuous controller no useful training landings even
  though the expected-count deployment decoder landed about 40%. A categorical engine-count sampler
  over the decoder's nested top-k schedules restored 25-40% stochastic landings. Full-policy PPO still
  destroyed the discrete schedule with actor changes near 1e-5 relative scale, so the successful run
  freezes the recurrent backbone and engine-logit rows and fine-tunes only continuous action rows at
  1e-6 after a 50-update critic warmup. It passed eleven consecutive braking stages through 75 m/s
  with no rollback; preserve this separation while the discrete engine schedule remains discontinuous.

## Carried over from 2D

- Shaping must be γΦ′ − Φ with zero terminal potential; otherwise it biases or pays for survival.
- Speed shaping only above a safe descent profile; braking at any altitude makes policies hover.
- Fixed lr; rsl_rl's adaptive schedule pinned it at 1e-2. Reward scaling and value normalization hurt.
- entropy_coef 0.002 with lr 1e-4 made policies decisive and broke plateaus.
- Near-ground starts first; the full scenario alone was not learnable.
