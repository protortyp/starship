# Upper-stage chopstick catch

`curriculum=catch` trains the Starship upper stage to descend between open arms, let its lower
fins pass, and reach the pins after the arms close. Super Heavy is the booster; this keeps the
existing upper-stage vehicle. The same policy flies the vehicle and decides when to close or reopen the arms.
Its final action is a Bernoulli switch: 1 closes both arms, 0 opens them.

```sh
uv run starship-train --config-name catch_local
uv run starship-diagnose catch_policy_local --probe --device cuda:0
```

Stage 0 starts upright near the arms with engines lit. Later stages add position/heading error,
ignition, the belly-flop and flip, randomized 1.5–5 km descents, wind, then entry handovers at
5, 8, 12 and 20 km. Promotion requires 90% catches in two evaluations. The final stage is a
late-entry handover, **not an orbital re-entry simulation**. No catch stage is demonstrated solved
until evaluation passes it. Start a fresh run for the policy-controlled task: automatic-arm, fixed-arm and ground-task
checkpoints carry incompatible curriculum meanings and best scores. `checkpoint=` resumes the
same policy-controlled curriculum.

All `sim.catch` dimensions and rates are **training assumptions**, not measured SpaceX hardware:

- Arms pivot horizontally about their tower-side ends, opening outward by 30° at reset.
- Their support surface is at 80 m. Each arm is 28 m long, 3 m wide and 1 m thick. Closed arm
  centerlines are at ±7 m, leaving an 11 m clear gap. The modeled lower fins span 15.4 m at 45°,
  so they cannot pass between closed arms in that pose.
- The policy controls timing directly, without a clearance or alignment override. Closing too early
  can strike the lower fins; closing too late misses the pins. It can reverse its command at any step.
- Closing/opening is limited to 15°/s: full closure takes two seconds. The motor never teleports
  arms or disables collisions.
- Pins are at body station 35 m and ±7 m laterally. Both must arrive descending at ≤2 m/s,
  with ≤1 m/s horizontal pin speed, ≤5° tilt, ≤5°/s angular rate, ≤0.75 m horizontal alignment
  error, and ≤0.5 m pin height difference. Open arms cannot register a catch.

Collision checks include all four hinged fins, using the aero/render geometry as oriented boxes
with 0.1 m thickness, plus the hull as a conservative full-radius capsule. A separating-axis test
checks fin boxes against the tower and rotated arms. Midpoint shapes are inflated by translation
and rotation bounds over each physics substep to conservatively detect swept contact. This may
reject some near misses; it does not integrate contact forces. Ground contact and a missed catch
plane fail. Capture is tested at the first descending pin-plane crossing and ends the episode.
Vertical tower motion, structural compliance, post-capture support, engine shutdown, heating and
catch-pin impact loads remain unmodeled. This is an approach/capture training model, not a
validated real-world catch controller.

A scripted upright 2 m/s passage with 45° fins clears the open arms, allows full closure and reaches
the capture plane without hull or fin collision. It is a geometry feasibility test, **not a learned
flight**. Regression tests also verify that the same fins hit closed arms, nearly folded fins fit,
closure takes time, and arriving with open arms fails.

Observations retain physical altitude/air density and a target-relative vector, three body-frame
world-+x direction channels, and normalized arm opening (32 channels total). Ground tasks append
zeros. Actions retain the original 11 flight controls and append the arm-close switch (12 total).
Older 24/28/31-channel inputs and 11-action outputs, including Adam moments, load by zero-padding;
the added switch defaults to open for deterministic playback. This does not establish transfer. The moving-task speed potential brakes 25 m above the catch plane to leave closure time.
`rate/caught` is success; `rate/landed` remains its alias for curriculum/rollback. Rerun displays the
actual per-episode arm angle, including terminal state. `moving=false, flap_collisions=false` is
retained only to reproduce the original fixed-arm baseline. With moving arms,
`policy_controlled=false` reproduces the automatic tower baseline: close after fin clearance and
alignment, otherwise reopen. New catch runs use `policy_controlled=true`.

The local preset uses 512 environments, 128-step recurrent rollouts and 16 minibatches. The older
1,024-env / 8-minibatch run developed allocation spikes as episode lengths changed, despite fitting
initial smoke tests. Run metadata and training status are in [curriculum.md](curriculum.md).

## Inspect catch decisions

Replay a specific numbered checkpoint with its paired model file (the probe reads that checkpoint's
stage, not the last or final stage):

```sh
uv run python -m starship.train.catch_probe logs/<experiment>/<run>/policy_1499.pt --rerun
rerun logs/<experiment>/<run>/catch-probe/policy_1499/policy-seed-0.rrd
```

The default eight seeds run under three arm-command modes: the policy, forced open, and the
automatic clearance/alignment rule sampled at the policy control interval. The flight policy
continues reacting in every mode, so trajectories can diverge. This isolates an intervention on the
arm command, not a guarantee of the same subsequent flight. No training parameters are changed.

The output directory contains a Markdown comparison, JSON summaries, and per-step CSV traces.
`--rerun` also creates three seed-0 recordings with flight/timing, raw/normalized-input and reward
views. Decision metrics are timestamped before the action; post-step metrics and the terminal
snapshot use their actual simulation time. Negative fin clearance means the aft fins have not
fully passed below the arm underside; it is not itself a collision. Pin ETA assumes constant
current pin velocity and excludes future acceleration/rotation changes.

The simulator now labels the first detected swept contact: hull, aft fin or forward fin against
tower or arms. Separate labels identify failed pin-plane capture constraints and ground contact.
These labels survive terminal capture and autoreset but are not policy observations. Conservative
swept contact can include near misses; the labels describe the simulator's termination decision,
not physical impact loads.

The 32 policy observations already include COM altitude, velocity, attitude, fins, catch-pin-center
target vector, tower heading and arm opening. Individual pin heights, fin clearance, closure time
and contact labels are currently inferred rather than explicit policy inputs. The CSV includes the
learned normalization as well as raw values; small raw scales alone do not prove missing information.

## Explicit geometry observation experiment

`--config-name catch_geometry_local` enables `obs.catch_geometry=true`, appending six inputs
to the original 32. The default `catch_local` remains the 32-input comparison.

| Channel | Meaning | Scale / bound |
|---|---|---|
| 32, 33 | each catch pin above support | /50 m, clipped to ±10; swap under reflection |
| 34 | arm underside minus highest aft-fin point | /50 m, clipped to ±10; positive means clear |
| 35 | maximum horizontal pin alignment error | /10 m, capped at 10 |
| 36 | current arm angle / motor angular speed | /2 s |
| 37 | earliest descending-pin constant-velocity ETA | /10 s, capped at 3; -0.1 if unavailable |

The features use current geometry and pin-point velocity, not future trajectories or a scripted
closure command. The policy still controls the motor and can collide. Ground tasks append zeros.
Terminal observations preserve these channels before autoreset. Existing 24/28/31/32-input actors,
critics, normalizer statistics and Adam moments can be padded to 38 without changing their
original outputs; loading a 38-input policy into a 32-input environment is rejected.

`--config-name catch_contact_local` tests an additional 100-point structural-contact penalty,
using the simulator's contact label rather than a geometric heuristic. Default
`reward.catch_contact_penalty=0` preserves earlier checkpoints' reward definitions. This preset
retains the base 32 observations and initializes the critic bias at the -300 contact floor.

## Demonstration warm-start

`uv run python -m starship.train.catch_pretrain --output logs/catch_demo_pretrain/<new-run>`
collects stage-0 reference-controller flights through the normal action interface, trains the
recurrent actor and critic, and evaluates the neural policy alone every 100 supervised updates.
The reference controller uses simulator state during data collection; it is absent from exported
policies and evaluations. Collision checks, actuator limits and policy-controlled arms stay active.
This is a short-approach teaching tool, not a controller for the full belly-flop curriculum.

By default, four additional dataset-aggregation rounds label states reached by mixed teacher/policy
rollouts (teacher action probabilities 1/2, 1/4, 1/8, 1/16). Each round performs 2,000 supervised
updates; `--rounds 0` isolates ordinary behavior cloning. Complete recurrent episodes are kept
separate and padding is masked. Demonstration tensors, evaluation history and best evaluation
metadata are saved beside the paired `model_-1.pt` / `policy_-1.pt` initialization checkpoint.
These supervised updates are not reported as PPO iterations or PPO environment steps. Resume
normal training with `checkpoint=/absolute/path/to/model_-1.pt`; PPO starts at iteration zero.
Teacher catch rate and standalone-policy catch rate are separate metrics; only the latter can
support a claim that the learned policy catches.

For a validated demonstration warm-start, `--config-name catch_demo_local` preserves the actor
and critic observation statistics (`runner.algorithm.freeze_observation_normalization=true`)
and permits continuous exploration std down to 0.005. The checkpoint's saved log-std still wins;
set it deliberately when preparing the initialization checkpoint. Ordinary PPO presets continue
updating normalization. Changing observation scales after cloning can invalidate the learned
controller even when all network weights are held fixed.

## Audit observation scales and critic predictions

`uv run python -m starship.train.learning_audit RUN/model_2999.pt --episodes 256` loads the
paired actor/critic checkpoint and its actual curriculum stage. It records complete deterministic
on-policy episodes, comparing critic predictions with discounted Monte Carlo returns (no
bootstrap). `--stage N` explicitly tests that checkpoint on a different stage. Reports include
explained variance, bias and RMSE across all states and separately at episode start, raw and
normalized per-channel ranges/outliers, engine use, and initial stopping-distance estimates.
The latter assume constant initial mass and vertical full thrust; they omit spool delay, tilt,
drag, fuel burn and collisions and are not a feasibility proof. TensorBoard metrics are restricted
to steps at or before the selected checkpoint; bootstrapped training EV and Monte Carlo EV are
different diagnostics. JSON, observation CSV and complete trajectory arrays are written under
`RUN/learning-audit/` so results remain inspectable. Zero return variance yields undefined EV,
not a spurious success score.

`--config-name catch_bridge_local` separately tests four intermediate approaches before the
original catch stage 1: 115 m/-8 m/s, 130 m/-12 m/s, 160 m/-18 m/s, and 200 m/-25 m/s. Original
later catch stages are preserved. This uses the validated warm-start, frozen normalization and
narrow exploration; it does not reset the original catch curriculum or claim later stages solved.

`uv run python -m starship.train.critic_calibrate RUN/model_N.pt --audit RUN/learning-audit/model_N
--output NEW_DIRECTORY` tests whether the critic can fit those complete returns while preserving
the actor and both observation normalizers. It checks the audit's checkpoint SHA-256, splits whole
episodes into training and validation sets, corrects the mean bias using only training episodes,
and selects by validation RMSE. It clears stale critic optimizer moments while retaining actor
moments and the original PPO iteration. The saved `model_calibrated.pt` is a supervised critic
experiment, not additional PPO progress. Run a fresh-seed learning audit before continuation;
validation-split performance can overstate generalization. This tool does not invoke the teacher.
