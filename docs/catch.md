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
