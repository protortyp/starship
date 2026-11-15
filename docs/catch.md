# Upper-stage chopstick catch

`curriculum=catch` trains the Starship upper stage to descend between open arms, let its lower
fins pass, and reach the pins after the arms close. Super Heavy is the booster; this keeps the
existing upper-stage vehicle. The tower controller operates the arms automatically; the policy
flies the vehicle and observes arm opening.

```sh
uv run starship-train --config-name catch_local
uv run starship-diagnose catch_moving_local --probe --device cuda:0
```

Stage 0 starts upright near the arms with engines lit. Later stages add position/heading error,
ignition, the belly-flop and flip, randomized 1.5–5 km descents, wind, then entry handovers at
5, 8, 12 and 20 km. Promotion requires 90% catches in two evaluations. The final stage is a
late-entry handover, **not an orbital re-entry simulation**. No catch stage is demonstrated solved
until evaluation passes it. Start a fresh run for the moving-arm task: fixed-arm and ground-task
checkpoints carry incompatible curriculum meanings and best scores. `checkpoint=` resumes the
same moving-arm curriculum.

All `sim.catch` dimensions and rates are **training assumptions**, not measured SpaceX hardware:

- Arms pivot horizontally about their tower-side ends, opening outward by 30° at reset.
- Their support surface is at 80 m. Each arm is 28 m long, 3 m wide and 1 m thick. Closed arm
  centerlines are at ±7 m, leaving an 11 m clear gap. The modeled lower fins span 15.4 m at 45°,
  so they cannot pass between closed arms in that pose.
- Both lower fins must clear the underside by 0.5 m before closure starts. The ship must be
  within 2 m of the target and within 10° of upright, with pins still above the support surface.
- Closing/opening is limited to 15°/s: full closure takes two seconds. Unsafe alignment commands
  reopening, also at the finite rate. The controller never teleports arms or disables collisions.
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
zeros. 24/28/31-channel networks and Adam moments load by padding, without claiming successful
transfer. The moving-task speed potential brakes 25 m above the catch plane to leave closure time.
`rate/caught` is success; `rate/landed` remains its alias for curriculum/rollback. Rerun displays the
actual per-episode arm angle, including terminal state. `moving=false, flap_collisions=false` is
retained only to reproduce the original fixed-arm baseline.

The local preset uses 512 environments, 128-step recurrent rollouts and 16 minibatches. The older
1,024-env / 8-minibatch run developed allocation spikes as episode lengths changed, despite fitting
initial smoke tests. Run metadata and training status are in [curriculum.md](curriculum.md).
