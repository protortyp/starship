# Upper-stage chopstick catch

`curriculum=catch` trains the existing Starship upper stage to descend onto two fixed arms.
Super Heavy is the booster; this does not replace the vehicle with booster physics.

```sh
uv run starship-train --config-name catch_local
uv run starship-diagnose catch_local --probe --device cuda:0
```

Stage 0 starts upright near the arms with engines lit. Later stages add position/heading error,
ignition, the belly-flop and flip, randomized 1.5–5 km descents, wind, then entry handovers at
5, 8, 12 and 20 km. Promotion requires 90% catches in two evaluations. The final stage is a
late-entry handover, **not an orbital re-entry simulation**. No catch stage is demonstrated solved
until evaluation passes it. Train from scratch for this new task; `checkpoint=` resumes the same
catch curriculum, including its stage. Do not resume a ground curriculum checkpoint directly:
its stored stage index and best score describe a different task.

All `sim.catch` dimensions are configurable **training assumptions**, not measured SpaceX
hardware: arms at 80 m, pins at body station 35 m and ±7 m laterally, a 0.75 m horizontal capture
tolerance and 0.5 m permitted pin height difference. Both pins must arrive descending at ≤2 m/s,
with ≤1 m/s horizontal pin speed, ≤5° tilt and ≤5°/s angular rate. The two symmetric headings
are equivalent. Heading is fixed near alignment initially and randomized later.

Capture is evaluated at the interpolated first descending pin-plane crossing, including angular
velocity in each pin's contact speed. A missed catch plane ends the attempt as a failure. Ground
contact always fails in catch mode. The tower and arms reject hull collisions using a conservative
full-radius capsule/expanded-box test (including the nose). Flaps, protrusions and catch-pin impact
loads are not collision meshes. Arms do not move; a catch is terminal, without support, compliance,
engine shutdown, heat or structural-load integration. This teaches an approach and capture envelope,
not a validated hardware catch controller.

Observations retain physical altitude/air density, direct the target vector to the catch midpoint,
and append three body-frame world-+x (away from tower) direction channels (31 total). Ground tasks append zeros.
24- and 28-channel actors/critics and Adam moments migrate by padding; this preserves ground-policy
outputs but does not establish transfer performance. Catch reward uses height above the pins' capture
plane, target distance and orientation alignment. `rate/caught` is the explicit success metric;
`rate/landed` is retained as its alias for existing curriculum/rollback code. `catch/pin_error_m`
measures worst pin horizontal alignment. Rerun shows the fixed tower, arms and vehicle pins.
