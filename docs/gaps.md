# Simulation gaps

Physics not modelled in `starship.sim`. Refs point to `docs/starship.md` sources.

| Gap | Effect | Approach | Refs |
|---|---|---|---|
| Wind and gusts | Lateral drift, AoA disturbances | Mean wind profile + Dryden turbulence in the air-relative velocity | MIL-F-8785C |
| Vortex shedding | Unsteady hull side force at St ≈ 0.2 (≈ 2 Hz at 85 m/s), 0.1–0.2 of crossflow force | Sinusoidal side force with random phase per env | [25] |
| Flap force noise | 10–20 % rms flap normal force | Correlated noise at St ≈ 0.1 | [12] |
| Plume–aero interaction | Changed hull and flap loads while engines fire into crossflow during the flip | No public data; scale crossflow force near the tail with thrust | [13] |
| Propellant slosh | CoM and inertia oscillation during flip and burn | Pendulum or spring–mass per tank | |
| Propellant settling | Liquid position follows acceleration, not the body axis | Fill tanks along the local acceleration direction | |
| Variable-mass terms | dI/dt and jet damping torques | Add ṁ·r² jet damping and İ·ω terms | |
| Supersonic / hypersonic regime | Spawns above ~20 km need M > 1 aero and heating | Extend crossflow and flap tables to M 10 | [12] [25] [27] |
| Engine failures | Relight failures, 2–5 % per engine | Random per-engine ignition failure | [19] [20] |
| Ignition conditions | Relight needs settled propellant and spin-up gas | Gate ignition on axial acceleration and tank state | |
| Header-tank feed limits | Feed switch from main to header tanks | Separate feed lines per tank | [6] |
| RCS propellant | Vent thrusters consume ullage gas | Ullage gas mass budget | [6] |
| Raptor 3 / V3 | Different thrust, mass, flap placement | Config preset once data exists | [14] [16] |
| Landing legs / tower catch | Touchdown loads, catch arms | Spring–damper legs or catch-pin geometry | [29] |
| Ground effect | Plume and flow changes near the pad | Thrust/aero correction below ~1 diameter | |
| Earth curvature, rotation | Coriolis, curvature over km-scale offsets | Negligible at < 10 km range | |
