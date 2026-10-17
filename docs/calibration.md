# Held-descent calibration

`starship.sim.calibrate` fits hull crossflow drag, flap effectiveness, dry mass and initial
flight-path angle with cross-entropy search. Each candidate is one Warp environment using the
production physics and a pitch PD flap controller. The two new aerodynamic scales default to
1.0; no fitted values are installed into the training configuration.

## Run

Place SofieBrink's MIT-licensed
[StarshipTelemetryExtractor CSVs](https://github.com/SofieBrink/StarshipTelemetryExtractor/tree/main/Examples)
in a directory as `IFT-4_Telemetry.csv`, `IFT-5_Telemetry.csv`, `IFT-6_Telemetry.csv`, then run:

```sh
WARP_CACHE_PATH=$PWD/.warp_cache uv run python -m starship.sim.calibrate \
  --flight IFT-5 --device cpu --data-dir /path/to/telemetry
```

`--device cuda:0` uses the same batched kernel on CUDA. `--raw` selects raw readings; corrected
readings are the default. `--population`, `--iterations`, `--seed`, `--dt` and `--out` are optional.
Defaults are 48 candidates, 10 iterations, seed 7, 0.02 s physics steps and `out/`.
The CLI fits only the named flight and evaluates all three with the same four parameters.

Outputs are `out/calibration_IFT-5.json` and `out/calibration_IFT-5_IFT-{4,5,6}.csv`.
The JSON includes bounds, timing, window selection, quality fractions and RMSE scope. CSVs retain
both channel quality masks; predictions after early termination are NaN with `sim_valid=0`.

## Measurement and simulation assumptions

- Time is frame number / 30; IFT-6's `MissionTime` formulas are ignored. Duplicate frames are
  removed. Missing, negative and isolated outlier readings are interpolated and masked.
  Differences between corrected and raw values are also masked, independently for each channel.
  Masked measurements do not enter either RMSE.
- The window starts at the first descending 20 km reading after exceeding 25 km. The final
  sustained speed drop below 3 km, within 40 seconds of near-zero speed, identifies braking.
  The window stops 2.5 seconds before this inferred onset. This is not an engine-ignition label.
- Altitude assumes nearest-km display quantization: error is zero inside the displayed value
  ±500 m. The objective is `(altitude_interval_RMSE / 500 m)^2 + (speed_RMSE / 10 m/s)^2`.
  These are fitting weights, not independent measurement-noise estimates. Display rounding vs
  truncation and the exact initial altitude remain uncertainties; spawn uses the displayed 20 km.
- The body starts horizontal, belly down; initial velocity points along the nose heading and
  down at the fitted angle. PD gains are 60° per unit vertical-axis component and per rad/s pitch
  rate; nominal aft/forward commands are 52°/20°, correction limited to ±0.7 rad. The controller
  updates every 0.1 s; predictions at 1 s are interpolated to telemetry frame times.
- Main propellant is zero, header propellant 23 t. Geometry remains the repository's V2 proxy,
  although these flights used V1 ships. Mass properties are recomputed for each candidate.
- Bounds: hull multiplier 0.5–1.8, flap multiplier 0.5–1.5, dry mass 100–200 t, angle 40–90°
  below horizontal. Broad mass bounds permit detecting disagreement with the 120–165 t reference.
  All four parameters, including entry angle, are transferred unchanged to validation flights;
  each flight supplies its own initial altitude and scalar speed.
- Premature ground contact, tumbling or bounds termination disqualifies a search candidate.
  Validation failures report only the valid prefix and are explicitly marked incomplete.

## CPU result, 2026-10-17

Default search on IFT-5 (corrected CSV): about 4.1 seconds of search with a warm Warp cache,
well below 15 minutes. Compilation and CSV loading are additional. No validation refitting.

| Parameter | Fit |
|---|---:|
| Hull crossflow drag multiplier | 0.944084 |
| Flap force multiplier | 1.278805 |
| Dry mass | 141,455.5 kg |
| Initial descent angle below horizontal | 64.0732° |
| IFT-5 initial horizontal speed | 121.33 m/s |

| Flight | Frame-clock start (s) | Window (s) | Altitude interval RMSE (m) | Speed RMSE (m/s) | Completion |
|---|---:|---:|---:|---:|---|
| IFT-5, fitted | 3785.033 | 139.500 | 173.09 | 4.163 | Full window |
| IFT-4, validation | 3809.833 | 126.633 | 643.69 | 6.806 | Full window |
| IFT-6, validation | 3764.700 | 146.067 | 169.74 | 8.144 | Prefix through 144.800 s only; early ground contact |

The dry mass is within the [reference](starship.md) range of 120–165 t. At sea level, 100 m/s,
90° AoA and nominal PD flap commands, the fitted model has a drag area of **307.6 m²**, inside
the reference 230–340 m² band around 280 m². With all flaps flat it is **365.2 m²**; drag area
must therefore be quoted with a flap setting. Hull scaling is modest; the +27.9% flap correction
is outside the research note's ±20% prior and may compensate for the assumed control/geometry.

These are effective fit parameters, not independent estimates of physical mass and flap loads.
Scalar speed and coarse altitude cannot identify all four uniquely. IFT-4's altitude mismatch
and IFT-6's early ground contact limit transferability. Keep the training defaults unchanged;
attitude/flap observations, per-flight geometry and initial altitude/angle uncertainty would be
needed before adopting this as a physical calibration.

Halving the physics step to 0.01 s at the fitted parameters gives IFT-5 RMSEs of 173.07 m and
4.1634 m/s. Maximum trajectory differences are 0.49 m and 0.0133 m/s.

Tests cover OCR cleaning/windowing, quantization masks, batched versus individual candidates,
termination rejection and synthetic recovery of drag and angle with mass/flaps fixed. A small
pre-change arm64 CPU Warp 1.18.0 fixture verifies all 24 snapshot arrays bit-for-bit, including
horizontal drift and rotational dynamics; this platform-specific golden test skips other CPU
architectures. CUDA execution was not available on this machine.
