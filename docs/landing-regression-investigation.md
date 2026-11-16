# Historical landing investigation

The available evidence points first to a changed training procedure, not a demonstrated
regression in ground-flight physics. This is a diagnosis of differences and a controlled
reproduction plan, not proof that any single difference caused the failure.

## Historical reference and limits

Commit `9947040dcb9a49816172bd2e63e316d3f4dbd1d3` (PR52, October17) records
98–99% randomized stage4 landings. Its contemporary curriculum notes identify the
successful belly-flop changes:128-step rollouts,4096 environments, actor learning
rate3e-5, KL stop0.01, and later GAE lambda0.98. They also record earlier failed
experiments, including32-step rollouts and a destructive1e-4 learning rate.
The original weights are gone. These are versioned results and configuration evidence,
not a freshly revalidated historical policy. The final committed settings do not
reconstruct every earlier resume command or the whole multi-run training sequence.

The current lineage started afresh with catch demonstrations and then rebuilt ground
skills; it never inherited the historical successful controller. The initial ground
rebuild itself used2048 environments,128 steps,16 minibatches and position scale30;
it was already different from the successful configuration before later interventions.
Do not attribute that earlier difficulty solely to the current retention guards.

## Differences from the successful belly-flop phase

The recent resolved configuration is
`landing_rollback_horizon_control_local/2026-11-16_20-45-33/.hydra/config.yaml`;
its reproduction script also supplies the mixture, replay and selection rules.

| Setting | Historical successful phase | Recent bridge experiment |
| --- | --- | --- |
| Environments × rollout steps |4096 ×128 |256 ×256 |
| Transitions per update |524,288 |65,536 |
| Environments on new steep-angle task |whole current stage |32/256;176 remain on earlier frontier |
| New-task transitions per update |524,288 |8,192:64× fewer |
| Actor / critic learning rates |3e-5 /3e-5 |1e-6 /1e-5 |
| Observation statistics |updated during learning |frozen inherited statistics |
| Position denominator |3000m |30m |
| Observation / action widths |24 /11 |32 /12, including unused ground-task arm |
| GAE lambda |0.98 |0.995 |
| Near-ground tilt shaping height |300m |1000m |
| Curriculum |advance after90% in two evaluations |manual mixture; automatic promotion effectively disabled |
| Rollback |50% relative drop for six evaluations, evaluated every25 updates |absolute retained-skill95% gates and speed budgets, evaluated every five updates |
| Extra replay |none |four shared-Adam throttle-mean replay steps per PPO update |

The last completed500-update block restored96 times:92 stretches of five updates
and four of ten. Thus it tested repeated short adaptation attempts, not500 consecutive
updates. Its total new-angle data was about4.10M transitions before accounting for
restores;125 historical full-stage updates collected65.54M. Iteration counts alone
are misleading. These ratios are not a prediction of speedup or proof of causation.
The recent rollout window is longer, so the original32-step BPTT problem is not the
same problem today.

Changing a raw denominator is not intrinsically harmful when normalization adapts.
But frozen near-upright statistics and a changing attitude distribution deserve a
controlled test. Existing audits report finite normalized inputs, sometimes with
p99 magnitudes around60–70; this does not by itself prove saturation. The completed
block showed no dead/dormant units or collapse in feature rank, so irreversible loss
of plasticity remains unproven.

## Direct simulator and reward comparison

`tools/landing_history_probe.py` replays identical seeded actions against each checkout,
using the original cumulative landing stages0–4, original default observation scales,
16 environments per stage and400 steps. It compares the common24 observation channels,
body state, first eight reward components, total reward, terminal statuses and masks.
Both versions ran in the same installed environment; `uv.lock` and `pyproject.toml`
are unchanged between the reference commit and the investigated code.

Across32,000 matched transitions per backend:

- CPU: all recorded arrays are bit-for-bit identical.
- RTX3090 CUDA: all terminal statuses/masks match. Maximum position difference is
  0.0001221m, velocity difference1.53e-5m/s, reward difference0.0002666, and common
  observation difference3.88e-7.

This finds no substantial ground-physics/reward regression in these probes. It does
not prove equivalence for every trajectory, trained policy, landing boundary, wind
case, catch geometry, optimizer update, or the historical installed runtime. Random
actions do not reproduce a skilled landing distribution. The current30m denominator,
frozen normalization and modified bridge reward are intentionally not used here:
this isolates code changes from training-configuration changes.

For reproduction, export the reference with `git archive 9947040` into a separate
directory. Run the same probe twice with `PYTHONPATH=<checkout>/src` and the shared
Python executable, passing `<checkout> <output>` as arguments. Set
`PROBE_DEVICE=cuda:0` for GPU. The probe rejects an incorrectly imported checkout.
Preserve the CUDA driver library path used by local training. Compare each backend
with `python tools/compare_landing_history.py <reference-output> <current-output>`;
the comparator rejects mismatched runtime settings, shapes and nonfinite arrays.

Results, resolved configs, probe, package/runtime provenance and SHA-256 manifest:
`/mnt/storage2/chris/starship-checkpoints/landing-history-investigation-2026-11-16`.

## Additional mechanism worth isolating

The arm action has no physical effect when catching is disabled, but
`HybridDistribution` still includes it in joint log probability, entropy and KL.
Consequently it can add gradient variance and affect PPO clipping or early stopping.
A constructed check changing only arm probability0.5→0.75 yields analytic and
balanced-sample KL0.14384, above the configured0.01 stop threshold, with physical
controls unchanged. This demonstrates coupling, not that trained arm drift is large
or that removing it would solve learning. No arm mask or optimizer fix is adopted
without measuring actual per-head contributions and testing retention.

## Next experiment

1. Establish a clean reference using the historical checkout and its successful final
   PPO settings, starting from scratch with the original curriculum. Preserve passed-stage
   checkpoints. This is a reconstructed baseline, not an exact replay of all historical runs.
2. Evaluate fresh deterministic and sampled full episodes; retain original physical limits.
   Compare environment transitions and elapsed time, not iteration numbers alone.
3. Once the reference is learnable, add modern inputs/arm output under matched training
   settings, isolating input changes from the inactive action head. Then test normalization
   freezing, mixture allocation and strict restore behavior separately, rather than combining
   all differences into another continuation.

The existing rollback-horizon experiment addresses one hypothesis only. At this
investigation, its controller has been interrupted and no shadow run is present;
no training was restarted. The agent goal remains paused. The investigation does
not establish end-to-end landing or catching, and PR67 remains unmerged.
