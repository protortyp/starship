# Curriculum

Run: `just vast train curriculum=spawn checkpoint=<model_*.pt> ...` (stages in `src/starship/train/conf/curriculum/spawn.yaml`).
Each stage adds one change on top of all earlier ones. Promotion: stage eval lands ≥ 90% in 2 consecutive evals.
TensorBoard: `curriculum/stage` (steps at promotions), Text tab `curriculum`, `eval/env/*` (current stage), `eval/target/*` (final stage). Run dir: `curriculum.json`, `model_stage<k>_passed.pt`, `policy_best.pt`.

## Stages

| # | change | status |
|---|---|---|
| 0 | randomized 3 km belly-flop (±300 m, ±10 m/s, ±11°), 20% near-ground practice spawns | solved (stage_c, it 1149: 100%) |
| 1 | time cost 0.25/s | |
| 2 | mirrored spawns (nose left/right) | |
| 3 | altitude 1.5–5 km | |
| 4 | vertical speed −40 to −130 m/s | |
| 5 | horizontal speed ±30 m/s, offset ±500 m | |
| 6 | orientation flat ±30° | |
| 7 | spin ±5°/s | |

## Findings

- Controlled runs (300 it from `model_stage1_passed.pt`, promotion off, one change each; same checkpoint + seed replay identically, so differences are causal): lateral shaping near the ground (`reward.lateral_weight=0.05`) is the only change that recovers (eval ends 74–82%); lr 1e-4 + clip 0.1, 2 learning epochs, and soft touchdown credit (`reward.soft_touchdown_speed=10`) all stay collapsed.
- CLI values lose to curriculum stage overrides for keys a stage sets (e.g. `reward.time_cost`); run controlled experiments without `curriculum=` and set the stage's values directly.

- Stage 0 cannot be held long enough to be promoted (curriculum run collapsed at it ~1399 exactly like stage_c_stable: same checkpoint and seed replay the same drift), so runs start at stage 1 with `curriculum.start_stage=1`.
- Switching on mirrored spawns as something to learn collapsed the policy (84% → 2% within 10 iterations, curriculum_v4 it 1399). Mirrored episodes are now encoded in the mirror frame (observations flipped, left/right thrusters swapped), so the policy sees each mirrored start as its unmirrored twin.
- Collapses were the speed term of the shaping potential rewarding braking at any altitude: policies braked and hovered at 0.5–1.5 km until the fuel ran out (traced curriculum_v3 it 1449). PPO's ~2 s GAE horizon follows shaping greedily, so the speed term now fades below 300 m (`reward.speed_fade_altitude`).
- Policies are good deterministically but poor when sampling (best policy: 100% vs 9–14%), so training rollouts mostly crash; entropy_coef 0.002 makes them decisive.
- Without a time cost, long runs drift into ever slower descents and collapse (stage_c at it ~1175, stage_c_stable at lr 1e-4 at it ~1399); `policy_best.pt` keeps the peak.
- Learning from the full 3 km scenario alone fails; near-ground starts first (stage_a, stage_b) made it learnable.
- Fixed lr 3e-4 learns; rsl_rl's adaptive schedule pinned lr at 1e-2 and stalled.
