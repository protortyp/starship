# Live catch-training PR

This branch is an ongoing **draft** training-progress PR. Leave it open while the local run is
active; GIFs and checkpoint metadata are refreshed in its description only, without comments.
The implementation PR was already merged separately.

`tools/catch_progress.py` loads matching `policy_N.pt` and `model_N.pt`, reads the checkpoint's
actual curriculum stage, and runs deterministic CPU rollouts for seeds 0 and 1. A headless Rerun
viewer renders the existing scene, tower and ship geometry. GIFs show both tower and chase views,
including the terminal state. No successful-episode filtering is used. Each complete episode is
resampled into 24 frames and played over four seconds; captions give actual simulated duration.

The worker uses a separate Git worktree on `ce/catch-training-progress`, updates only
`docs/assets/catch-progress`, runs `just check` and `just test` before committing, pushes the assets,
and updates the PR body with immutable image URLs. The default refresh interval is 100 training
iterations (6,553,600 environment steps for the current 512 × 128 batch), plus the final saved
checkpoint. Initial publication renders the latest saved checkpoint immediately. The latest GIFs
replace those in the description; older versions remain in branch history.

The local worker runs in tmux session `starship-catch-progress`. Stop only the updater with
`tmux kill-session -t starship-catch-progress`; training has its own `starship-catch` session.
Closing the PR or marking it ready also prevents further publication. A file lock prevents two
updaters from running concurrently. Transient failures retry without advancing the publication
marker, and the final checkpoint is published when the training session ends.

To restart (use the existing environment and this worktree):

```sh
/home/chris/dev/mine/starship/.venv/bin/python tools/catch_progress.py \
  --run /home/chris/dev/mine/starship/logs/catch_contact_local/2026-11-15_13-56-04 \
  --pr <progress-PR-number> --every 100
```

Worker log and publication state live in the training run's `pr-progress/` directory. Use
`--render-only --once` to prepare local assets without commits or GitHub writes.

The automatic-tower run was also stopped at 0% catches to start learning closure timing.
The fixed-arm run was stopped at iteration 1902 with 0% catches and is superseded. The updater now
follows `catch_contact_local`: the policy controls flight and arm closing/reopening. It uses the checkpoint's arm state and preserves terminal arm/flap/spool
state in the last frame. Historical baseline GIFs remain in branch history but are not evidence of
a physically valid catch.

The reward-only run keeps the original 32 observations and sets all catch failure penalty floors
to 200. GIF captions now use simulator contact labels for crashes.

The current critic-initialization experiment starts the value output bias at -200, retaining
the reward-only experiment's penalty floors and observations.

The current contact experiment adds 100 for confirmed structural collisions (300 total floor),
retains 32 inputs, and initializes the critic at -300. Earlier runs remain archived.
