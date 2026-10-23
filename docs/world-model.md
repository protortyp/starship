# Differentiable world-model experiment

Investigate [SuperMimic](https://arxiv.org/pdf/2610.28480)'s learned-dynamics policy
gradients for our landing controller. Keep the existing PPO trainer and exported
policy interface as the baseline. This is an experiment, not a claim that its
humanoid compute gains transfer to this simulator.

## Sequence and acceptance criteria

1. Collect full physical state and executed actions from the existing landing
   policy, preserving terminal transitions before reset. Split by environment,
   not overlapping windows, for held-out validation.
2. Fit a normalized residual dynamics model on autoregressive 8-step windows.
   Report held-out 1/8/32-step physical errors and action-sensitivity agreement
   with simulator finite differences, especially around ignition and the flip.
3. Only after useful gradients are demonstrated, add short imagined actor-critic
   updates. Start with the binary engine policy fixed and continuous controls
   trainable. Reconstruct recurrent state with replay burn-in. Retain exact
   simulator evaluations and rollback.
4. Compare against PPO on equal GPU-hour budgets and several seeds: landing
   rate, fuel, touchdown quality, roll travel, wind robustness and regressions
   on earlier stages. Keep model training and data collection in the budget.

The model needs individual tank masses, filtered actuator states and wind/filter
state, beyond the 28 actor observations. Reward history and episode context must
also be available before imagination training. Stochastic wind innovations must
be sampled explicitly or treated as uncertain; predicting an average turbulent
trajectory is not a deterministic dynamics oracle. Start with calm stage 4.

At 10 Hz, 32 steps span only 3.2 seconds. Do not shorten the existing policy's
recurrent training window based on the paper's horizon. Binary ignition cannot
receive an ordinary pathwise gradient; any relaxation needs separate validation.

## Status

Initial implementation: replay collection and model/gradient feasibility harness.
Policy updates remain gated on measured simulator agreement. GPU results and the
current next step are recorded here and in the draft PR description.
