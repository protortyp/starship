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

## First measurements

Stage 4, seed 0, existing best landing policy, continuous command noise 0.1,
two hidden layers of width 256, 8-step fitting windows. Validation environments
are disjoint from fitting environments. Errors below are endpoint RMSE on
nonterminal, open-loop flight windows; they do not measure landing success.
Position and velocity RMSE average over coordinates. Sensitivity comparisons use
the model's physical channel scales, separately within each reported group.

| Run | Transitions | Updates | Horizon | Position (m) | Velocity (m/s) | Attitude (deg) |
|---|---:|---:|---:|---:|---:|---:|
| Small feasibility run | 65,536 | 2,000 | 0.1 s | 0.32 | 0.13 | 0.63 |
| Small feasibility run | 65,536 | 2,000 | 0.8 s | 2.51 | 0.96 | 4.77 |
| Small feasibility run | 65,536 | 2,000 | 3.2 s | 8.96 | 3.33 | 14.38 |
| Larger feasibility run | 524,288 | 5,000 | 0.1 s | 0.16 | 0.08 | 0.31 |
| Larger feasibility run | 524,288 | 5,000 | 0.8 s | 1.08 | 0.52 | 2.06 |
| Larger feasibility run | 524,288 | 5,000 | 3.2 s | 4.26 | 1.69 | 7.65 |

The small run predicts trajectories substantially better than holding state
constant, but its action Jacobians are not trustworthy. For example, at 0.8 s
and perturbation 0.01, the forward-starboard flap's rotation cosine is -0.51
(54 measurable samples), while throttle's translation cosine is 0.74 (47
samples). Roll RCS has badly wrong sensitivity magnitude. Thus trajectory
accuracy alone does not establish useful policy gradients. These exploratory
numbers are one seed, with no confidence intervals or policy improvement claim.

Sensitivity reports separate translation, rotation, actuator states and tanks;
actuator accuracy must not hide wrong vehicle response. They also count invented
responses where the simulator's derivative is near zero. Both perturbation sizes
use the same windows, and unperturbed simulator replay checks state restoration
and command mapping. The small run reproduces recorded position, velocity,
angular velocity and tanks exactly on its probe windows.

The larger run uses 256 environments and 2,048 collected steps. Re-evaluation
uses seed 123, 1,024 held-out windows per prediction horizon and 256 per
sensitivity horizon. At 0.8 s, the forward-starboard flap's rotation cosine
remains -0.63 (232 measurable samples, perturbation 0.01) and -0.59 (229,
perturbation 0.003). Gimbal rotation cosines are 0.68 and 0.59 at perturbation
0.01; roll RCS relative rotation error is approximately 696. Unperturbed replay
at this horizon reproduces position exactly, velocity to 3e-7 m/s RMSE,
attitude to 2e-6 degrees and tank masses exactly. The bad model sensitivities
therefore persist after checking replay and perturbation size.

**Decision:** do not feed this model's gradients to the policy. More data and
fitting improve endpoint accuracy without fixing important action derivatives.
Next test: independent counterfactual action perturbations from the same saved
states, with balanced flip/ignition/landing coverage, to separate action effects
from correlations in successful-policy trajectories. Compare to this same model
and held-out sensitivity protocol before implementing imagined policy updates.
An analytical kinematics/actuator component is another candidate if action
coverage alone does not help. No landing-rate gain has been measured.
