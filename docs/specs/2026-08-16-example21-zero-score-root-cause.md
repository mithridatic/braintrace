# Example 21 zero-score root cause

Status: Phase 1 and Phase 2 complete. Phase 3 discriminator pending.

Subject: run `var/example21-full-scientific-final-v5`, commit `44ff7be`.

## Question

Example 21 reported exact ARC pass@1 = pass@2 = 0.0000 at every effort level.
The working hypothesis was insufficient capacity (2,048 LIF neurons, 16,384
recurrent edges). This document records the measurements that reject that
hypothesis and identify what actually happened.

## Finding

The run performed no learning. Every reported metric is what a randomly
initialized decoder emits. Capacity was never exercised and therefore cannot be
falsified by this run.

## Evidence

### 1. Exact ARC score carries no diagnostic signal

Trivial predictors scored through this example's own scorer
(`score_query_candidates` / `aggregate_arc_metrics`) on all 400 ARC-AGI-1
evaluation tasks / 419 queries:

| predictor | pass@1 | shape | pixel |
|---|---|---|---|
| `copy_input` | 0.0000 | 0.6611 | 0.6032 |
| `zeros_input_shape` | 0.0000 | 0.6611 | 0.4597 |
| `mode_color_input_shape` | 0.0000 | 0.6611 | 0.5017 |
| `zeros_demo_shape` | 0.0000 | 0.8687 | 0.4876 |
| `copy_or_rule_shape` | 0.0000 | 0.8687 | 0.6336 |
| `uniform_random_input_shape` | 0.0000 | 0.6611 | 0.1030 |
| pp-prop, effort 32 (measured) | 0.0000 | 0.0859 | 0.0628 |

Exact pass@1 is 0.0000 for every trivial predictor, including ones scoring 60%
pixel accuracy. Exact ARC success is therefore not usable as a debugging signal
at this stage; the shape and pixel diagnostics are.

The model scores below every floor, including uniform random.

### 2. The color head is uniform random

Candidate-one color histogram over all 419 effort-32 queries:

```
{0: .1142, 1: .1022, 2: .0914, 3: .1100, 4: .0730,
 5: .0836, 6: .1243, 7: .1048, 8: .0980, 9: .0985}
entropy = 2.2924 nats        ln(10) = 2.3026 nats
```

### 3. The shape head is a constant prior

```
predicted shapes: (10, 10) on 358 of 419 queries
corr(predicted height, true height) = -0.0385
corr(predicted width,  true width)  = +0.0042
```

### 4. The decode path is not mis-indexed

Restricting to the 36 queries where the predicted shape happened to match the
target:

```
pixel accuracy | shape correct = 0.1208     (chance = 0.10)
```

This is chance, not below chance. The aggregate 0.0628 is an artifact of the
scorer dividing overlap-region matches by `truth.size`: wrong shapes shrink the
overlap window and suppress the ratio. A color-axis argmax error, an off-by-one
color index, and a row/column transpose are all rejected by this measurement.

### 5. The loss sat at the uniform-random entropy floor throughout

Loss weights are `height: 1.0, width: 1.0, valid_cell_color: 1.0`. The
uniform-random cross-entropy floor is:

```
ln(30) + ln(30) + ln(10) = 3.4012 + 3.4012 + 2.3026 = 9.1050
```

Measured: first 16 updates mean 9.0576; last 16 updates mean 8.7878; per-step
noise approximately +/- 1.0, larger than the 0.27 trend across the whole run.

96 optimizer updates at lr 1e-4 against 2,130,716 parameters.

### 6. Effort levels are indistinguishable

Scores at efforts 0/8/16/32 are identical to four decimal places while firing
rate decays 0.2439 -> 0.1429 and readout entropy stays flat at 2.32. Physical
state evolves; no output-relevant information is carried by additional ticks.

## Rejected hypotheses

- **Insufficient capacity.** Untestable from this run. The parameters never
  left their initialization distribution.
- **Grid decode bug** (color argmax axis, color index off-by-one, row/column
  transpose). Rejected by measurement 4.

## Open hypothesis, and the test that discriminates it

Remaining split:

- **H1, undertrained.** The pipeline learns, and 96 updates is simply four
  orders of magnitude short.
- **H2, broken gradient path.** Grid loss cannot reach the recurrence. The
  compiler emitted four retained warnings on this run: `readout_projection.weight`
  is excluded from ETP for violating the non-parametric-tail invariant, and
  `height_head.weight`, `width_head.weight`, `color_factor_head.weight` each
  report no connected hidden states and are treated as non-temporal. All four
  output paths receive single-tick current-window gradients only. Measurement 6
  is consistent with this.

### Discriminator design

A plain single-task overfit is **not** a valid discriminator. H2 is an
assertion about a learning-rule property, namely that grid loss does not reach
the recurrence. Per the finite-window rule in `AGENTS.md`, such an assertion
passes vacuously when measured through a path with no truncation left. Here the
concrete failure mode is: `height_head`, `width_head`, and `color_factor_head`
all receive exact current-window gradients, so a single task with a fixed
target is learnable from the single-tick path alone. The heads memorize the
answer, pixel accuracy reaches 1.0, and H1 is falsely concluded while the
recurrent weights contributed nothing.

The discriminator must therefore separate the two paths:

1. **Cheap gradient probe, run first.** Compute `d(grid_loss)/d(rec_syn.comm.weight)`.
   If it is identically zero, H2 is confirmed with no training run at all and
   the four compiler warnings are the root cause. If it is nonzero, record
   whether its magnitude depends on `latent_steps`.
2. **Two-arm overfit.** Run the single-task overfit at `latent_steps=0` and at
   `latent_steps=32`. If both reach approximately 1.0 at similar rates, the
   recurrence is decorative and H2 survives regardless of the loss curve.
   Measurement 6 already predicts this outcome.
3. **Per-group parameter movement.** Log the L2 delta for
   `rec_syn.comm.weight` and `ff_syn.comm.weight` separately from the three
   heads. `_parameter_change_evidence` already emits this per group. Heads
   moving while temporal synapses do not is H2 directly.
4. **Task selection.** Choose a task whose output shape differs from its input
   shape, so a constant-shape prior cannot score.

Only the conjunction of "loss falls to approximately 0", "the `latent_steps=32`
arm beats the `latent_steps=0` arm", and "temporal synapses moved" supports H1.

Note on the effort lever: `ExperimentConfig.latent_steps` has a hard minimum of
32, so the zero-tick arm must be driven through the terminal supervision mask
(`mask[query_stop - 1 + effort]`), which is the same lever the frozen evaluation
uses for its 0/8/16/32 checkpoints.

## Step A result: H2 rejected

`docs/diagnostics/example21_gradient_probe.py`. One real ARC training episode
(task `007bbfb7`, query input 3x3, output 9x9, so a constant-shape prior cannot
score) driven through the production `learner.etrace_grad` call. Gradient L2 norm
per parameter group:

| parameter | effort 0 | effort 8 | effort 16 | effort 32 |
|---|---|---|---|---|
| `readout_projection.weight` | 1.0169e+01 | 9.1377e+00 | 7.6604e+00 | 6.1974e+00 |
| `width_head.weight` | 2.1501e+00 | 1.9035e+00 | 1.7160e+00 | 1.4371e+00 |
| `height_head.weight` | 2.1438e+00 | 1.8930e+00 | 1.7093e+00 | 1.4325e+00 |
| `rec_syn.comm.weight` | 1.7365e-02 | 3.7438e-02 | 3.2738e-02 | 2.4628e-02 |
| `ff_syn.comm.weight` | 2.0176e-02 | 3.1866e-02 | 1.7510e-02 | 3.7725e-03 |
| `color_factor_head.weight` | 1.2417e-02 | 9.9761e-03 | 5.0055e-03 | 2.2329e-03 |

Gradients are nonzero for every group at every effort and vary with effort. The
grid loss does reach the recurrent synapses, so **H2 is rejected in its strong
form**: it is not true that the loss cannot reach the recurrence.

Scope limit on this result. A single `etrace_grad` call over one episode is not
a finite-window measurement, so per the rule in `AGENTS.md` it cannot settle a
learning-rule property. A nonzero recurrent gradient is consistent with pure
current-window flow through the recurrent state at the terminal tick. The four
retained warnings assert something narrower than H2, namely that
`readout_projection` is excluded from *eligibility traces* and that the three
heads have no connected hidden states. This probe did not test that. The
measured claim is: the grid loss reaches the recurrent synapses; whether that
gradient carries temporal eligibility is untested and the warnings remain a
live hypothesis.

Two secondary findings that the aggregate report did not surface:

- **Gradient scale imbalance of roughly 2,800x.** `color_factor_head` receives
  the smallest gradient of any group while being responsible for approximately
  900 of the roughly 902 predicted values per grid. `readout_projection`
  receives the largest.
- **Learning signal decays as effort rises.** Every head's gradient shrinks
  monotonically from effort 0 to effort 32, tracking the firing-rate decay of
  0.2439 -> 0.1429. For an example whose thesis is that more latent ticks buy
  more reasoning, more ticks currently buy less learning signal. This is
  independent of the budget problem and should be treated as its own defect.

## H1 confirmed: the parameters never left initialization

True parameter shapes, measured by instantiating the model (total reconciles
exactly to the reported 2,130,716):

| group | l2 delta | n | rms move/coord | rms init | move / init sigma |
|---|---|---|---|---|---|
| `color_factor_head.weight` | 0.4411 | 144,480 | 1.160e-03 | 0.08795 | 1.32% |
| `ff_syn.comm.weight` | 0.7045 | 1,699,840 | 5.404e-04 | 0.13894 | 0.39% |
| `height_head.weight` | 0.1678 | 3,870 | 2.698e-03 | 0.08739 | 3.09% |
| `readout_projection.weight` | 0.5225 | 262,272 | 1.020e-03 | 0.02207 | 4.62% |
| `rec_syn.comm.weight` | 0.1558 | 16,384 | 1.217e-03 | 0.28203 | 0.43% |
| `width_head.weight` | 0.1664 | 3,870 | 2.674e-03 | 0.08648 | 3.09% |

Every group moved between **0.39% and 4.62% of one initialization standard
deviation**. Adam's per-coordinate step is bounded by approximately the learning
rate, so 96 updates at lr 1e-4 permit at most 0.0096 of movement per coordinate
regardless of gradient magnitude. The binding constraint is the update count,
not the gradient.

This is a complete and sufficient explanation of every measurement in this
document: uniform color output, constant shape prior, loss pinned at the
uniform-random entropy floor, and identical scores across effort levels.

## Step B: the pipeline learns

`docs/diagnostics/example21_overfit_arm.py`. Single-task overfit of `007bbfb7`
through the production training path: same `learner.etrace_grad`, same Adam,
same `clip_grad_norm`, same terminal-mask supervision. Only the update count,
learning rate, and terminal effort vary. No augmentation, so the target is fixed.

First measurement, at effort 0 and lr 1e-3:

```
20 updates: loss 9.3730 -> 0.0227
```

The pipeline reaches near-zero loss on a single task in **20 updates**, against
the 96 updates the full run spent spread across 82 tasks. Learnability is not
in question. **H1 is confirmed**: the zero ARC score is a training-budget and
learning-rate outcome, not a capacity, wiring, or decode outcome.

### Two-by-two arm matrix, 150 updates each

| learning rate | effort | loss first | loss final | loss min |
|---|---|---|---|---|
| 1e-4 | 0 | 9.3730 | **0.0236** | 0.0189 |
| 1e-4 | 32 | 9.2928 | **5.9445** | 5.5924 |
| 1e-3 | 0 | 9.3730 | **0.0000** | 0.0000 |
| 1e-3 | 32 | 9.2928 | **0.0000** | 0.0000 |

Per-group L2 delta over the same 150 updates:

| group | 1e-4 / e0 | 1e-4 / e32 | 1e-3 / e0 | 1e-3 / e32 | full run (96 upd) |
|---|---|---|---|---|---|
| `color_factor_head.weight` | 2.6371 | 2.0559 | 3.3654 | 5.5997 | 0.4411 |
| `ff_syn.comm.weight` | 1.6594 | 1.7468 | 2.7304 | 4.7459 | 0.7045 |
| `height_head.weight` | 0.6498 | 0.5840 | 1.0445 | 1.8906 | 0.1678 |
| `readout_projection.weight` | 1.8209 | 1.3198 | 3.4635 | 3.8602 | 0.5225 |
| `rec_syn.comm.weight` | 0.4409 | 0.6964 | 0.7547 | 1.9337 | 0.1558 |
| `width_head.weight` | 0.7015 | 0.5868 | 1.1346 | 1.9228 | 0.1664 |

## Second defect: effort suppresses learning at the configured rate

At lr 1e-4, the configured rate of the full run, effort 32 reaches loss 5.94
where effort 0 reaches 0.024 in the same 150 updates. That is roughly a 250x
worse objective from the effort setting alone. At lr 1e-3 both arms reach
0.0000, so the effect is a rate problem rather than an impossibility, but at the
rate actually used it is severe.

This confirms the gradient-decay observation in Step A as a behavioral effect
rather than a curiosity: **longer latent rollouts make the model learn more
slowly, not better.** It compounds the budget problem, because the full run
drew two thirds of its 96 updates at efforts 16 and 32 (32 updates at each of
8, 16, and 32). The effective training budget was therefore materially worse
than the raw update count suggests.

Read the two columns differently. The **effort-32-at-1e-4 result, 5.94 against
0.024, stands on its own** as a defect. The **tie at lr 1e-3 does not**: single
-task memorization has no reason to need recurrence, so a tie there is the
expected outcome and is *not* evidence that the recurrence is useless. This
experiment provides no positive evidence for the latent-reasoning thesis, and
no falsification of it either. Do not cite the tie as either.

## Conclusions

1. The zero ARC score is fully explained by training budget. The parameters
   moved less than 5% of one initialization standard deviation.
2. Insufficient capacity is **rejected as an explanation of this run**. It
   remains an open and untested question.
3. Exact ARC pass@1 must not be used as the optimization signal at this stage.
   Every trivial baseline also scores 0.0. Track the shape and pixel
   diagnostics against the floor table in measurement 1.
4. The effort schedule interacts badly with the learning rate. Any rerun must
   either raise the rate or fix the gradient decay, otherwise the high-effort
   updates contribute little.
5. The gradient scale imbalance across parameter groups, roughly 2,800x with
   `color_factor_head` starved, should be addressed before a long run.

## Confirmation: the fix works

The predicted fix was applied as a pure configuration change, lr 1e-3 with a
larger budget, no code modified. Full ARC-AGI-1 evaluation, 400 tasks / 419
queries, on GPU.

| run | updates | lr | shape diagnostic | pixel diagnostic |
|---|---|---|---|---|
| original | 96 | 1e-4 | 0.0859 | 0.0628 |
| rerun | 2048 | 1e-3 | **0.3198** | **0.4035** |
| uniform-random floor | | | 0.6611 | 0.1030 |
| `copy_or_rule_shape` floor | | | 0.8687 | 0.6336 |

Shape improved 3.7x and pixel accuracy 6.4x from a configuration change alone.
Pixel accuracy is now well clear of the uniform-random floor of 0.1030, where
the original run sat below it. This is direct confirmation that the diagnosis
was correct: the original run was budget-starved, not capacity-starved.

Exact pass@1 and pass@2 remain 0.0000, which is expected and uninformative:
every trivial predictor in measurement 1 also scores 0.0000.

The rerun is a fully qualified run, not a development run: `structural=True`
and `scientific=True`, with all eight structural and all twelve scientific
checks passing, despite the recovered out-of-memory warnings during training.
Its splits are identical to v5's (400 evaluation tasks, 419 queries, 0
rejected, 0 duplicates, 1 explicit exclusion), which independently confirms
that the ARC-AGI-1 head used here, `3990304`, carries the same task grids as
the manifest-pinned `aa922be`.

Artifacts: `var/example21-rerun-lr1e-3-2048/` (`report.txt`, `result.json`,
`data_manifest.json`, `latent_reasoning.png`). As with the v5 artifacts, `var/`
is gitignored, so these are local to the machine that produced them.

### Effort ordering reversed

At the original lr 1e-4 the pixel diagnostic *fell* with effort
(0.0650 / 0.0665 / 0.0660 / 0.0628 at efforts 0 / 8 / 16 / 32). At lr 1e-3 it
*rises* monotonically:

```
effort  0: shape 0.3198, pixel 0.3840
effort  8: shape 0.3198, pixel 0.3885
effort 16: shape 0.3222, pixel 0.4021
effort 32: shape 0.3007, pixel 0.4035
```

This is the first positive signal that latent depth contributes anything. The
margin is small, roughly 0.02 in pixel accuracy across 419 queries, and shape
accuracy does not share the trend, so it must not be over-read. It does show
that the effort-suppression defect recorded above is rate-dependent rather than
structural.

### Budget ceiling is memory, not compute

`_prepare_training` pre-materializes the full event tensor of shape
`(updates, sequence, 1, 830)` before training. At 2048 updates the run emitted
recovered `CUDA_ERROR_OUT_OF_MEMORY` warnings on 4 GiB allocations. Training
compute is cheap: 2048 updates plus the full 400-task evaluation completed in
415 seconds, on a GPU power-capped at 210 MHz against a 2100 MHz maximum.

At 4096 updates the run fails outright:

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED:
Out of memory while trying to allocate 4.58GiB
```

**2048 updates is therefore the hard ceiling** for this design on a 12 GB
device. Raising the budget further requires streaming or chunking the training
tensor rather than materializing it up front. That is a code change to
`_prepare_training` and `_train_model`, it touches a run path covered by the
example's qualification gates, and it should be scoped and approved separately
rather than folded into this diagnosis.

### Recommended next run

Done for the budget that fits: lr 1e-3 at 2048 updates, reported above. The
remaining work, in order:

1. **Stream the training tensor** so the update budget is not bounded by device
   memory. This is the blocking item; 2048 updates is not enough to clear the
   trivial floors.
2. **Fix the gradient scale imbalance** (roughly 2,800x, with
   `color_factor_head` starved while producing about 900 of the roughly 902
   values per grid). Per-group normalization or loss reweighting.
3. **Rerun and compare against the floor table**, not exact pass@1. The bar is
   `copy_or_rule_shape`: shape 0.8687, pixel 0.6336.
4. Only after the model clears that bar does the original capacity question,
   whether 2,048 neurons and 16,384 edges suffice, become meaningful and
   testable.

Constraint: do not re-run the 400-task evaluation and do not raise the training
budget until this discriminator passes.

## Container change

`numba==0.67.0` was added to `.github/containers/braintrace-gpu/Dockerfile`.
Without it the CPU sparse kernels cannot build (`csrmm` has no usable backend on
`cpu`), which is what forced the earlier probes onto `sparse_backend="jax_raw"`.
cp314 wheels exist for both `numba` 0.67.0 and `llvmlite` 0.49.0, so there is no
Python-version conflict.

The image is shared with Example 20, so the change was verified on GPU rather
than by import alone: `jax` 0.11.0 resolves `CudaDevice(id=0)` with default
backend `gpu`, and the effort-32 / lr-1e-3 arm rerun on GPU with the numba
sparse backend reproduces the CPU result exactly, 9.2928 -> 0.0000. Note that
the host GPU was power-capped at 210 MHz against a 2100 MHz maximum during this
check; that affects speed, not correctness.

## Status of the diagnostic scripts

The three scripts under `docs/diagnostics/` are one-off investigation probes,
not part of the package. They sit deliberately outside the gated surface: CI
runs `pytest braintrace/` and `pytest examples/`, and this repository declares
no `[tool.ruff]` configuration, so nothing under `docs/` is linted or tested.
They therefore carry no co-located `*_test.py` siblings, which would otherwise
be required by working agreement 9. They were checked with `ruff check` and
`ruff format` manually and pass both.

If any of these measurements becomes a standing regression check rather than a
one-off, it must move under `examples/pp_prop/` and acquire a `*_test.py`
sibling at that point.

## Reproduction

`docs/diagnostics/example21_dummy_floor.py`, run against an ARC-AGI-1 checkout.

Commit note: the source manifest pins ARC-AGI-1 at `aa922be`. The floor table
in measurement 1 was produced against `3990304`, the repository head at the time
of measurement. The task grids are believed identical between the two; this is
recorded so the discrepancy does not have to be re-derived.

Measurements 2 through 4 read `var/example21-full-scientific-final-v5/result.json`
directly.
