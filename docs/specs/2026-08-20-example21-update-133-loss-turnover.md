# Example 21: the "update 133" training-loss turnover

Date: 2026-08-20
Branch: `investigate/ex21-update-133-turnover`
Status: investigation complete; **no code change recommended**

## Observation under investigation

Three Example 21 runs (`--training-updates 260 --training-batch-size 32
--training-chunk-size 5 --latent-steps 60 --max-demonstrations 10`) each placed
`min(training.losses)` at **exactly index 133**, across a 3x learning-rate
change and a 2x recurrent-edge change, with the loss apparently rising
afterwards.

| run | first | min | argmin | last | last-20 avg |
|---|---|---|---|---|---|
| 2048n/2048e/lr1e-3 | 3.160 | 1.635 | 133 | 2.014 | 2.366 |
| 2048n/2048e/lr3e-3 | 3.160 | 2.049 | 133 | 2.589 | 3.063 |
| 2048n/4096e/lr1e-3 | 3.203 | 1.617 | 133 | 2.024 | 2.348 |

## Root cause

`training.losses[i]` is the objective on the **`i`-th freshly drawn random
batch**, and the batch draw is a pure function of `--seed`. It is not a
running estimate of model quality on a fixed quantity, so it is not comparable
across updates. Index 133 is simply the batch that reads easiest under seed
2108 at this effort set. There is no schedule, curriculum, ordering boundary,
or optimizer event at update 133.

`_training_chunks` (`examples/pp_prop/21-latent-reasoning-in-context.py:1640-1647`)
draws the entire training stream up front from one generator seeded only by the
experiment seed:

```python
rng = brainstate.random.RandomState(config.seed + 1000)
efforts = _effort_schedule(config.training_updates, rng, config.training_efforts)
task_indices = np.asarray(
    rng.randint(0, len(pool), size=config.training_updates * batch),
    dtype=np.int32,
)
```

Nothing in that stream depends on learning rate, neuron count, or recurrent
edge count. Consequently every run at a given `(seed, training_updates,
batch_size, latent_steps)` sees the **identical** 8320-sample batch sequence in
the identical order at the identical effort. Batch-to-batch difficulty variance
(which ARC tasks were drawn, which demonstration was held out, how large the
target grids are) then dominates the per-update loss, and whichever update
happens to draw the easiest batch wins the argmin in every such run.

## Evidence chain

### 1. The batch sequence is byte-identical across the three runs

All three record `seed = 2108` and an identical 8320-element
`training.training_task_fingerprints` list and identical `effort_schedule`:

```
fingerprint seqs identical A==B: True   A==C: True
```

### 2. Per-update loss variance is model-independent

Pearson correlation of the 260-point loss sequences:

| pair | difference | r |
|---|---|---|
| 2048e vs 4096e (same lr) | 2x recurrent edges | **0.9986** |
| lr1e-3 vs lr3e-3 (same net) | 3x learning rate | **0.8691** |

After removing a 21-point moving-average trend, the *residuals* still
correlate at **0.9983** (edges) and **0.8425** (learning rate). A 2x
architecture change leaves 99.7% of the per-update variance untouched. The
signal is the data, not the model.

Residual sigma is **0.346** against a smoothed dynamic range of **0.988** --
roughly 35% of the visible movement in `losses` is batch noise.

### 3. Index 133 is not special even among the noise

`loss[133]` is a **-2.0 sigma** residual. The most negative residual is index
**112** in both lr1e-3 runs. 133 wins the raw argmin only because a strong
negative batch residual lands inside an already-low region of the smoothed
trend. The six most negative residuals in each run:

```
A (lr1e-3): [112, 133, 101, 90, 51, 127]
B (lr3e-3): [  0, 133, 112, 51, 13, 138]
C (4096e):  [112, 133, 101, 90, 51, 127]
```

### 4. Direct causal proof: change only the seed

Re-ran 2048n/2048e/lr1e-3/u260/l60 with `--seed 7777`, everything else
identical (`var/example21-2048n-2048e-b32-u260-l60-lr1e-3-seed7777`):

| quantity | seed 2108 | seed 7777 |
|---|---|---|
| argmin | **133** | **179** |
| min loss | 1.635 | 1.435 |
| `loss[133]` | 1.635 | 2.670 |
| fingerprint sequence | -- | different |
| effort schedule | -- | different |
| correlation with seed-2108 losses | 1.000 | **0.114** |

Changing the seed alone moves the argmin and collapses the cross-run
correlation from 0.9986 to 0.114. Changing the architecture or the learning
rate does neither. That is the causal test.

`--seed 7777` changes the batch order, the effort order, **and** the parameter
initialisation together, so on its own it does not isolate which of the three
matters. The A-vs-C pair closes that confound: their
`training.parameter_sha256_before` values differ
(`a7fca2b7c8ecd79f...` vs `687cd9133f7aedf0...`), so those two runs start from
different initial parameters, yet their residual correlation is **0.9983**.
Initialisation varies and the residual structure does not move; the batch and
effort sequence varies and it collapses. The cause is the seed-fixed data
stream.

### 5. The claimed post-133 rise does not exist

20-update block means:

| block | 0-19 | 20-39 | 40-59 | 60-79 | 80-99 | 100-119 | 120-139 | 140-159 | 160-179 | 180-199 | 200-219 | 220-239 | 240-259 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A lr1e-3 | 3.017 | 2.562 | 2.485 | 2.368 | 2.429 | 2.449 | 2.302 | 2.201 | 2.323 | 2.265 | 2.370 | 2.303 | 2.366 |
| B lr3e-3 | 4.118 | 3.598 | 3.476 | 3.288 | 3.288 | 3.395 | 3.104 | 3.181 | 3.098 | 3.135 | 3.109 | 3.012 | 3.063 |
| C 4096e | 3.022 | 2.559 | 2.481 | 2.366 | 2.427 | 2.438 | 2.310 | 2.274 | 2.323 | 2.274 | 2.374 | 2.307 | 2.348 |
| seed 7777 | 2.740 | 2.421 | 2.434 | 2.363 | 2.310 | 2.308 | 2.342 | 2.183 | 2.303 | 2.335 | 2.296 | 2.184 | 2.335 |

Training improves from ~3.02 to a ~2.30 plateau by block 3 and then stays
flat. There is no monotonic rise after 133. The impression of one comes
entirely from comparing a single outlier trough (1.635) against ordinary
subsequent values.

## What was ruled out, and why

**A curriculum / depth-weighting / supervised-depth ramp.** `supervised_depths`
is the constant string `"latent_row_ticks_1..effort"`, `depth_weighting` is the
constant `"uniform_unit_sum_per_update"`, and `per_update_depth_weight_sum` is
the scalar `1.0`. None of them is a per-update array; none can change at 133.

**A phase boundary at the 130/130 effort split.** `optimizer_updates_by_effort
= {"30": 130, "60": 130}` looks like a half-and-half phase change, and 133 is
near 130. It is not a phase: `_effort_schedule` (line 1114) builds
`np.resize(efforts, updates)` and then applies `rng.permutation(updates)`, so
the two efforts are *interleaved*, not blocked. Observed
`effort_schedule[130:140] = [60,60,30,60,30,60,60,60,60,30]`. Additionally,
mean loss by effort is 2.433 +/- 0.034 (effort 30, n=130) vs 2.404 +/- 0.036
(effort 60, n=130); the difference is 0.028 against a standard error of 0.050,
t = 0.57. Effort does not shift the loss scale, which is what the `1.0/effort`
mask is there to guarantee.

**A normalization / denominator that moves with update index.** Both
normalizations are per-update constants. The per-tick reduction divides by the
supervised count (`21-latent-reasoning-in-context.py:2409`), and the depth mask
is `1.0/effort` (line 1277), giving a unit-sum weighting over exactly the
`effort` supervised row ticks. Because `TRAINING_EFFORTS = (30, 60)` and both
are multiples of `MAX_GRID_SIZE = 30`, every supervised window is a whole
number of answer-row sweeps, so `row_refinement_loss_per_example`'s
`30 / (target_height + 1)` color rescale
(`examples/pp_prop/latent_workspace_refinement.py:705-707`) does what its
docstring claims and the single shape term fires once per sweep. The loss is
correctly normalized and *is* comparable across efforts. It is only
incomparable across **batches**, which is the actual finding.

**A data-ordering boundary (shard, epoch wrap, holdout split, chunk).**
`sampling_with_replacement = true`, drawn by `rng.randint` over 1296 folds from
399 base tasks; `training_holdout_tasks = 0`; `checkpoint_every_chunks = 0`.
There is no epoch structure to wrap. `training_chunk_size = 5` divides 260 into
52 chunks and 133 = 5*26 + 3 is not a chunk boundary.

**A fixed fraction of the run (a schedule).** Argmins across every archived
Example 21 run share no fraction:

| run | updates | argmin | fraction |
|---|---|---|---|
| u13-l390 (1024n, 2048n, both timed variants) | 13 | 3 | 0.231 |
| u130-l390 (1024e **and** 1047552e) | 130 | 75 | 0.577 |
| u260-l150 (1024n, 2048n) | 260 | 251 | 0.965 |
| u260-l150 (b64) | 260 | 258 | 0.992 |
| u260-l60 (all three) | 260 | 133 | 0.512 |
| u390-l150 | 390 | 333 | 0.854 |
| u1040-l150 | 1040 | 965 | 0.928 |

No common fraction and no common absolute index -- but note that *within* each
configuration group the argmin is invariant to architecture, including a
**1024e vs 1047552e** comparison (a ~1000x change in recurrent edges) that
still yields argmin 75. That invariance is the signature of a seed-fixed data
stream, not of optimization.

**An optimizer basin.** Excluded by the same invariance: a 3x learning-rate
change cannot leave an optimization landmark at the same integer index, and the
seed ablation moves the index without touching the optimizer at all.

## Supporting detail: an identical batch sequence with a different argmin

`example21-shared-2048n-2048e-b32-u260-l150` shares `seed`, `training_updates`,
and `training_batch_size` with the l60 runs and has a **byte-identical**
8320-element `training_task_fingerprints` sequence, yet its argmin is 251, not
133.

This comparison is **confounded three ways** and isolates nothing on its own:
`latent_steps` 150 vs 60 (hence `training_efforts` `(30,60,90,120,150)` vs
`(30,60)`), `learning_rate` 1e-4 vs 1e-3, and `training_chunk_size` 1 vs 5. It
is recorded only as consistency evidence for the general form

```
loss[i] = f(batch_i, effort_i, model_state_i)
```

where `batch_i` and `effort_i` are both fixed by the seed and independent of
every hyperparameter under study. The causal proof is the seed ablation in
section 4, not this comparison.

## Real failure or accounting artifact?

**The update-133 turnover is an accounting artifact.** The number moved; the
model did not get worse at 133. Forced by: (a) argmin invariance under a 3x
learning-rate change, which no genuine optimization event can survive; (b)
residual correlation 0.9983 between two different architectures, meaning the
per-update variance is essentially all data; (c) the seed ablation, which moves
the argmin to 179 and drops the correlation to 0.114 while holding the
optimizer fixed; (d) block means that are flat, not rising, after 133.

### There is no second, smaller "real" rise either

A smoothed (21-point moving average) version of the series appears to bottom
out and then climb:

| run | smoothed argmin | smoothed min | smoothed last |
|---|---|---|---|
| A lr1e-3 | 154 | 2.195 | 2.311 |
| B lr3e-3 | 200 | 2.927 | 3.123 |
| C 4096e | 154 | 2.192 | 2.303 |
| seed 7777 | 232 | 2.158 | 2.448 |

**This is the same selection bias one level up and must not be reported as an
effect.** `smoothed_min` is the *selected* minimum of a still-noisy series, so
`smoothed_last > smoothed_min` is guaranteed by construction in every run
whether or not any trend exists -- exactly the reasoning that produced the
original index-133 table.

Two tests dispose of it.

**Block means against their own standard error.** With residual sigma 0.346, a
20-update block mean has SE = 0.346/sqrt(20) = 0.077, so a difference of two
blocks has SE 0.109:

| run | lowest 20-block from update 60 | last block | gap | gap / SE |
|---|---|---|---|---|
| A lr1e-3 | 2.201 | 2.366 | 0.165 | **1.51** |
| B lr3e-3 | 3.012 | 3.063 | 0.052 | **0.30** |
| C 4096e | 2.199 | 2.348 | 0.149 | **1.36** |
| seed 7777 | 2.183 | 2.335 | 0.152 | **1.45** |

No run reaches 2 sigma, and these are *selected* minima, so even 1.5 sigma
overstates the case.

**Ordinary least squares on the post-warmup segment.** Fitting a line to
updates 60-259 -- no minimum selected, no smoothing -- the slope is
**negative in all four runs**:

| run | slope / update | SE | t | implied drift over 200 updates |
|---|---|---|---|---|
| A lr1e-3 | -0.00041 | 0.00046 | -0.89 | -0.081 |
| B lr3e-3 | -0.00162 | 0.00069 | -2.33 | -0.323 |
| C 4096e | -0.00040 | 0.00046 | -0.88 | -0.081 |
| seed 7777 | -0.00032 | 0.00044 | -0.74 | -0.065 |

The loss is flat-to-still-slightly-falling after update 60. It is not rising at
all. The apparent late climb is entirely an artifact of measuring from a
selected trough.

**Conclusion: artifact, full stop.** Training falls from ~3.02 to a ~2.30
plateau by update 60, and every subsequent movement -- including the index-133
trough and the apparent recovery from it -- is within batch noise. There is no
second effect to chase.

## Recommendation

**No fix.** The loss computation and both normalizations are correct, effort is
correctly neutralised by the `1.0/effort` mask, and there is no real
degradation anywhere in the run. The turnover is a property of reading a noisy
per-batch training loss as if it were a trend, not a defect in Example 21.

### Optional follow-up (proposed, not implemented)

Per-update training loss is ~35% batch noise (residual sigma 0.346 vs smoothed
range 0.988) and is therefore not a readable convergence signal. A fixed probe
batch -- the same N episodes at the same effort, scored every K updates without
an optimizer step -- would give a comparable-across-updates curve and prevent
the same misreading recurring. This is a behavioural change to a scientific
artifact generator and to the entry point's hashed sources, so per AGENTS.md
working agreement item 1 it is proposed here for approval rather than
implemented.

## Reproduction

Analysis over archived result JSONs (no GPU required):

```bash
python - <<'PY'
import json, statistics as st, glob, os
for d in sorted(glob.glob('var/example21-*')):
    p = os.path.join(d, 'result.json')
    if not os.path.exists(p):
        continue
    losses = json.load(open(p))['training'].get('losses')
    if not losses:
        continue
    n = len(losses)
    argmin = min(range(n), key=lambda i: losses[i])
    print('%-52s n=%4d argmin=%4d frac=%.3f' % (os.path.basename(d), n, argmin, argmin / n))
PY
```

Seed ablation (GPU, ~3 minutes, `braintrace-gpu:0.11.0-py314-msgspec-arc`):
the canonical command from `examples/pp_prop/README.md` with
`--seed 7777 --neurons 2048 --recurrent-edges 2048 --latent-steps 60
--training-updates 260 --training-batch-size 32 --training-chunk-size 5
--learning-rate 1e-3`.
