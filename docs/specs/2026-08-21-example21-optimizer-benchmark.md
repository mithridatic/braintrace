# Example 21 optimizer benchmark

## Objective

Compare the existing Adam training path with AdamW and Muon on the same
Example 21 ARC workload. The benchmark is a controlled optimizer comparison,
not a hyperparameter search: every arm uses the same model seed, source data,
training schedule, gradient clipping, learning rate, topology, and evaluation.

## User-visible interface

`ExperimentConfig` and the Example 21 CLI gain:

- `optimizer`: one of `adam`, `adamw`, or `muon`; default `adam` preserves the
  existing behavior.
- `weight_decay`: a finite nonnegative scalar. Its default is `0.0` for Adam
  and `0.01` for AdamW and Muon when the caller does not specify a value.

The command-line forms are `--optimizer` and `--weight-decay`. The resolved
optimizer and weight decay are included in `configuration` and the training
report so artifacts identify the exact update rule.

## Optimizer construction

- Adam remains `braintools.optim.Adam(lr=learning_rate)` and ignores no new
  hidden defaults.
- AdamW uses `braintools.optim.AdamW` with the resolved learning rate and
  decoupled weight decay.
- Muon uses `optax.contrib.muon` inside
  `braintools.optim.OptaxOptimizer`. Rank-two parameter leaves take Muon
  updates; other leaves take Muon's built-in AdamW fallback. Both branches use
  the resolved learning rate and weight decay.

The optimizer is registered against the same `learner.param_states` tree and
is updated from the same clipped BrainTrace gradients inside the existing
compiled training driver. No Python step loop is introduced. Optional
task-local adaptation remains Adam because it is a separate diagnostic and is
disabled in the benchmark.

Optax 0.2.8 is a direct project dependency and is pinned in the GPU image. It
is the first released version used here with the current Muon presets and the
JAX 0.11 runtime.

## Validation

Colocated Example 21 tests must cover:

- unchanged Adam defaults and CLI round-tripping;
- optimizer-name and weight-decay validation;
- conditional default decay resolution;
- construction and parameter registration for all three optimizers;
- Muon matrix-versus-nonmatrix partition behavior;
- artifact reporting of the resolved optimizer policy;
- a compiled update that moves parameters and produces finite state for each
  optimizer.

The focused Example 21 regression gate must pass before GPU benchmarking.

## Benchmark protocol

Run three sequential GPU arms from the same worktree revision and source
manifest:

| Setting | Value |
| --- | --- |
| seed | `2108` |
| neurons | `4096` |
| recurrent edges | `8192` |
| training batch size | `32` |
| training updates | `260` |
| training chunk size | `5` |
| latent steps | `60` |
| learning rate | `1e-3` |
| sparse backend | `jax_raw` |
| Adam weight decay | `0.0` |
| AdamW weight decay | `0.01` |
| Muon weight decay | `0.01` |

The source manifest is the existing complete ARC-AGI-1 v1.0.2 manifest.
Runs are sequential to avoid GPU contention and share the persistent JAX
compilation cache. The fixed learning rate intentionally favors control over
per-optimizer tuning; the result must not be described as each optimizer's
best achievable performance.

## Result comparison

For each arm retain the raw Example 21 output directory and report:

- initial, final, minimum, and mean training loss plus finite-loss count;
- per-parameter movement and total movement;
- primary pass@1, pass@2, strict-task, exact-query, shape, and pixel metrics;
- total runtime and phase timing when available;
- peak GPU memory and runtime-safety status;
- source revision, dirty state, image identity, and package versions.

The winner, if any, is descriptive for this single controlled seed and budget.
Equal zero ARC scores must be reported as a tie on ARC rather than broken using
training loss alone.
