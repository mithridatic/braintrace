# Example 21 — resumable pretraining checkpoints

Status: implemented; tests pass
Date: 2026-08-19
Branch: `feat/example21-row-refinement`
Follows `2026-08-19-example21-batched-training-results.md` §6 item 1.

## 1. Why

The measured evidence names pretraining episodes as the only surviving lever on
exact ARC score: the loss curve was still descending at 96,000 episodes, and the
four alternative levers were each eliminated by measurement. Extending that
curve by an order of magnitude is a multi-hour block of GPU time, and the
current checkpoint contract cannot express it.

`parameter_checkpoint` is a write-through cache with exactly two behaviours: an
absent path trains and writes, a present path restores and *skips* training. A
run therefore cannot start from previously trained parameters and keep
optimizing, so a long curve is one indivisible process. Two facts make that
unacceptable:

- this GPU runs under WDDM and has already killed one 64-minute run with
  `CUDA_ERROR_UNKNOWN` at a host synchronisation, losing the whole stage;
- every intermediate point of an episode-scaling curve is itself a measurement,
  and re-running from zero to reach each point costs the sum of the prefix.

## 2. Change

### 2.1 Restore-then-continue

`ExperimentConfig` gains `initial_checkpoint: pathlib.Path | None = None`.

When set and present, its parameter leaves are restored into the model before
training and training then proceeds normally. It reads the same file format
`parameter_checkpoint` writes and fails closed on the same tree-structure and
shape mismatches, so a checkpoint from a different neuron count, edge count, or
decoder mode is rejected rather than reshaped.

`initial_checkpoint` and `parameter_checkpoint` are independent. Setting both to
distinct paths chains a segment: restore the previous segment's parameters,
train `training_updates` more, write the result. Setting `parameter_checkpoint`
to a path that already exists keeps its current restore-and-skip meaning and
wins over `initial_checkpoint`, because the cheaper answer is already on disk.

A restored-then-continued run reports `performed = True` — an optimizer update
did happen — together with the initial checkpoint's path and digest, so the
artifact records the parameters it started from.

The Adam moments are not carried across a segment boundary. This is a
deliberate simplification: bias correction makes the first step after a reset
approximately one learning rate in magnitude, and segments are thousands of
updates long. The report records the boundary so the transient is attributable.

### 2.2 Periodic checkpoint writes

`ExperimentConfig` gains `checkpoint_every: int = 0`, counted in training
chunks. A positive value writes `parameter_checkpoint` after every that-many
chunks of a training run, so a device fault costs at most one interval rather
than the whole segment. Zero preserves current behaviour exactly: one write
after training.

The write requires `parameter_checkpoint` to be configured, and is rejected
otherwise rather than silently ignored. Writes are made from the model that
training optimizes; the parameter tree does not depend on batch size, so the
file is interchangeable with one written after `_copy_parameters`.

Checkpoints are written to a temporary sibling and moved into place, so a fault
during a write cannot leave a truncated file that a later run would restore.

### 2.3 A reserved training holdout

`ExperimentConfig` gains `training_holdout_tasks: int = 0`. A positive value
withholds that many tasks from the tail of the training split, so they never
enter the optimization pool.

An episode-scaling curve needs a measurement surface at every intermediate
point. Without a reserved holdout the only uncontaminated surface is the ARC
evaluation split, and spending it on curve decisions turns the reported number
into a selected one. Reserving the tail of the *training* split keeps every
curve decision off the evaluation data entirely. `var/qual/adapt_probe.py`
already scores `data.training[-N:]`, so a matched `training_holdout_tasks = N`
makes that probe honest by construction.

Zero preserves current behaviour, and the reported run trains on the whole
split. The value is recorded in the training report.

## 3. Protocol

Unchanged. Checkpoints carry parameters only. No evaluation target enters the
training path, and restoring parameters cannot introduce one.

## 4. Tests

- an `initial_checkpoint` restores every leaf and reports the restored digest;
- a run seeded from an `initial_checkpoint` reports `performed = True`, starts
  from exactly the restored parameter digest, and moves the parameters;
- an absent `initial_checkpoint` leaves the fresh initialization untouched;
- an `initial_checkpoint` from a different neuron or edge count is rejected;
- `checkpoint_every` writes a restorable file at the expected chunk boundaries
  and every chunk reaches the callback;
- `checkpoint_every` without `parameter_checkpoint` is rejected;
- `checkpoint_every = 0` builds no writer, so training writes once as now;
- a failed write leaves the previous checkpoint byte-identical;
- a reserved training holdout trims the tail of the pool, and the withheld task
  appears in no sampled training schedule;
- a holdout that consumes the whole split is rejected.

Exact parameter equality between one `n + m` run and a chained `n` then `m` run
is **not** asserted, and the specification does not claim it. The two are not
the same experiment: the effort and task draws are functions of
`training_updates`, so the chained pair trains on a different episode schedule,
and the Adam moments reset at the boundary. What the tests establish is that a
segment starts from exactly the parameters it restored and continues to
optimize them.

## 5. Gate

This is infrastructure, not a score change. It is accepted when the tests above
pass and a chained two-segment run continues to descend rather than restarting
the loss curve. No ARC-score claim is made by this specification.
