# Example 21 — chunked training stream

Status: implemented in `c303207`; verified at 4096 updates
Date: 2026-08-17
Branch: `feat/example21-latent-reasoning`
Follows: `docs/specs/2026-08-16-example21-zero-score-root-cause.md`

## Problem

`_prepare_training` materialises every update of the run as one dense array
before a single optimizer step is taken:

```
events   (training_updates, sequence, 1, input_width)  float32
advances (training_updates, sequence, 1)               bool
colors   (training_updates, 1, 30, 30)                 int32
```

`events` dominates. At the measured packed sequence length the per-update cost
is ~1.2 MB, so `training_updates = 4096` asks for a 4.58 GiB single allocation
and dies with `RESOURCE_EXHAUSTED` on a 12 GB card. `2048` is the hard ceiling,
and the memory bound — not the science — is what sets it.

## Why more updates is the right thing to buy

Windowed mean loss over the confirmed `lr 1e-3 @ 2048` run
(`var/example21-rerun-lr1e-3-2048/result.json`):

| updates | Δ mean loss over the doubling |
| ------- | ----------------------------- |
| 256→512 | −1.41 |
| 512→1024 | −0.74 |
| 1024→2048 | −0.63 |

The curve is not flat. After the initial transient it settles to a steady
≈ −0.65 per doubling and is still descending at the ceiling, ending at a mean
of 4.95 — far above the single-task overfit arms, which reach 2e-6 in 150
updates at both effort 0 and effort 32. Capacity and the optimizer are not the
constraint; per-task exposure is. 2048 updates over 400 augmented training
tasks is roughly five exposures each.

## Approach

Stream the run in fixed-size chunks. Nothing about the mathematics, the update
order, or the random stream changes — only how much of the schedule is resident
at once.

### 1. Configuration

Add `ExperimentConfig.training_chunk_size: int = 0`.

- `0` means "one chunk" and reproduces today's behaviour exactly, so every
  existing config, artifact, and the `lr 1e-3 @ 2048` control arm stay
  bit-identical.
- A non-zero value must satisfy `training_updates % training_chunk_size == 0`;
  otherwise raise in `__post_init__`. A remainder chunk would be a second scan
  length and a second XLA compile for no benefit. The divisibility check is
  gated on `not structural_only and training_updates > 0` so the
  `structural_only, training_updates=0` config still constructs.
- Validated through the existing `_integer` tuple in `__post_init__` with
  `minimum=0`.

### 2. Preparation

Split `_prepare_training` into:

- `_training_row(...)` — builds one update's rows. Body lifted verbatim from
  the current per-update loop.
- `_training_chunks(data, config, row_config)` — a generator yielding
  `_TrainingTensors` of `training_chunk_size` updates each.
- `_prepare_training(...)` — retained, now the concatenation of every chunk.
  Signature and return value unchanged, so the existing tests that call it keep
  their meaning. It is no longer on the run path; keeping it on the run path
  would rebuild all 4096 updates on the host and save nothing.

`run_experiment` passes `_training_chunks(...)` — the generator itself — to
`_train_model`, whose second parameter becomes an *iterable of*
`_TrainingTensors` rather than one materialised tensor.

**Random-stream identity is the correctness risk, not memory.** `efforts` and
`task_indices` are drawn up front at the full `training_updates` size and the
per-update walk consumes the same `rng` sequentially. Both properties are
preserved: the up-front draws stay up front at full size, and the walk visits
updates in the same order across chunk boundaries.

### 3. Training

`_train_model` traces `train_all` once at the chunk length and applies it to
each chunk in turn. `State` — model, learner, and Adam `m`/`v`/step count —
persists across the call boundary, so chunking is invisible to the optimizer.
Verified empirically before writing any of this: eight Adam updates split 1/2/4
ways across separate `for_loop` calls produce bitwise-identical parameter
trajectories, so the bias-correction count is carried as state and not captured
as a trace-time constant.

The host-side chunk loop lives in a separate **module-level** `_train_chunks`
helper, not in `_train_model` and not nested inside it — `ast.walk` descends
into nested function definitions and would find the loop anyway. This is not
evasion of AGENTS.md rule 10: the loop that drives the model stays a single
compiled `brainstate.transform.for_loop` over the whole chunk. `_train_chunks`
iterates a handful of data-staging steps. The AST guard test is *extended*
rather than relaxed — see below.

`_train_chunks` returns both the concatenated losses and the combined
per-update metadata (`task_fingerprints`, `base_task_fingerprints`,
`source_names`, `query_indices`, `efforts`), so that the accumulation loop lives
with the chunk loop and `_train_model` stays loop-free. That metadata feeds
`sample_records` and the `losses_complete` / `mixed` structural gates, which
assert full `training_updates` length; truncating to the last chunk would
silently pass those gates with wrong data.

The realised effort schedule is recorded in the training result dict. The
`lr 1e-3 @ 2048` curve could not be decomposed by effort because `result.json`
never carried it; one line fixes that for every run hereafter.

Two guards inside the loop:

- chunk *k*'s sequence length must equal chunk 0's, converting a silent
  recompile storm into a loud error;
- the accumulated update count must equal `training_updates`.

## Tests

Co-located in `examples/pp_prop/21-latent-reasoning-in-context_test.py`.

1. **Chunking is a no-op on the numbers.** Run the smoke config through
   `_train_model` with one chunk and with several, assert the returned `losses`
   are bitwise identical. This is a deterministic scan, so bitwise equality must
   hold. The single assertion simultaneously proves the random stream is intact,
   Adam's moments and step counter survive the jit-call boundary (a step counter
   captured as a Python int at trace time would freeze bias correction from
   chunk 2 on and this catches it), and the per-update `reset_state` is
   unaffected.
2. **Metadata concatenates in order** — chunked and unchunked
   `_prepare_training` agree on every array and every metadata tuple.
3. **Config validation** — a chunk size that does not divide `training_updates`
   raises; `0` is accepted and means one chunk.
4. **AST guard, extended** — `_train_model` stays loop-free; the new
   `_train_chunks` may loop but must contain no `etrace_grad` call, and the
   compiled path must still use `brainstate.transform.for_loop`.

## Explicitly out of scope

- The ~2,800× gradient scale imbalance starving `color_factor_head`. Separate
  change, separate spec; one fix at a time.
- Storing `events` as `int8` and casting inside `step_loss`. If the event rows
  really are 0/1 indicators this is a 4× ceiling multiplier for two lines and it
  stacks with chunking rather than competing with it. Worth pricing next, but
  bundling it would make the bitwise-identity test above meaningless.

## Verification

`--training-updates 4096 --training-chunk-size 512 --learning-rate 1e-3`,
full 400-task evaluation, GPU, 686 s. The same budget previously died with
`RESOURCE_EXHAUSTED` on a 4.58 GiB allocation; chunked, peak device memory
stayed under 3 GiB. All 8 structural and all 12 scientific gates pass.

Loss continues to descend past the old ceiling, at the rate the 2048 run
predicted:

| updates | mean loss |
| ------- | --------- |
| 0–256 | 7.887 |
| 256–512 | 6.484 |
| 512–1024 | 5.724 |
| 1024–2048 | 5.032 |
| 2048–4096 | 4.341 |

The 2048→4096 doubling bought −0.69, in line with the −0.65 trend. The
mechanism works and the ceiling is gone.

### What the extra updates actually bought

| metric (intact, best effort) | 2048 | 4096 |
| ---------------------------- | ------ | ------ |
| shape | 0.3222 | 0.4105 |
| pixel | 0.4035 | 0.3915 |

Shape improved; pixel did not. Halving the loss again is not currently
converting into pixel accuracy, so a further doubling of the budget is not
obviously the next thing to buy.

### The finding that supersedes the remaining backlog

The `shuffled_demonstrations` control scores within ±0.01 of the intact arm at
every effort (shape 0.3723–0.4010, pixel 0.3918–0.3969). Destroying the
demonstration input/output pairing costs the model nothing. The
`no_context` control does collapse (shape 0.1026–0.1456, pixel
0.1556–0.1707), so the model is using the query grid — it is learning output
priors conditioned on the query, not the demonstrated transformation.

This reframes the backlog. Chasing the ~2,800× gradient imbalance, or more
updates, optimises a model that is not doing in-context reasoning at all. The
next question is why the demonstration channel carries no usable signal, and
`shuffled_demonstrations` is already the discriminator for it.

Also settled: the effort ordering at 2048 (pixel rising 0.3840 → 0.4035 across
efforts 0→32) was noise. At 4096 pixel is flat across effort
(0.3911, 0.3835, 0.3899, 0.3915). The earlier note not to lean on that signal
was correct.
