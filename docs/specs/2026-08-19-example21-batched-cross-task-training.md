# Example 21 — batched cross-task training for row refinement

Status: proposed
Date: 2026-08-19
Branch: `feat/example21-row-refinement`
Supersedes nothing. Implements delivery-sequence steps 5, 7, and 8 of
`2026-08-18-example21-row-wise-binding-refinement.md` without changing its
score protocol.

## 1. Evidence

The retained row-refinement runs report `shape_accuracy_diagnostic = 0.0000` at
every effort, which pins exact `pass@2` at zero regardless of the color path.
Three measurements taken on this branch identify the cause as the training
regime, not the architecture:

1. **The architecture learns.** The synthetic row-binding qualification
   (`latent_workspace_binding_qualification`) at 400 updates, 256 neurons, and
   4,096 recurrent edges reaches all four families exact — identity, recolor,
   non-square transpose, and a demonstration-dependent row reverse — with loss
   falling 0.4004 → 0.0347. Shape-changing and demonstration-conditioned
   transforms are both inside the learned path's capacity.
2. **The retained ARC runs were untrained.** They applied 3 optimizer updates
   at `learning_rate = 1e-4` with one episode per update.
3. **Batched episodes plus a larger step learn held-out shape.** A held-out
   probe over 32 training-split tasks never used for fitting moved from
   shape accuracy 0.0000 / pixel accuracy 0.1019 to:

   | episodes | updates | batch | shape accuracy | pixel accuracy |
   |---:|---:|---:|---:|---:|
   | 160 | 20 | 8 | 0.1875 | 0.5902 |
   | 9,600 | 600 | 16 | 0.4375 | 0.6716 |

   at `learning_rate = 3e-3`, with training loss still descending
   (0.902 → 0.718 over the run). Sweep 2 also overtakes sweep 1 on pixel
   accuracy once trained (0.6095 → 0.6716), reversing the degradation the
   untrained runs showed and confirming that refinement helps a model that
   has learned anything at all.

Measured cost on the local RTX 3080 Ti, batch 1, through the production
pipeline: 0.075 s per update at 512 neurons / 65,536 edges and 0.21 s per
update at 4,096 neurons / 1,048,576 edges, at 0.61 GiB peak device memory.
Wall clock is not the constraint; episodes seen per unit time is.

## 2. Iteration protocol

Every hyperparameter, architecture, and schedule decision is made against
held-out **training-split** tasks. The 399 admitted ARC-AGI-1 training tasks
split into a fit pool and a probe pool; probe tasks contribute no gradient. The
evaluation split is read only by the frozen scorer after training, as
`2026-08-18-example21-row-wise-binding-refinement.md` §2 requires. Using
`--evaluation-task-limit` output as an iteration signal is a protocol
violation, not a shortcut.

## 3. Change

### 3.1 Batched training episodes

`ExperimentConfig` gains `training_batch_size` (default 1, preserving existing
byte-identical behaviour). The training model is constructed at that batch
size, and each optimizer update consumes that many independent
leave-one-demonstration-out folds.

Episodes in one batch have different demonstration counts, so their latent
windows start at different physical ticks. Per-example masking therefore moves
inside the step function: a tick contributes loss for an example only when that
example's advance gate is high and the event is not a valid row event. The
outer `etrace_grad` mask weights ticks where at least one example is latent.
This is the layout the synthetic qualification already validates at batch 4.

### 3.2 Episode bank

Encoding dominates wall clock once training is batched — 217 s of episode
construction against 73 s of training in the 600-update measurement.
`ExperimentConfig` gains `training_bank_size`: that many augmented folds are
encoded once and updates draw from the bank with replacement. Zero keeps the
current behaviour of encoding one fresh fold per update slot. Bank provenance
(size, distinct base tasks, augmentation counts) enters the report.

### 3.3 Learning rate

The CLI default for `row_refinement` becomes 3e-3. The value is recorded in
the report and remains a flag; no default changes for `legacy_cp`.

## 4. Non-goals

Score protocol, candidate provenance, the model-only gate, the edges-per-neuron
invariant, and the 85% VRAM fail-closed rule are unchanged. Candidate 2
construction is unchanged: the existing earlier-sweep argmax with the
logit runner-up fallback already supplies a distinct learned hypothesis.

## 5. Tests

Co-located in `21-latent-reasoning-in-context_test.py` and the affected
modules:

- `training_batch_size = 1` reproduces the current schedule and losses exactly;
- a batch of folds with differing demonstration counts places each example's
  supervision on exactly its own latent ticks and nowhere else;
- an example contributes no loss on a tick outside its latent window;
- bank draws are reproducible from the seed and independent of chunking;
- a bank of size zero reproduces the per-update encoding path;
- configuration rejects a non-positive `training_batch_size` and a bank size
  smaller than the batch;
- the held-out probe reports shape, pixel, and exact metrics from decoded model
  candidates only;
- no evaluation-split target reaches the bank, the batch, or the scheduler.

## 6. Gates

1. Focused co-located tests green.
2. Held-out training-split probe shape accuracy stays above the 0.4375 already
   measured, at equal or lower episode cost.
3. A full-scale run records realized neurons, edges, edges per neuron, peak
   device memory below 85%, and parameter movement.
4. Model-only ARC-AGI-1 exact `pass@2` reported honestly, zero included.
