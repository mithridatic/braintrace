# Example 21 — adaptation data and parameter checkpoints

Status: proposed
Date: 2026-08-19
Branch: `feat/example21-row-refinement`
Follows `2026-08-19-example21-batched-cross-task-training.md`.

## 1. Two problems this addresses

### 1.1 Pretraining is not reusable

`run_experiment` hashes the trained parameters into the report but never writes
their values. An hour of pretraining dies with the process, so every question
about the evaluation stage — a different adaptation budget, a different
candidate rule, an ablation — costs a full re-pretrain. This is the single
largest drag on iterations per hour.

### 1.2 Per-task adaptation is starved of data

A task supplies 2 to 9 demonstrations, so leave-one-out yields only a handful of
folds. Measurement showed exact `pass@2` flat at 0.0667 across 40, 120, and 300
adaptation steps and across three learning rates: repeating the same three or
four folds stops helping quickly. The folds themselves, not the step count, are
the limit.

The training path already owns semantics-preserving augmentation — a bijective
color permutation, the eight dihedral transforms, and demonstration-order
sampling — applied consistently to demonstrations, query, and target. Applying
the same transforms to a task's own adaptation folds multiplies the adaptation
set by up to eighty without introducing any hand-written rule and without ever
touching the official query target. The transform is chosen by
`brainstate.random`, is recorded in the report, and cannot inspect the held-out
output.

## 2. Change

### 2.1 Parameter checkpoints

`ExperimentConfig` gains `parameter_checkpoint: pathlib.Path | None`. It behaves
as a write-through cache:

- absent path, or none configured: train as now, and when a path is configured
  write the trained parameter leaves to it after training;
- present path: restore those leaves into the model, skip training, and report
  `performed = False` with `reason = "restored_parameter_checkpoint"`, the
  checkpoint's SHA-256, and the recorded configuration it was trained under.

`training_updates` may be zero when a checkpoint is restored. The restored
digest is compared with the digest recorded at write time and a mismatch fails
closed. A checkpoint written under a different neuron count, edge count, or
decoder mode is rejected on the parameter tree structure rather than silently
reshaped.

### 2.2 Augmented adaptation folds

`ExperimentConfig` gains `adaptation_augmentations: int = 0`. For each task the
compact adaptation bank carries the original folds plus that many augmented
copies, each a semantics-preserving transform of the whole task applied
consistently to every demonstration. Zero preserves current behaviour exactly.

The fold schedule already supports repeats through `adaptation_epochs`;
augmentation adds distinct folds rather than repeats, so the two compose. The
report records, per task, the number of original folds, the number of augmented
folds, and the transform identity of each.

### 2.3 The episode bank does not scale to longer pretraining

One encoded episode occupies `(330 + 60) x 830` float32, or 1.29 MB, so a bank of
4,000 episodes per supervised effort holds 10.4 GB of host memory — measured at
14.76 GiB of a 23.47 GiB container during the first complete run. Raising the
bank to keep episodes distinct across a 300,000-episode run would need roughly
31 GB and cannot fit.

The bank exists only because episode encoding is synchronous with training. The
measured duty cycle is about two thirds GPU-busy and one third host-busy
stacking the next chunk, so overlapping the two recovers that third *and*
removes the need to retain episodes at all. The replacement is a prefetch of the
next chunk on a worker thread while the current chunk trains; the encoders are
NumPy-bound and release the interpreter lock. Until that lands, bank size is
capped by host memory and episodes are reused, which is recorded in the report
rather than hidden.

## 3. Protocol

Unchanged. Augmentation reads demonstrations only. The official query target is
absent from the adaptation bank by construction and remains available only to
the scorer after inference. Candidate provenance stays model-only.

## 4. Tests

- a written checkpoint restores byte-identical parameters and a matching digest;
- a checkpoint from a different neuron or edge count is rejected;
- a restored run performs no optimizer update and says so in the report;
- `adaptation_augmentations = 0` reproduces the current bank byte-identically;
- an augmented fold applies one transform consistently to every demonstration
  input and output of that fold;
- no augmented fold contains an official query target;
- augmentation draws are reproducible from the seed;
- augmented folds and repeated epochs compose to the expected schedule length.

## 5. Gate

Held-out training-split exact `pass@2` under augmented adaptation must exceed
the 0.0667 measured without it, at the same pretrained checkpoint, before the
option is used in a reported ARC run.
