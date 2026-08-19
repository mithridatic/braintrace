# Example 21 — batched cross-task training results

Status: results, held-out training-split measurements final; complete ARC number
recorded in §5
Date: 2026-08-19
Branch: `feat/example21-row-refinement`
Spec: `2026-08-19-example21-batched-cross-task-training.md`

All numbers below come from the local RTX 3080 Ti Laptop GPU (16 GiB, WDDM)
through the `braintrace-gpu:0.11.0-py314` container.

## 1. The architecture was never the problem

The synthetic row-binding qualification, run at 400 updates with 256 neurons and
4,096 recurrent edges:

| family | exact before | exact after |
|---|---|---|
| identity | no | yes |
| recolor | no | yes |
| non_square_transpose | no | yes |
| demo_dependent_row_reverse | no | yes |

Loss 0.4004 → 0.0347, all four families exact, every strict gate satisfied. At 8
updates only `identity` is learned, so the qualification is measuring learning
rather than initialization. Learned shape change and demonstration-conditioned
transformation are both inside the row-refinement path's capacity.

## 2. Held-out training-split learning curve

A probe trains on 367 ARC-AGI-1 *training* tasks and scores 32 training tasks
that contribute no gradient (`var/qual/train_probe.py`). Learning rate 3e-3,
augmentation on, 512 neurons, 65,536 recurrent edges.

| episodes | updates | batch | shape accuracy | pixel accuracy | mean loss |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | — | 0.0000 | 0.1019 | — |
| 160 | 20 | 8 | 0.1875 | 0.5902 | 0.858 |
| 9,600 | 600 | 16 | 0.4375 | 0.6716 | 0.737 |
| 96,000 | 6,000 | 16 | 0.5938 | 0.6905 | 0.468 |

Loss deciles over the longest run: 0.772, 0.616, 0.555, 0.521, 0.514, 0.500,
0.492, 0.483, 0.473, 0.468 — still descending at the end, so the curve is not
converged and episodes remain the limiting resource.

The retained runs this replaces used one episode per update at `1e-4`, which is
why they reported shape accuracy 0.0000 and therefore exact `pass@2` 0.0000: no
color path can produce an exact grid of the wrong shape.

Refinement direction also reverses once the model has learned anything. At
96,000 episodes, sweep 2 beats sweep 1 on both axes (shape 0.5312 → 0.5938,
pixel 0.6539 → 0.6905), where the untrained runs decayed monotonically with
effort (0.1499 → 0.1147 → 0.0684).

## 3. Per-task adaptation is what makes an answer exact

Cloning the pretrained parameters per held-out training task and adapting on
that task's own leave-one-out folds — never the scored target
(`var/qual/adapt_probe.py`, 16 tasks, 15 with enough demonstrations):

| | shape accuracy | exact pass@1 | exact pass@2 | pixel accuracy |
|---|---:|---:|---:|---:|
| frozen | 0.4667 | 0.0000 | 0.0000 | 0.4482 |
| adapted | 0.7333 | 0.0000 | 0.0667 | 0.5108 |

Adaptation budget sweep, same checkpoint and tasks:

| steps | rate | shape accuracy | exact pass@2 |
|---:|---:|---:|---:|
| 40 | 3e-3 | 0.7333 | 0.0667 |
| 120 | 3e-3 | 0.6667 | 0.0667 |
| 300 | 3e-3 | 0.6667 | 0.0667 |
| 120 | 1e-2 | 0.6667 | 0.0667 |

Exact score is flat across a 7.5× range of budgets and a 10× range of rates.
Past roughly 40 steps the pretrained model and the small number of available
folds bind, not the optimizer budget. The budget below 40 steps is unmeasured.

## 4. Cost and resource envelope

Measured through the production pipeline:

| scale | batch | seconds/update | peak device |
|---|---:|---:|---:|
| 512n / 65,536e | 1 | 0.075 | 0.52 GiB |
| 512n / 65,536e | 16 | 0.135 | 2.09 GiB |
| 4,096n / 1,048,576e | 1 | 0.210 | 0.61 GiB |
| 1,024n / 262,144e | 16 | 0.319 | 4.04 GiB |

The last row is the reported configuration, measured over a complete
6,000-update pretraining run: 2,167.6 s total, of which roughly 1,912 s is
training and 180 s is bank encoding.

Device memory is never the constraint: the largest configuration measured used
4.04 GiB against the 13.7 GiB fail-closed limit, 25% of physical VRAM against an
85% ceiling. Peak scales with `training_chunk_size`, not with model size.

Host memory is the real constraint, and the episode bank is the
reason — 1.29 MB per encoded episode means a 4,000-episode-per-effort bank holds
10.4 GB, measured at 14.76 GiB of a 23.47 GiB container.

Realized connectome shape for the reported run: 1,024 neurons, 262,144 recurrent
edges, 256 edges per neuron, 25% of the 1,024-per-neuron policy cap. That is a
32-fold increase in edges per neuron over the 8 per neuron of the runs this
replaces.

## 5. Complete 400-task model-only ARC result

**First attempt failed on a device fault, not a score.** The complete run reached
the 400-task adaptation stage after 64 minutes and died with
`CUDA_ERROR_UNKNOWN: Failed to check stream capturing status` at the first host
synchronisation. The GPU recovered fully and passed a matmul check immediately
afterwards, and the fault is attributed to dispatching all 400 tasks' adaptation
as one compiled program on a WDDM device. Two changes follow: task-grouped
evaluation bounds each dispatch, and pretrained parameters are now written to a
checkpoint so a fault in the evaluation stage no longer destroys the training
stage.

### 5.1 The complete run

Re-run from the pretrained checkpoint with task-grouped dispatch: 400 evaluation
tasks, 419 official queries, complete split, `primary_candidate_mode` model-only,
rule channel off, no evaluation target reachable before the scorer. Total runtime
3,761.9 s, of which task-local adaptation is 3,350.1 s.

| effort | query pass@1 | query pass@2 | strict task pass@1 | strict task pass@2 | shape | pixel |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (diagnostic) | 0.0024 | 0.0024 | 0.0025 | 0.0025 | 0.0453 | 0.2277 |
| 30 (diagnostic) | 0.0000 | 0.0024 | 0.0000 | 0.0000 | 0.5227 | 0.4734 |
| **60 (submitted)** | 0.0000 | **0.0024** | 0.0000 | **0.0025** | **0.6014** | **0.4902** |

**Strict model-only task `pass@2` is 1 of 400.** The solved task is `7039b2d7`,
a 3 by 3 output, decoded from the model's own refined logits as candidate 2
(`latest_sweep_logit_runner_up`, provenance `model`). Candidate 1 is exact on no
query at any effort, so `pass@1` is zero.

This does not clear milestone M0. The retained integrated run's neural candidate
also contributed one exact answer, so the count ties rather than exceeds it. What
changed is everything upstream of exactness: shape accuracy 0.0000 → 0.6014 and
valid-cell pixel accuracy 0.1019-class → 0.4902 on the complete split. Effort 60
and effort 0 each solve one task, but not the same one (`7039b2d7` versus
`642d658d`), so more computation redistributed rather than accumulated exact
answers.

### 5.2 Adaptation is causally responsible for the score

Same checkpoint, same split, same scorer, adaptation the only difference:

| | shape | pixel | strict task pass@2 |
|---|---:|---:|---:|
| frozen, no adaptation | 0.0310 | 0.2235 | 0.0000 |
| task-local pp-prop adaptation | 0.6014 | 0.4902 | 0.0025 |

13,630 fold updates were applied — 1,363 distinct leave-one-out folds replayed
over 10 epochs, fold capacity 70, learning rate 3e-3, task group 20. Parameters
were restored to the shared checkpoint after every task and the run verified
that restoration.

### 5.3 Where the 419 queries actually land

Mismatched valid cells for candidate 1 at the submitted effort:

| distance from target | queries | share |
|---|---:|---:|
| exact | 0 | 0.0% |
| 1–3 cells | 11 | 2.6% |
| 4–10 cells | 42 | 10.0% |
| 11–30 cells | 67 | 16.0% |
| 31+ cells | 132 | 31.5% |
| wrong shape | 167 | 39.9% |

Two things follow that the aggregate pixel number hides. Wrong shape is the
single largest bucket at 40%, and shape is only 60 logits — the cheapest place to
buy accuracy. And 11 queries sit within three cells of exact, with 42 more within
ten; candidate 2 currently flips exactly one decision, so a candidate that
repairs the k lowest-margin cells jointly, still purely from model logits, is
addressable score rather than speculation.

The earlier six-task observation that candidate 2 never varies the output shape
does not hold at scale: candidate 2 proposes a different shape on 70 of the 167
wrong-shape queries, 42%.

**A follow-up measurement kills the candidate-construction lever.** Resolving the
near-miss tail by exact distance gives 3 queries one cell off, 5 two cells off,
and 3 three cells off. For the three one-cell misses candidate 2 does change
exactly one cell, and only one of the three becomes exact — the single solve of
the whole run. Logit margin therefore identifies the wrong cell roughly one time
in three. Four of the five two-cell misses receive a candidate 2 that changes a
single cell and so cannot be exact at all; a two-cell repair would have to pick
both wrong cells, at an implied success rate near one in nine.

The tail is thin and the ranking that would exploit it is weakly informative, so
better candidate construction is worth well under one additional exact answer.
The lever is model accuracy — pretraining episodes and adaptation data — not
decoding. This is the opposite of what the bucketed distribution alone suggests,
which is why the tail was resolved cell by cell before acting on it.

### 5.4 Resource and participation evidence

- Realized 1,024 neurons, 262,144 recurrent edges, 256 per neuron, 25.0% of the
  1,024-per-neuron policy cap, no violations.
- Peak device memory 3.69 GiB against the 13.74 GiB fail-closed allocator limit,
  27%. Host-side `nvidia-smi` sampling peaked near 5.5 GiB of 16,384 MiB, 34%.
- The run's own `nvidia-smi` cross-check could not be collected inside the
  container (`process_peak_bytes_missing`, exit status 3 and 17, "no unique device
  memory row"), so `gpu_runtime_resource_safe` reports **insufficient evidence**
  rather than safe. The allocator evidence is `safe`; the second, independent
  measurement is missing. This is a WDDM/container limitation, not a breach.
- Recurrent-current L2 is 2,285.7 against feed-forward 3,375.3, so recurrent
  edges carry 40% of the workspace drive. The runs this replaces measured 43
  against 285, about 13%.

### 5.5 What this artifact cannot claim

`pp_prop_compiler_routes` is false and the reported temporal route count is zero
**because this run restored a checkpoint and performed no optimizer update**. The
pp-prop route evidence — seven eligibility-trace temporal routes including both
learned row and shape heads — lives in the pretraining artifact under
`var/example21-pretrain-1024`, not here. Structural and scientific qualification
therefore remain false for this artifact, as does the 40% completion gate. Split
across two artifacts is a reporting weakness introduced by checkpoint restore and
should be fixed by carrying the checkpoint's recorded training evidence forward
into the restored run's report.

## 5.6 Augmented adaptation folds: gate not met

Adaptation is fold-starved, so a task's own folds were augmented with seven
semantics-preserving transformed copies — the same colour permutation and
dihedral family the training path already uses, applied consistently to every
demonstration of a variant, reading no scored target. Matched checkpoint, matched
tasks, 40 adaptation steps at 3e-3, 28 held-out **training** tasks:

| arm | shape | pixel | exact pass@2 |
|---|---:|---:|---:|
| frozen, no adaptation | 0.5357 | 0.5447 | 0.0000 |
| adapted, no augmentation | 0.5714 | 0.5534 | 0.0357 |
| adapted, 7 augmentations | 0.5357 | 0.6006 | 0.0000 |

Augmentation buys a real pixel-accuracy gain, +0.047, the largest single-change
pixel gain measured in this session. It does not buy exact answers: 0 of 28
against 1 of 28. At that sample size 1 versus 0 is indistinguishable, but the
gate in `2026-08-19-example21-adaptation-data-and-checkpoints.md` §5 requires
exceeding the baseline exact score, and it does not. **The option stays off in
reported runs.** A 15-task run of the same comparison agreed on direction (pixel
0.5108 → 0.5409, exact 1 of 15 → 0 of 15).

The pixel gain is worth revisiting once the model is close enough to exact for
pixel accuracy to convert, which §5.3 shows it currently is not.

## 6. What the evidence says to do next

1. Pretraining is the binding constraint on exact score, and its curve has not
   converged. More episodes, not more neurons, is the next lever — the loss was
   still descending at 96,000 episodes. Every other lever measured this session
   either failed its gate (augmented adaptation folds, §5.6), was worth under one
   exact answer (candidate construction, §5.3), or was already saturated
   (adaptation budget, §3).
2. The episode bank cannot grow far enough to feed a much longer run. Prefetching
   the next chunk on a worker thread removes the bank entirely and recovers the
   third of wall clock currently spent host-side.
3. Adaptation is fold-starved. Semantics-preserving augmentation of a task's own
   folds multiplies the adaptation set without adding any hand-written rule.

4. Candidate 2 never proposes a different output shape. On the six-task
   validation subset, both wrong-shape queries had a candidate 2 with the same
   shape as candidate 1, so the second submission slot cannot rescue the
   dominant failure mode. Deriving candidate 2's shape from the runner-up of the
   learned shape logits stays model-only and costs nothing, but it only pays off
   once the color path is close enough for a shape fix to produce an exact grid —
   the measured miss sizes are 11 or more cells even when the shape is right.

Items 2 and 3 are specified in
`2026-08-19-example21-adaptation-data-and-checkpoints.md`.
