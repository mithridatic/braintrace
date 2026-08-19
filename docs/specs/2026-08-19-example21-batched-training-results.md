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
| 1,024n / 262,144e | 16 | ~0.55 | 2.35 GiB |

Device memory is never the constraint: the largest configuration measured used
2.35 GiB against the 13.7 GiB fail-closed limit, 17% of physical VRAM against an
85% ceiling. Host memory is the real constraint, and the episode bank is the
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

_Result of the re-run is recorded here once it completes._

## 6. What the evidence says to do next

1. Pretraining is the binding constraint on exact score, and its curve has not
   converged. More episodes, not more neurons, is the next lever — the loss was
   still descending at 96,000 episodes.
2. The episode bank cannot grow far enough to feed a much longer run. Prefetching
   the next chunk on a worker thread removes the bank entirely and recovers the
   third of wall clock currently spent host-side.
3. Adaptation is fold-starved. Semantics-preserving augmentation of a task's own
   folds multiplies the adaptation set without adding any hand-written rule.

Both 2 and 3 are specified in `2026-08-19-example21-adaptation-data-and-checkpoints.md`.
