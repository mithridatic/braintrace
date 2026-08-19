# Example 21 — learned row-wise binding and answer refinement

Status: approved for implementation
Branch: `feat/example21-latent-reasoning`
Target: strict model-only ARC-AGI-1 `pass@2 >= 0.40`

This spec supersedes the proposed `edit_rule` Stage 2 in
`2026-08-18-example21-decoder-redesign.md` and Stages 2–3 in
`2026-08-18-example21-arc-score-recovery.md`. The independent post-scan readout
optimization remains useful only after its regression gate passes and applies
to the legacy carrier-only decoder, not history-dependent row-wise answers. The
Stage 1 rule-channel evidence remains valid. The rule channel remains available
only as a separately attributed diagnostic and must be disabled for the primary
score.

## 1. Evidence and decision

The retained integrated run reaches strict `pass@2 = 27/400`, but 26 of those
task solves are attributable to demonstration-verified host rules. The neural
candidate contributes one exact answer, emits an entire grid from one global
128-wide bottleneck, and has no direct query-grid path. Its recurrent graph has
only 16,384 edges for 2,048 neurons and contributes a small fraction of the
workspace drive.

The replacement must therefore change the learning problem, not add another
host rule family:

1. train on many target-bearing within-task binding episodes;
2. retain the query grid as model state;
3. construct and refine an explicit answer grid row by row;
4. make each generated row depend directly on the recurrent carrier;
5. adapt on each evaluation task's demonstrations without reading the official
   query target;
6. scale neurons and recurrent edges inside measured GPU and graph limits.

The first implementation scale is 4,096 neurons and 1,048,576 recurrent edges,
or 256 edges per neuron. A completed structural probe measured approximately
2.13 GiB of JAX peak allocation and 2,956 MiB process memory on the local
16-GiB RTX 3080 Ti. This is evidence that the starting shape fits; it is not a
training-throughput or score claim.

## 2. Non-negotiable score protocol

The primary result is computed over all admitted ARC-AGI-1 evaluation tasks and
all their official queries.

- Candidate generation is neural only. `rule_channel_enabled` is false.
- No fixed shape, color, geometric, object, or edit rule may emit or alter a
  submitted primary candidate.
- At most two model candidates are submitted per query.
- A task is exact only under the existing strict task aggregation.
- `pass@2 >= 160/400` is the completion gate. Query accuracy may be reported,
  but cannot substitute for task accuracy.
- Evaluation query outputs are available only to the scorer after inference.
  They cannot affect adaptation, candidate generation, effort selection, or
  ranking.
- Every run records source manifest, configuration, seed, parameter hash,
  checkpoint hash, candidate provenance, and peak device memory.

The 6.5% Stage 1 score is not a model baseline. The neural milestones are:

| gate | strict model-only task `pass@2` | purpose |
|---|---:|---|
| M0 | greater than the retained neural count | prove the path can learn exact grids |
| M1 | 0.065 | exceed the complete Stage 1 integrated score without rules |
| M2 | 0.10 | establish useful cross-task binding |
| M3 | 0.20 | establish useful per-task refinement |
| M4 | 0.40 | approved completion target |

## 3. Episode construction

### 3.1 Leave-one-demonstration-out episodes

For every task with demonstrations `(d0, …, dn-1)`, construct `n` supervised
episodes. Fold `i` uses all demonstrations except `di` as context, `di.input` as
the query, and `di.output` as the out-of-band target. Demonstration order is
preserved. The episode carries stable task/fold metadata but never encodes a
task identifier into model features.

The admitted ARC-AGI-1 training collection supplies roughly 1,300 such folds
from 399 tasks. Every admitted task has at least two demonstrations. The helper
still fails closed for a one-demonstration task rather than constructing an
episode with empty context.

Training uses two sources:

- cross-task pretraining from leave-one-out folds; and
- per-task test-time adaptation from only the evaluation task's demonstration
  folds.

Official training queries with labels may be additional cross-task episodes,
but their labels cannot be mixed into evaluation-task adaptation.

### 3.2 Augmentation

Each supervised training episode may be transformed by semantics-preserving
augmentations applied consistently to demonstrations, query, and target:

- a bijective permutation of the ten ARC colors;
- one of the eight dihedral grid transforms;
- bounded whole-grid translations when padding and cropping preserve a defined
  target;
- task-order and demonstration-order sampling.

Random draws use `brainstate.random`. Augmentation provenance is recorded. An
augmentation may not inspect the held-out evaluation query output.

## 4. Model contract

### 4.1 Opt-in compatibility

`ModelConfig` gains an opt-in row-refinement mode. The default stays the current
legacy factorized decoder, with byte-identical initialization and output for
existing configurations. All new random initialization happens after every
legacy draw.

Configuration validates:

- refinement steps are positive and do not exceed `max_latent_steps`;
- a complete sweep is 30 row ticks;
- `recurrent_edges <= 1024 * neuron_count`;
- the physical no-self edge capacity still holds;
- all query/row feature slices are supplied and in bounds when refinement is
  enabled;
- incompatible decoder and loss modes fail closed.

The initial scale is:

```text
neurons          = 4,096
recurrent_edges  = 1,048,576
edges / neuron   = 256
refinement steps = 60 initially (two 30-row sweeps)
```

### 4.2 Explicit state

The opt-in model owns these state tensors:

| state | kind | shape | role |
|---|---|---:|---|
| `query_grid` | `ShortTermState` | `(B,30,30,10)` | captured target-free query colors |
| `query_shape` | `ShortTermState` | `(B,60)` | captured query height/width one-hots |
| `answer_grid` | `ShortTermState` | `(B,30,30,10)` | accumulated answer logits |
| `answer_row` | `HiddenState` | `(B,300)` | current recurrently generated row |
| `answer_shape` | `HiddenState` | `(B,60)` | learned output-shape logits |
| `reasoning_index` | `ShortTermState` | `(B,)` | next row index / sweep position |

Reset clears query and answer state, sets answer logits to a neutral zero prior,
and resets the reasoning index. It must not initialize the answer by copying the
query or invoking any ARC rule.

Valid query-row events capture query colors and shape. Demonstration events do
not write the query buffer. Target colors never enter it.

### 4.3 Row-wise recurrent update

On latent tick `t`, row `r = t mod 30` is refined.

1. Read query row `r` and the current soft answer row `r` from state.
2. Construct a differentiable feedback event in the existing row-event layout.
   It contains query-side row features plus answer-side row probabilities and
   learned/current shape information. It remains a latent event for semantic
   memory gates, so it cannot be mistaken for a demonstration write.
3. Advance feed-forward, recurrent, neuron, and contextual-memory read paths.
4. Apply a direct `neuron_count -> 300` learned row head to the current
   recurrent workspace carrier.
5. Store the row head in `answer_row` and scatter it into `answer_grid[:, r]`.
6. Apply a direct `neuron_count -> 60` shape head and store it in
   `answer_shape`.
7. Advance `reasoning_index`.

The row and shape heads are deliberately attached to `HiddenState` values used
on subsequent ticks. The ETP compiler gate must show their parameter paths on
the temporal/eligibility route. A head classified only as an excluded terminal
tail fails the design gate.

Repeated model execution uses `brainstate.transform.for_loop` or
`brainstate.transform.scan`; no bare Python model loop is allowed. The scan
carry records only requested sweep checkpoints, not a full 9,060-logit tensor
at every physical context tick.

### 4.4 Output and candidates

At a sweep checkpoint, the learned decoder exposes:

- 30 height logits and 30 width logits from `answer_shape`; and
- `(30,30,10)` color logits from `answer_grid`.

Candidate 1 is the joint learned argmax. Candidate 2 must be a distinct learned
hypothesis: initially the best distinct refinement checkpoint or the existing
deterministic logit runner-up operation applied only to model logits. It cannot
come from the rule channel.

The output path must remain compatible with the existing strict decoder and
scorer or introduce an explicit new validated output type. Silent dispatch by
array width is prohibited.

## 5. Learning contract

### 5.1 Cross-task pretraining

Pretraining samples leave-one-out episodes and compiles the full context plus
refinement sweep. Supervision is deep:

- shape cross-entropy at every selected refinement checkpoint;
- color cross-entropy for the row updated at each latent tick;
- full-grid color loss at every completed sweep;
- optional consistency loss between consecutive sweeps, stopped on the older
  sweep so it cannot collapse both predictions together.

Loss masks use only the out-of-band target shape and valid cells. The target is
never concatenated to input events.

The pp-prop learner remains the production update path. Whole-sequence BPTT may
be used only as an oracle/control. Claims about the learning rule use the
finite-window online gradient path required by the repository agreement.

### 5.2 Per-task adaptation

Before predicting an evaluation query:

1. clone or restore the shared pretrained parameters;
2. build leave-one-out folds from that task's demonstrations;
3. perform a bounded number of pp-prop updates on those folds;
4. restore/reset dynamic state between folds as specified by the batch driver;
5. infer the official query without its target;
6. discard task-local adapted parameters before the next task.

The default adaptation scope includes the learned row and shape heads plus the
recurrent/ETP paths proven to participate. An ablation compares head-only,
recurrent-only, and joint adaptation. Per-task state or parameters must never
leak between tasks.

## 6. Resource and participation gates

### 6.1 GPU safety

The launcher sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`. Runtime samples both
JAX allocator statistics and `nvidia-smi` process/device memory. A run fails
closed if observed use exceeds 85% of physical VRAM. Out-of-memory retry may
reduce batch size or checkpoint more aggressively; it may not silently reduce
the configured neuron or edge count in a reported run.

### 6.2 Connectome constraints

- `recurrent_edges <= 1024 * neuron_count` is a configuration invariant.
- The report records realized neuron count, edge count, edges per neuron,
  recurrent parameter share, recurrent-current norm, and feed-forward-current
  norm.
- No score is attributed to added neurons or edges without causal ablation.

Required matched-checkpoint ablations are:

| ablation | question |
|---|---|
| recurrent weights zeroed | do recurrent edges affect exact score and logits? |
| edge count reduced | is the selected density useful? |
| neuron slots ablated | is performance distributed across the population? |
| answer feedback disabled | does iterative refinement matter? |
| one sweep vs later sweep | does more computation improve a fixed model? |
| no demonstrations | is context necessary? |
| shuffled demo outputs | is input/output binding necessary? |
| no per-task adaptation | does test-time learning add score? |

An ablation is causal evidence only when it changes the named mechanism while
holding checkpoint, candidate budget, dataset, and scorer fixed.

## 7. Test plan

Tests are co-located with the modules under test and use the `_test.py` suffix.

### Data tests

- every leave-one-out fold holds out exactly one original demonstration;
- context order and immutable grids are preserved;
- held-out output is target-only and absent from encoded events;
- one-demonstration tasks fail closed;
- transformations apply identically to context, query, and target;
- adaptation episodes never include official evaluation query labels.

### Model tests

- legacy construction and logits remain byte-identical;
- edge-per-neuron cap rejects its first violating value;
- row-refinement configuration rejects incomplete slices and invalid steps;
- reset clears every new state;
- only query events capture `query_grid`;
- a latent tick updates exactly the selected answer row;
- tick 30 wraps to row zero without a Python model loop;
- feedback-disabled and feedback-enabled trajectories diverge after the first
  sweep under controlled weights;
- row/shape head weights are classified on the expected ETP path;
- gradients of both heads and recurrent weights are finite and nonzero on a
  synthetic binding episode;
- chunked and unchunked finite-window losses/updates agree within the existing
  numerical contract.

### Adaptation and scoring tests

- task-local parameters reset between tasks;
- changing an evaluation target after prediction cannot change candidates;
- primary candidates all report neural provenance;
- enabling the rule channel is rejected in model-only mode;
- pass@2 aggregation remains strict by task;
- memory accounting rejects a synthetic value above 85%;
- a tiny synthetic family learns identity, recoloring, non-square shape, and a
  demonstration-dependent row transform before a full ARC run is authorized.

Changed modules target greater than 90% statement and branch coverage, with
particular emphasis on validation, state reset, target isolation, row wrap,
task isolation, and fail-closed resource paths.

## 8. Delivery sequence

1. Land this spec and leave-one-out episode contracts.
2. Land configuration, explicit state, query capture, and reset tests.
3. Land one learned row tick and ETP-classification gate at small scale.
4. Land compiled sweeps, full-grid checkpoints, and deep supervision.
5. Land cross-task pretraining and synthetic-family qualification.
6. Land per-task pp-prop adaptation and target-isolation tests.
7. Run the 4,096-neuron / 1,048,576-edge GPU qualification.
8. Run model-only ARC-AGI-1, then iterate architecture, data, and training
   against M0–M4 without relaxing the primary protocol.

The feature is complete only when M4 and all resource/integrity gates are
retained in one reproducible report. Intermediate milestones are evidence, not
completion claims.
