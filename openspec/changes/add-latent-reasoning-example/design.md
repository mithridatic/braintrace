## Context

The paper evaluates recurrent latent computation on ordinary ARC tasks: several
input/output demonstrations define a task, one or more test inputs must be
completed, and success requires an exact output grid. Its private architecture,
private data, and training recipe are not available. Examples 18–20 provide the
implementable substrate in this repository: BrainPy LIF neurons, exponential
current synapses, BrainTrace dense and sparse operators, and pp-prop online
updates.

The previous Example 21 instead generated one-symbol permutation lookups and
used a hand-written binary state machine. That prototype cannot answer whether
latent spiking computation improves ARC solutions. This change replaces its
scientific contract and implementation while retaining the failed experiment in
Git history.

## Goals / Non-Goals

**Goals:**

- Preserve standard ARC semantics: rectangular color grids, multiple training
  demonstrations, multiple test queries, variable output shape, and exact-grid
  scoring.
- Train one recurrent spiking model with pp-prop and evaluate the same frozen
  parameters on byte-identical tasks at efforts 0, 30, and 60, with effort 60
  nominated for submission.
- Use 4,096 physical LIF neurons and 4,194,304 recurrent sparse edges for the
  full experiment, with a reduced configuration only for smoke and unit tests.
- Make data provenance and train/evaluation separation auditable.
- Measure both answer quality and the evolution and causal relevance of the
  latent trajectory.
- Require a cumulative exact score of at least 16 through candidates whose
  bytes and exact membership demonstrably depend on the trained checkpoint.

**Non-goals:**

- Reproducing the proprietary BDH-CQ implementation, its private training set,
  reported benchmark result, or inference cost.
- Claiming state-of-the-art ARC performance.
- Treating pixel accuracy as ARC success or treating a synthetic smoke task as
  model-quality evidence.
- Counting raw demonstration-only forest or rule candidates as network answers,
  even when those candidates are accurate.
- Making a BPTT-equivalence claim for pp-prop.

## Decisions

### D1. Standard ARC is the task contract

An `ArcTask` contains one or more training input/output pairs and one or more
test inputs, with optional test outputs when evaluating. Grids are rectangular,
1–30 cells on each axis, and contain integer colors 0–9. Each test query is run
as an episode with the complete demonstration set; metrics are aggregated back
to the original task so multi-test tasks are strict only when every query is
correct.

The loader accepts the ordinary per-task JSON shape and collection/JSONL
adapters used by public derived corpora. It validates rather than silently
cropping grids or demonstrations. A canonical content fingerprint ignores file
names and metadata so renamed copies are still detected across splits.

### D2. Row events preserve the whole two-dimensional problem

Each grid is encoded as a sequence of at most 30 row events. A row event carries
the row index, grid height and width, phase/type fields, valid-cell mask, and a
position-specific one-hot color for all 30 columns. A demonstration event may
carry one input row and one output row; a query event carries only its input
row. Ten fixed 30-row demonstration blocks plus one query block produce a
default `(330, 830)` context tensor. Missing rows and cells are explicitly
masked. No rule label, target test output, task identifier, or source identifier
enters the model.

This representation is lossless and bounded while avoiding a 9,000-wide dense
input projection. A fixed-shape batch pads unused demonstrations and rows after
an explicit validity field. Unused blocks freeze state. Occupied demonstration
blocks use a fixed recurrent clock even on exact-zero missing-row ticks, so an
unequal-height output derangement changes associations and content placement
without changing context duration. Repeated model execution uses
`brainstate.transform.for_loop`; Python loops may prepare static host data but
never drive neuron or synapse updates.

### D3. Public data has a fail-closed provenance boundary

The training adapter can ingest public sources named by the paper when the user
has obtained them: ARC-AGI-1 training, RE-ARC, ConceptARC, ARC-Heavy, and
ARC-GEN100K. The manifest records source name, source role, declared version,
license/reference metadata supplied by the dataset manifest, path, file hashes,
task counts, and deduplication outcomes. It never labels private paper data as
available.

ARC-AGI-1 evaluation is evaluation-only. A fresh generator adapter may provide
an additional `arc-task-gen` evaluation set, also evaluation-only. Canonical
fingerprints are compared before training; any overlap between training,
validation/tuning, and either evaluation set aborts the run. Model selection and
early stopping may use a held-out split of training-role sources, never an
evaluation source.

Color permutation, dihedral grid transforms, and demonstration-order shuffling
are training augmentations. They use `brainstate.random`, preserve each task's
input/output relation, and never mutate evaluation examples.

### D4. The full model is an Example-18-style spiking network

The full configuration has 4,096 `brainpy.state.LIF` neurons, interpreted for
analysis as 64 slots of 64 neurons. Input reaches the population through a
BrainTrace `Linear` operation and an `Expon`/`CUBA` current synapse. Recurrent
reasoning uses an `AlignPostProj` path with a BrainTrace `SparseLinear` operator,
an exponential synapse, and exactly 4,194,304 nonzero directed edges (mean out
degree 1,024). Self-edges are excluded unless explicitly enabled and reported.

The grid prediction head is low-rank: a BrainTrace projection from 4,096 spikes
to a configurable bottleneck, followed by separate output-height and
output-width heads and CP row/column/color factors. The compact width is
`60 + rank * (30 + 30 + 10)` (1,180 at rank 16); a tensor product expands those
factors to 30×30×10 logits for loss and analysis. This deliberately constrains
the repository model relative to an unrestricted 9,000-logit head and is not
claimed as a paper detail. Invalid cells are masked by the target shape. The
network predicts the output rather than copying an oracle-derived transform.

The slow input synaptic current and neuronal state provide contextual memory;
the recurrent LIF voltage/spikes form the latent workspace. Both are ordinary
model state, not trainable parameters, and are reset between query episodes.
The implementation exposes synaptic current, voltage, and spikes separately so
the report cannot confuse them.

### D5. One model supports variable latent effort

The model call takes an external event and a state-advance gate. Invalid unused
capacity has a zero event and a false gate, restoring voltage and both synaptic
currents exactly. After the last query row, the event remains exactly zero but
the gate is true, so the recurrent network continues for up to 60 compiled
latent steps. The accepted effort set is 0, 30, and 60. Training updates that
sample a post-query terminal effort use 30 or 60, so both nonzero effort levels
supervise the same parameters; there is not one separately trained model per
depth. The effort schedule and counts are recorded.

At evaluation, one frozen 60-step trajectory supplies checkpoints at 0, 30,
and 60. Checkpoint 0 is the query-terminal state before any zero-input update.
Task order, encoded inputs, initial state, parameters, and decoding are identical
across checkpoints. Thus score differences isolate additional recurrent
updates, not retraining or resampling.

The paper names LOW/MEDIUM/HIGH effort conditions without publishing iteration
counts. The local 0/30/60 effort checkpoints are an operational proxy, not a
claim that these checkpoints correspond to the paper's proprietary tiers.

### D6. pp-prop receives terminal ARC supervision

`braintrace.compile(..., braintrace.pp_prop(...))` compiles the recurrent model,
and `etrace_grad` supplies online parameter gradients. A training example has
three cross-entropy components at its sampled terminal checkpoint: height,
width, and valid target-cell colors. Shape components are never inferred from
the target while decoding. Loss weights and optimizer groups are reported.

Only a terminal checkpoint is supervised for a sampled training episode. The
training driver may compile the supported nonzero effort lengths separately, but
they all share one parameter object and optimizer state. Repeated simulation is
expressed with `brainstate.transform.for_loop` or `scan`, never a bare Python
loop.

### D7. ARC scoring remains exact

Candidate one is the joint argmax for height, width, and each cell. Candidate two
is a deterministic single-decision alternative: among the second-best height,
second-best width, and the second-best color at every cell inside candidate
one's shape, choose the alternative with the smallest logit-margin penalty and
decode the resulting grid. Duplicate candidates collapse to one.

Per-query pass@1 and pass@2 require exact shape and every cell correct. Strict
task pass@k requires every test query in that task to pass under its first `k`
candidates. Shape accuracy and valid-cell pixel accuracy are labelled diagnostic
near-miss metrics and are never substituted for exact success.

### D8. Latent evidence includes trajectory and causal controls

For every latent checkpoint, the result stores provisional candidates, changed
cell count from the previous checkpoint, output entropy and margin, spike count
and firing rate, voltage norm, state displacement, and convergence/fixed-point
indicators. A bounded neuron sample supports spike-raster plots without storing
an unbounded dense trace.

Frozen evaluation includes:

- no-context: demonstrations are masked while the query is unchanged;
- shuffled-demonstration: demonstration outputs are deranged across pairs while
  their inputs, shapes, query, and tensor magnitudes are retained;
- truncation: the required 0/30/60 checkpoints;
- slot ablation: zero the recorded activity of a deterministic 64-neuron slot
  before continuing the recurrent rollout.

Controls report exact ARC metrics and state deltas. A control that produces a
byte-identical trajectory is explicitly reported as causally null rather than
being interpreted from accuracy alone.

### D9. Full and smoke regimes have different evidentiary status

The full run requires the 4,096-neuron, 4,194,304-edge configuration and
defaults to GPU. Unit tests and `--smoke` use a smaller network and a tiny
embedded ARC fixture to exercise the complete pipeline quickly; their scores
are marked plumbing-only and cannot satisfy the scientific acceptance criteria.

The full report includes device/backend, seeds, parameter count, actual sparse
edge count, training updates by effort, dataset manifest hashes, split counts,
runtime, peak memory when available, and every metric/control above. Generated
datasets, downloaded corpora, checkpoints, and reports remain outside Git.

Same-process, same-device repeatability is checked separately for every retained
checkpoint and evaluation query. Spikes, decoded candidates, exact scores, and
diagnostics must be identical. The normative physical-state gate requires
float32 voltage, feedforward synaptic current, and recurrent synaptic current,
each with per-query RMS across neurons at most `1e-6` for every
checkpoint-query pair. An additional decoder-state check applies the same
threshold across compact-logit features so unchanged candidates cannot hide
material logit drift. This is a fixed float32 reproducibility tolerance, not an
ARC score tolerance. Literal byte identity remains a separate reported fact.
The slot-ablation arm must satisfy the same state, candidate, and metric gate at
its pre-intervention checkpoint before any post-intervention difference is
treated as causal.

### D10. Submitted candidates are checkpoint-owned and perturbation-qualified

The accepted full-matrix profile fixes seed 31337, 60 latent steps, retained
efforts 0/30/60, submission effort 60, and the `checkpoint_conditioned` answer
head. Baseline, repeat, scale, trained-checkpoint swap, and deterministic reseed
all use this profile; only the checkpoint intervention differs as specified
below.

The prior demonstration-fitted forest exposes a useful conjunctive inductive
bias, but it computes candidates from raw demonstrations without reading the
trained parameters or recurrent trajectory. Its previously measured cumulative
32 is therefore a `demonstration_only_diagnostic`; the earlier carrier-row
parameter-consuming baseline at cumulative 6 was also superseded and is not the
accepted result. Diagnostic forest and rule candidates remain reportable but
may not occupy either primary candidate slot in their raw ordering.

Every counted candidate is ordered by a target-free executed network path that
consumes recurrent carrier or memory features and model-owned parameter leaves
restored from the trained checkpoint. A task-local forest may generate bounded
proposal grids, but demonstrations may not overwrite checkpoint leaves and the
raw forest order is diagnostic-only. Each proposal is submitted in descending
order of

```text
forest_log_probability
+ 1.0 * trained_network_candidate_log_probability,
```

where the network term is the factorized model likelihood of the candidate's
predicted dimensions and row-major cells. The coefficient is fixed before
evaluation-label scoring. Each submitted candidate records its proposal source,
ranking source, dependency class, answer-head version, score components, and
participating parameter-leaf paths. A separately appended model-dependent
second candidate cannot launder an unranked demonstration-only first candidate.

A read-only 419-query prototype replay selected this design: the fixed
seed-31337 checkpoint produced `9/9/7/7 = 32`; scaling the same logits by 0.5
produced `8/9/6/7 = 30`; a same-schema update-700 checkpoint produced
`8/9/7/7 = 31`; and deterministic reseeded logits produced `7/9/5/7 = 28`.
Those prototype measurements established direction but were nonqualifying.

The subsequent authoritative production-path GPU matrix produced baseline and
exact repeat `8/9/6/7 = 30`; exact 0.5x scale, independently trained swap, and
deterministic BrainState seed-73 reseed each produced `7/9/5/7 = 28`. Repeat
preserved canonical candidate and membership bytes. Every perturbation changed
parameter bytes, candidate bytes, exact membership, and cumulative score from
baseline. The full-profile, manifest, checkpoint-schema, training-origin, and
execution-contract audits all passed.

The qualifying checkpoint is nominated before complete evaluation-label
scoring and must reach

```text
query pass@1 count + query pass@2 count
+ strict task pass@1 count + strict task pass@2 count >= 16.
```

The integer counts are reported separately; the sum is a local engineering
gate rather than an official ARC metric. Candidate construction receives no
held-out output or target-derived selector.

Qualification runs one matched eval-only matrix: baseline, reload/repeat,
predeclared non-unit scaling of every floating parameter leaf on the recorded
answer path, an independently seeded trained same-schema checkpoint swap, and a
deterministic `brainstate.random` reseed of the exact baseline parameter schema.
All three perturbations are mandatory and none substitutes for another. Repeat
must preserve canonical prediction bytes, exact-membership bytes, all four
counts, and the cumulative score. Each perturbation must independently change
the answer-parameter digest, canonical prediction bytes, exact membership, and
the cumulative integer score. A flat score fails even when logits or state move.

Canonical prediction bytes include ordered candidate rank, dimensions, and
row-major colors in manifest order; reranking the same proposal set therefore
changes the digest, while metadata and score values cannot satisfy movement.
Exact rank membership contains ordered per-query pass@1/pass@2 and per-task
strict pass@1/pass@2 booleans. Reports include full-checkpoint,
participating-parameter, topology, ordered-candidate, membership, and manifest
SHA-256 values plus the exact changed query/task identifiers and ranks.
Checkpoint restoration validates ordered leaf path, shape, and dtype and fails
closed on partial loading.

If the qualifying arm claims EI/Dale dependence, it additionally records the
neuron-type mask and proves zero effective-weight sign violations. A matched
predeclared sign control must move candidate bytes, exact memberships, and
cumulative score; otherwise Dale dependence is reported as null. This optional
evidence does not replace the mandatory checkpoint perturbations.

Repeated network, task-local model-routing, and trainable-state updates use
`brainstate.transform` loop primitives; host Python loops are limited to static
metadata preparation and scoring already-produced outputs. All initialization,
topology, augmentation, reseed, and perturbation randomness uses
`brainstate.random`, never `jax.random` directly.

## Risks / Trade-offs

- **ARC is much harder than symbol lookup.** Exact scores may remain near zero.
  The experiment still remains valid because near-miss diagnostics and causal
  trajectory evidence are reported without relabelling them as success.
- **Long pp-prop traces at 4,096 neurons are expensive.** Row streaming and the
  low-rank output head bound dense widths; full qualification is GPU-only and
  smoke runs are explicitly non-scientific.
- **A recurrent SNN can saturate or go silent.** The report measures occupancy,
  voltage, state movement, and trajectory diversity. It does not tune thresholds
  on ARC evaluation. Instability aborts qualification rather than being hidden.
- **Derived datasets have heterogeneous formats and licenses.** Each source is
  optional behind a manifest-backed adapter. Unknown provenance or a missing
  license/reference field is reported and excluded by default.
- **The paper's internals are unavailable.** All prose calls this a repository
  instantiation of the observable task/effort contract, not a reproduction.
- **Parameter sensitivity can be cosmetic.** Candidate-level ownership plus
  prediction-byte, exact-membership, and cumulative-score movement prevents a
  changed unused logit or metadata field from qualifying a demonstration-only
  answer.

## Migration Plan

1. Replace and strictly validate the active OpenSpec artifacts.
2. Replace the repository specification before implementation.
3. Implement and test ARC data/provenance, scoring/analysis, then the spiking
   model and driver.
4. Replace demonstration-only primary answers with a target-free,
   checkpoint-conditioned answer path and fail-closed provenance reporting.
5. Run focused coverage and the repository gate, then the full baseline,
   repeat, scale, same-schema trained-swap, and deterministic-reseed matrix.
6. Commit only source, specs, tests, and documentation on the active worktree
   branch; keep downloaded and generated artifacts untracked.

## Open Questions

- Exact public-corpus availability is environmental. The run records which
  approved sources were actually present; it never silently substitutes private
  or synthetic data.
- The full-matrix acceptance profile is locked to 4,096 neurons, 4,194,304
  recurrent edges, efforts 0/30/60, submission effort 60, seed 31337, and the
  `checkpoint_conditioned` answer head; changing it requires a new approved
  specification rather than an ad hoc scaling arm.
