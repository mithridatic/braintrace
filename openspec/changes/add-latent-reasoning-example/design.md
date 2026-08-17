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
  parameters on byte-identical tasks after 0, 8, 16, and 32 latent steps.
- Use 2,048 physical LIF neurons and 16,384 recurrent sparse edges for the full
  experiment, with a reduced configuration only for smoke and unit tests.
- Make data provenance and train/evaluation separation auditable.
- Measure both answer quality and the evolution and causal relevance of the
  latent trajectory.

**Non-goals:**

- Reproducing the proprietary BDH-CQ implementation, its private training set,
  reported benchmark result, or inference cost.
- Claiming state-of-the-art ARC performance.
- Treating pixel accuracy as ARC success or treating a synthetic smoke task as
  model-quality evidence.
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

The full configuration has 2,048 `brainpy.state.LIF` neurons, interpreted for
analysis as 32 slots of 64 neurons. Input reaches the population through a
BrainTrace `Linear` operation and an `Expon`/`CUBA` current synapse. Recurrent
reasoning uses an `AlignPostProj` path with a BrainTrace `SparseLinear` operator,
an exponential synapse, and exactly 16,384 nonzero directed edges (mean out
degree eight). Self-edges are excluded unless explicitly enabled and reported.

The grid prediction head is low-rank: a BrainTrace projection from 2,048 spikes
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
the gate is true, so the recurrent network continues for up to 32 compiled
latent steps. Training updates draw terminal
effort from 8, 16, and 32, so all effort levels supervise the same parameters;
there is not one separately trained model per depth. The effort schedule and
counts are recorded.

At evaluation, one frozen 32-step trajectory supplies checkpoints at 0, 8, 16,
and 32. Checkpoint 0 is the query-terminal state before any zero-input update.
Task order, encoded inputs, initial state, parameters, and decoding are identical
across checkpoints. Thus score differences isolate additional recurrent
updates, not retraining or resampling.

The paper names LOW/MEDIUM/HIGH effort conditions without publishing iteration
counts. The local 8/16/32 recurrent ticks are an operational proxy, not a claim
that these checkpoints correspond to the paper's proprietary tiers.

### D6. pp-prop receives terminal ARC supervision

`braintrace.compile(..., braintrace.pp_prop(...))` compiles the recurrent model,
and `etrace_grad` supplies online parameter gradients. A training example has
three cross-entropy components at its sampled terminal checkpoint: height,
width, and valid target-cell colors. Shape components are never inferred from
the target while decoding. Loss weights and optimizer groups are reported.

Only a terminal checkpoint is supervised for a sampled training episode. The
training driver may compile the three supported effort lengths separately, but
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
- truncation: the required 0/8/16/32 checkpoints;
- slot ablation: zero the recorded activity of a deterministic 64-neuron slot
  before continuing the recurrent rollout.

Controls report exact ARC metrics and state deltas. A control that produces a
byte-identical trajectory is explicitly reported as causally null rather than
being interpreted from accuracy alone.

### D9. Full and smoke regimes have different evidentiary status

The full run requires the 2,048-neuron, 16,384-edge configuration and defaults
to GPU. A larger 4,096/32,768 arm is optional only after measured memory
headroom. Unit tests and `--smoke` use a smaller network and a tiny embedded ARC
fixture to exercise the complete pipeline quickly; their scores are marked
plumbing-only and cannot satisfy the scientific acceptance criteria.

The full report includes device/backend, seeds, parameter count, actual sparse
edge count, training updates by effort, dataset manifest hashes, split counts,
runtime, peak memory when available, and every metric/control above. Generated
datasets, downloaded corpora, checkpoints, and reports remain outside Git.

## Risks / Trade-offs

- **ARC is much harder than symbol lookup.** Exact scores may remain near zero.
  The experiment still remains valid because near-miss diagnostics and causal
  trajectory evidence are reported without relabelling them as success.
- **Long pp-prop traces at 2,048 neurons are expensive.** Row streaming and the
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

## Migration Plan

1. Replace and strictly validate the active OpenSpec artifacts.
2. Replace the repository specification before implementation.
3. Implement and test ARC data/provenance, scoring/analysis, then the spiking
   model and driver.
4. Run focused coverage and the repository gate, then a full GPU qualification
   if public data and runtime are available.
5. Commit only source, specs, tests, and documentation on
   `feat/example21-latent-reasoning`; keep downloaded and generated artifacts
   untracked.

## Open Questions

- Exact public-corpus availability is environmental. The run records which
  approved sources were actually present; it never silently substitutes private
  or synthetic data.
- A 4,096-neuron scaling arm remains optional and is not part of acceptance.
