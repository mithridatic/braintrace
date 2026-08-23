## Purpose

Define an auditable experiment that trains one recurrent spiking network with
pp-prop on standard ARC-format tasks, varies its inference-time latent effort,
and measures exact task quality and the resulting latent trajectory.

## ADDED Requirements

### Requirement: Standard ARC task semantics

The example SHALL represent ordinary ARC tasks containing one or more training
input/output pairs and one or more test inputs, with rectangular 1–30 by 1–30
grids whose cells are colors 0–9. It SHALL predict both output dimensions and
all output cells without reading the held-out output.

#### Scenario: Variable grids and multiple demonstrations survive encoding

- **WHEN** a valid task with unequal input/output dimensions and multiple training pairs is encoded and decoded
- **THEN** every grid, dimension, color, pair boundary, and query boundary is recovered exactly

#### Scenario: Multiple test queries remain one task

- **WHEN** a task contains multiple test inputs
- **THEN** each query is evaluated with the complete shared demonstrations and results retain the task and query indices needed for strict whole-task scoring

#### Scenario: Invalid ARC data fails closed

- **WHEN** a grid is ragged, empty, larger than 30 on either axis, contains a value outside 0–9, or a task lacks training or test examples
- **THEN** loading raises a clear error naming the task and offending field instead of cropping or coercing it

### Requirement: Lossless bounded event representation

The example SHALL convert ARC grids to bounded row events carrying row and grid
positions, dimensions, masks, phase/type fields, and position-specific colors.
Padding SHALL be distinguishable from valid content. A test target, task ID,
source ID, or transformation label SHALL NOT enter the model input.

#### Scenario: Query encoding has no target leakage

- **WHEN** two copies of a task differ only in the held-out test output
- **THEN** their demonstration and query event tensors are byte-identical

#### Scenario: Padding is inert

- **WHEN** the configured static capacity exceeds a task's demonstrations or row count
- **THEN** padded events are explicitly invalid and produce no external drive

#### Scenario: Padding and latent zero input are distinct

- **WHEN** an invalid unused-capacity tick and a post-query latent tick both carry the exact zero event
- **THEN** an explicit advance gate freezes model state for the former and advances recurrence for the latter

#### Scenario: Occupied fixed blocks preserve matched timing

- **WHEN** an occupied 30-tick demonstration block contains an absent aligned input or output row
- **THEN** that row supplies an exact-zero external event while the recurrent state still advances
- **AND** only wholly unused demonstration blocks freeze state

#### Scenario: Derangement duration is shape invariant

- **WHEN** demonstration outputs of unequal heights are deranged across fixed demonstration blocks
- **THEN** the query boundary and recurrent advance schedule remain identical to the intact episode

#### Scenario: Repeated model execution is compiled

- **WHEN** context or latent events repeatedly drive the model
- **THEN** the repetition is expressed through a `brainstate.transform` loop primitive and not a bare Python `for` or `while` loop

### Requirement: Auditable public-data boundary

The example SHALL support public training sources named by the paper—ARC-AGI-1
training, RE-ARC, ConceptARC, ARC-Heavy, and ARC-GEN100K—when present. It SHALL
identify ARC-AGI-1 evaluation and fresh `arc-task-gen` tasks as evaluation-only,
shall never claim access to the paper's private data, and SHALL emit a manifest
of sources actually used.

#### Scenario: Manifest records actual data

- **WHEN** a corpus is loaded
- **THEN** the manifest records its declared name, role, version, license or reference metadata, path, file hashes, valid/rejected task counts, and deduplication outcomes

#### Scenario: Evaluation leakage aborts

- **WHEN** a canonical task fingerprint occurs in both a training or tuning split and an evaluation split
- **THEN** the run aborts before optimization and identifies the overlapping source roles and fingerprints

#### Scenario: Missing private data is not substituted

- **WHEN** only public sources are configured
- **THEN** the report states that private paper data and the private training recipe were unavailable and does not relabel generated fixtures as paper data

#### Scenario: Training augmentation preserves semantics

- **WHEN** color, dihedral, or demonstration-order augmentation is applied
- **THEN** it uses `brainstate.random`, applies the same transformation consistently to related input/output grids, and leaves evaluation tasks unchanged

### Requirement: BrainTrace recurrent spiking substrate

The full experiment SHALL use 4,096 BrainPy LIF neurons, BrainTrace input and
readout projections, exponential current synapses, and a BrainTrace sparse
recurrent projection containing exactly 4,194,304 directed edges. The report
SHALL record physical neuron and edge counts.

#### Scenario: Full configuration has the declared scale

- **WHEN** the full model is constructed
- **THEN** it contains 4,096 LIF neurons grouped as 64 analysis slots of 64 and its recurrent sparse operator contains exactly 4,194,304 edges

#### Scenario: Compact color factors expand to full ARC logits

- **WHEN** the default rank-16 readout is evaluated
- **THEN** 1,180 compact shape and CP-factor values expand deterministically to 30 height, 30 width, and 30×30×10 color logits

#### Scenario: State and parameters are distinct

- **WHEN** demonstrations and latent steps run without an optimizer update
- **THEN** synaptic current, voltage, and spikes may change but every trainable parameter remains bitwise unchanged

#### Scenario: Smoke scale is not scientific evidence

- **WHEN** a reduced model is run by tests or `--smoke`
- **THEN** its output is marked plumbing-only and cannot be reported as the full experiment or as an ARC benchmark result

### Requirement: One model with variable latent effort

One parameter set SHALL support the fixed effort set 0, 30, and 60. Training
updates that use post-query terminal supervision SHALL use effort 30 or 60. The
same frozen parameter set and the same encoded queries SHALL be evaluated at
latent checkpoints 0, 30, and 60, and effort 60 SHALL be the nominated
submission checkpoint.

#### Scenario: Mixed effort shares parameters

- **WHEN** training updates use different configured effort lengths
- **THEN** they update one model and one optimizer state, and the report records the update count at each effort

#### Scenario: Checkpoint zero precedes latent input

- **WHEN** effort 0 is evaluated
- **THEN** the output is decoded from the query-terminal state before any zero-input recurrent reasoning step

#### Scenario: Effort changes only recurrent computation

- **WHEN** the 0, 30, and 60 checkpoints are compared
- **THEN** parameters, task order, demonstrations, query events, initial state, and decoder are identical and only the number of zero-external-input recurrent steps differs

### Requirement: pp-prop terminal training

The model SHALL be compiled with BrainTrace pp-prop and trained using
`etrace_grad`. Its terminal objective SHALL include output-height,
output-width, and valid-cell color cross-entropy, and SHALL NOT obtain target
shape or cells through an input or decoding shortcut.

#### Scenario: All output factors receive supervision

- **WHEN** a training target has shape `h × w`
- **THEN** height and width losses are present and color loss covers exactly the `h × w` valid cells

#### Scenario: Supervision is terminal

- **WHEN** an effort length is sampled for a training episode
- **THEN** the ARC loss is attached to that terminal checkpoint and intermediate latent states are retained only for diagnostics

#### Scenario: No gradient-equivalence claim is made

- **WHEN** Example 21 tests and reports are inspected
- **THEN** none asserts that pp-prop equals or is within a bound of a BPTT gradient oracle

### Requirement: Exact ARC candidate scoring

The example SHALL decode at most two deterministic candidate grids per query.
Pass@1 and pass@2 SHALL require exact shape and exact cell equality. Strict task
pass@k SHALL require every query belonging to a task to pass under its first `k`
candidates. Shape and pixel metrics SHALL be labelled diagnostics only.

#### Scenario: One wrong cell fails exact success

- **WHEN** a prediction has the target shape but differs in one cell
- **THEN** its pass@1 is false while its pixel diagnostic reflects the near miss

#### Scenario: Wrong shape fails before cell comparison

- **WHEN** a candidate's dimensions differ from the target
- **THEN** exact success and shape accuracy are false and no padding outside the candidate is counted as a correct cell

#### Scenario: Pass two uses a real second candidate

- **WHEN** deterministic decoding yields two distinct candidates and only the second exactly matches
- **THEN** pass@1 is false and pass@2 is true

#### Scenario: Multi-query strictness is conjunctive

- **WHEN** one of a task's test queries fails pass@k
- **THEN** strict task pass@k is false even if every other query passes

### Requirement: Model-owned target-free candidate provenance

Every candidate included in primary exact metrics SHALL be the deterministic
result of target-free proposal generation followed by an executed network path
that consumes recurrent state or features and trained model-owned parameter
leaves restored from the nominated checkpoint. Candidate provenance SHALL
identify proposal source, ranking source, answer-head version, score components,
and participating parameter-leaf paths. Raw demonstration-only forest ordering
and rules MAY be reported as diagnostics but SHALL NOT determine a primary
candidate slot.

#### Scenario: Held-out targets cannot influence candidates

- **WHEN** held-out outputs are replaced while demonstrations, test inputs, checkpoint, and configuration remain fixed
- **THEN** canonical candidate bytes are identical
- **AND** candidate construction receives no target shape, target cell, exactness bit, task/source identity, or target-derived selector

#### Scenario: Demonstration-only forest ordering is diagnostic

- **WHEN** a forest or rule produces candidates from demonstrations and the query without consuming an executed checkpoint-owned value
- **THEN** its dependency class is `demonstration_only_diagnostic` or `rule_diagnostic`
- **AND** none of its candidates contributes to primary query, strict-task, or cumulative metrics

#### Scenario: Checkpoint likelihood can rank target-free forest proposals

- **WHEN** the target-free forest supplies bounded proposal grids for the model-owned answer path
- **THEN** every proposal is ordered by `forest_log_probability + 1.0 * trained_network_candidate_log_probability`
- **AND** the network term uses the recurrent model's factorized height, width, and cell-color likelihood under the restored checkpoint
- **AND** the coefficient and proposal policy are fixed before evaluation-label scoring

#### Scenario: Every counted candidate owns its dependence

- **WHEN** either primary candidate contributes to pass@1 or pass@2 membership
- **THEN** that candidate has `model_checkpoint` provenance and a nonempty executed parameter-dependency set
- **AND** movement by another candidate cannot qualify it

#### Scenario: Repeated answer-head execution is compiled

- **WHEN** network inference, task-local model routing, or trainable-state updates repeat
- **THEN** repetition uses a `brainstate.transform` loop primitive rather than a bare Python `for` or `while` loop
- **AND** model, topology, augmentation, reseed, and perturbation randomness uses `brainstate.random` rather than `jax.random` directly

### Requirement: Per-step latent trajectory evidence

The example SHALL retain provisional outputs and state measurements throughout
the latent rollout. It SHALL report changed output cells, predictive entropy or
margin, spike count and rate, voltage magnitude, state displacement, and a
convergence indicator at each required checkpoint and, where configured, each
intermediate latent step.

#### Scenario: A fixed point is visible

- **WHEN** consecutive spike and voltage states and decoded outputs are unchanged
- **THEN** displacement and changed-cell count are zero and the trajectory is labelled converged for that transition

#### Scenario: Saturation and silence are visible

- **WHEN** firing occupancy approaches all neurons or zero neurons
- **THEN** the report exposes the measured rate and flags the degenerate regime rather than interpreting depth scores alone

#### Scenario: Provisional outputs are ARC grids

- **WHEN** a trajectory checkpoint is decoded
- **THEN** its provisional output has independently predicted dimensions and colors and can be scored by the same exact scorer as the terminal output

### Requirement: Causal controls

Frozen evaluation SHALL include no-context, shuffled-demonstration,
truncation, and deterministic 64-neuron slot-ablation controls. Controls SHALL
preserve the query and model parameters and SHALL report both score differences
and latent-state differences from the intact trajectory.

#### Scenario: No-context removes demonstrations only

- **WHEN** the no-context control runs
- **THEN** demonstration events provide zero drive while query events, initial state, parameters, and decoder match the intact arm

#### Scenario: Shuffling breaks associations

- **WHEN** a task has at least two demonstrations
- **THEN** demonstration outputs are deranged across demonstration inputs while retaining every input and output grid exactly once

#### Scenario: Slot ablation has a precise target

- **WHEN** slot `s` is ablated
- **THEN** exactly neurons `[64s, 64(s+1))` are zeroed at the intervention boundary and the chosen slot is recorded

#### Scenario: A causally null control is stated plainly

- **WHEN** a control's latent states are byte-identical to the intact trajectory
- **THEN** the report calls the intervention causally null at measured precision even if aggregate task scores also match

### Requirement: Reproducible qualification and claim boundary

The full report SHALL contain seeds, configuration, device/backend, parameter
and edge counts, data-manifest hashes, split counts, effort-update counts,
runtime, exact metrics, diagnostics, controls, and trajectory summaries. It
SHALL describe the work as an instantiation of the public task/effort contract,
not a reproduction of proprietary internals.

The nominated baseline SHALL have a cumulative exact score of at least 16,
defined as the integer sum of query pass@1, query pass@2, strict-task pass@1,
and strict-task pass@2 counts on one complete fixed evaluation manifest. It
SHALL additionally pass matched repeat, checkpoint-scale, and same-schema
trained-checkpoint-swap and deterministic-parameter-reseed evaluations. The
three perturbations are separately mandatory and SHALL NOT substitute for one
another.

The accepted full-matrix configuration SHALL use 4,096 neurons, exactly
4,194,304 recurrent edges, 60 latent steps, retained effort checkpoints
0/30/60, submission effort 60, fixed evaluation seed 31337, and the
`checkpoint_conditioned` answer head. Baseline, repeat, scale, same-schema
trained-checkpoint swap, and deterministic reseed SHALL match this profile
apart from their declared checkpoint intervention.

#### Scenario: Qualification profile is fixed across the matrix

- **WHEN** the five-arm parameter-dependence matrix is evaluated
- **THEN** every arm records 4,096 neurons, 4,194,304 recurrent edges, 60 latent steps, checkpoints 0/30/60, submission effort 60, evaluation seed 31337, and answer head `checkpoint_conditioned`
- **AND** changing model scale, effort, evaluation seed, answer head, task order, decoder, backend, or scorer cannot be used as evidence of checkpoint dependence

#### Scenario: Same-run intact execution is reproducible

- **WHEN** the same frozen intact arm is executed twice within one process using the same data fingerprints, configuration, seed, software, and device
- **THEN** their spikes, decoded candidates, exact scores, and diagnostics are identical at every retained checkpoint and query
- **AND** voltage, feedforward synaptic current, and recurrent synaptic current are float32 with per-query neuron-axis RMS difference at most `1e-6` at every checkpoint-query pair
- **AND** an additional compact-logit feature-axis RMS check uses the same threshold
- **AND** byte identity is reported separately rather than inferred from the RMS tolerance

#### Scenario: Cumulative score reaches the accepted threshold

- **WHEN** the nominated baseline is evaluated on the complete fixed manifest
- **THEN** the report gives all four integer exact counts separately
- **AND** their sum is at least 16
- **AND** candidates from diagnostic-only sources contribute zero to every term

#### Scenario: Checkpoint scaling moves model answers

- **WHEN** a predeclared finite non-unit factor scales every floating trainable leaf on the recorded answer-dependency path under otherwise matched evaluation
- **THEN** the answer-parameter digest, canonical candidate-byte digest, exact-membership digest, and cumulative integer score all differ from baseline
- **AND** a flat score, invalid numeric output, or partial checkpoint load fails qualification

#### Scenario: Same-schema trained-checkpoint swap moves model answers

- **WHEN** an independently seeded trained checkpoint with identical ordered leaf paths, shapes, and dtypes is substituted
- **THEN** the answer-parameter digest, canonical candidate-byte digest, exact-membership digest, and cumulative integer score all differ from baseline
- **AND** architecture, input manifest, task order, effort, decoder, backend, and scorer remain matched

#### Scenario: Deterministic parameter reseed moves model answers

- **WHEN** every trainable leaf on the recorded answer-dependency path is deterministically reseeded under the exact baseline schema using `brainstate.random`
- **THEN** the answer-parameter digest, canonical candidate-byte digest, exact-membership digest, and cumulative integer score all differ from baseline
- **AND** passing the trained-checkpoint swap does not waive this separate gate

#### Scenario: Prediction movement excludes metadata

- **WHEN** candidate and exact-membership digests are computed
- **THEN** candidate bytes contain only candidate rank, dimensions, and row-major colors in manifest and submitted-rank order
- **AND** membership bytes contain ordered per-query pass@1/pass@2 and per-task strict pass@1/pass@2 booleans
- **AND** reranking an identical proposal set changes candidate bytes while provenance strings, score values, timestamps, filenames, and hashes cannot themselves satisfy movement

#### Scenario: Parameter and prediction provenance is auditable

- **WHEN** any baseline, repeat, scale, swap, or reseed arm completes
- **THEN** the artifact records source revision, dirty state, manifest digest, checkpoint/run identity, training seed and configuration, full-checkpoint SHA-256, ordered participating-leaf schema and SHA-256, topology digest, candidate digest, membership digest, all four exact counts, cumulative score, and perturbation details
- **AND** exact-schema restoration fails closed on missing, extra, reshaped, or dtype-changed leaves

#### Scenario: Optional Dale dependence is measured rather than asserted

- **WHEN** a qualifying arm claims EI/Dale dependence
- **THEN** it records the neuron-type mask digest and proves zero effective recurrent-weight sign violations
- **AND** a predeclared sign control changes candidate bytes, exact membership, and cumulative score or the Dale-dependence claim fails

#### Scenario: Slot ablation has a matched pre-intervention state

- **WHEN** the intact and slot-ablation arms are compared at checkpoint 0 before the intervention
- **THEN** they satisfy the same exact spike, decoded-candidate, exact-metric, and per-query state-RMS requirements as a repeated intact run
- **AND** only post-intervention differences are interpreted as effects of the ablation

#### Scenario: Full qualification requires full scale

- **WHEN** a result is labelled a full Example 21 qualification
- **THEN** its report proves 4,096 neurons, 4,194,304 recurrent edges, successful pp-prop compilation, a non-evaluation training source, held-out evaluation tasks, retained checkpoints 0/30/60, submission effort 60, evaluation seed 31337, and answer head `checkpoint_conditioned`

#### Scenario: Proprietary claims are excluded

- **WHEN** the generated report and repository specification are read
- **THEN** they state that the paper's private data, architecture details, and training recipe were unavailable and make no paper-score, cost, or reproduction claim
