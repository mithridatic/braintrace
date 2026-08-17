# Example 21: ARC latent reasoning with pp-prop

Date: 2026-08-16
Status: implementation specification

## Research question

On standard ARC-format tasks, does giving the same frozen recurrent spiking
network more zero-input recurrent computation improve exact predicted grids,
and what changes in its voltage, spikes, and provisional answers while it
computes?

The experiment combines the observable contract of *BDH-CQ: In-Context
Learning with Recurrent Latent Reasoning* (arXiv:2608.09888) with this
repository's established substrate from Examples 18–20. The paper supplies the
task format, evaluation convention, and variable latent-effort idea. BrainPy
LIF neurons, synapses, sparse recurrence, BrainTrace operators, and pp-prop come
from this repository. The paper's private model dimensions, update rules,
training recipe, and private data are unavailable. This is not a reproduction,
and it makes no claim about the paper's score or inference cost.

The paper publishes qualitative LOW/MEDIUM/HIGH effort conditions but does not
disclose their iteration counts. This repository's 8/16/32 recurrent ticks are
an explicit operational proxy chosen for Example 21; they are not a mapping to
the paper's undisclosed effort tiers.

## ARC contract

An ARC task consists of:

- one or more `train` pairs, each with an input and output grid;
- one or more `test` entries, each with an input grid and, for scored data, an
  output grid;
- rectangular grids between 1×1 and 30×30;
- integer colors 0 through 9.

Input and output dimensions may differ. Demonstrations within one task may also
have different dimensions. Every test query is executed as a separate episode
with the task's complete demonstration set, then re-associated with its task for
strict scoring. The implementation must not reduce a task to a symbol, class,
single cell, fixed shape, or one demonstration.

### Exact quality metrics

The decoder predicts a height in 1–30, a width in 1–30, and a color distribution
for each of 30×30 positions. Candidate one uses the joint argmax. Candidate two
changes exactly one lowest-margin decision to its runner-up among height, width,
or a cell inside candidate one's shape; duplicates are removed.

- Query pass@1: candidate one has the exact target shape and every cell matches.
- Query pass@2: either of the first two candidates exactly matches.
- Strict task pass@k: every test query of that ARC task passes at `k`.
- Shape accuracy: diagnostic only.
- Valid-cell pixel accuracy: diagnostic only and never an ARC success count.

Wrong shape is an exact failure. Padding does not earn pixel credit. A one-cell
error is an exact failure even when pixel accuracy is high.

## Data contract and training evidence

### Approved public roles

When locally available and accompanied by a source manifest, training may use
the public sources named by the paper:

- ARC-AGI-1 training;
- RE-ARC;
- ConceptARC;
- ARC-Heavy;
- ARC-GEN100K.

ARC-AGI-1 evaluation is evaluation-only. Fresh tasks from `arc-task-gen` may be
used as a second evaluation-only source. The paper's private tasks are not
available and must never be implied, synthesized, or relabelled as present.

Each source declaration includes name, role, version, local path, license or
reference metadata, and format. The emitted manifest adds file hashes, task
counts, rejection reasons, canonical content fingerprints, and duplicate
counts. Derived-source licensing remains the operator's responsibility; a
source with missing declared license/reference metadata is excluded by default.

### Leakage boundary

The canonical fingerprint serializes normalized train and test content, not the
file name or source metadata. Before optimization, all fingerprints are compared
across training, tuning, and evaluation roles. Any training/tuning overlap with
evaluation aborts and reports both roles. ARC evaluation data is not used for
hyperparameter selection, early stopping, prompt/template selection, threshold
tuning, or decoder design.

The code can reserve a deterministic validation partition from training-role
sources. Evaluation data is decoded only after model and configuration freeze.

### Training augmentation

Training-only augmentation supports:

- a bijective permutation of colors 1–9 while retaining background color 0;
- all eight dihedral rotations/reflections, consistently applied to every grid
  in the task;
- demonstration-order permutation.

Random choices use `brainstate.random`. The same color and geometric transform
is applied to related input and output grids. The original `ArcTask` is
immutable, and evaluation adapters never call augmentation.

### Embedded fixtures

Tiny hand-authored ARC tasks may be committed solely for unit and smoke tests.
Their source role is `fixture`; reports label their model-quality metrics
`plumbing_only=true`. Fixture results cannot qualify the experiment.

## Lossless row-event representation

Every grid is serialized as row events rather than flattened into one enormous
dense vector. The default standard-ARC layout reserves ten 30-row
demonstration blocks and one 30-row query block: 330 events of width 830. An
event contains:

- a valid flag and phase (`demonstration`, `query`, or padding);
- demonstration index and input/output side flags;
- normalized row index, height, and width plus one-hot/bucketed equivalents;
- a 30-cell validity mask;
- position-specific one-hot colors for up to 30 input columns and 30 output
  columns.

A demonstration row event can contain one aligned row from its input grid and
one from its output grid. When their heights differ, side-specific row-valid
flags distinguish the missing side. A query event contains only its input side.
All invalid rows are exactly zero and provide zero external drive. Unused
demonstration blocks are frozen. During a matched intervention, an occupied
demonstration's complete fixed 30-tick block advances recurrent state even
where one aligned row is absent; this keeps time identical when unequal-height
outputs are deranged. The external event on those clock-only ticks remains the
exact zero vector.

The representation is tested by lossless round trip. Changing only a held-out
test output must not change any model input byte. Task IDs, corpus IDs, transform
names, and test outputs never appear in model features.

Host-side loops may validate JSON and construct static arrays. Repeated neuron
and synapse execution must use `brainstate.transform.for_loop` or `scan`.

## Network architecture

### Full scientific configuration

| Component | Full value |
| --- | ---: |
| LIF neurons | 2,048 |
| Analysis slots | 32 |
| Neurons per slot | 64 |
| Sparse recurrent edges | 16,384 |
| Mean recurrent out degree | 8 |
| Maximum latent steps | 32 |
| Evaluated checkpoints | 0, 8, 16, 32 |
| Training terminal efforts | 8, 16, 32 |
| Maximum demonstrations | 10 |
| Fixed context events | 330 |
| Row-event input width | 830 |
| Maximum grid axis | 30 |
| Colors | 10 |

The full model follows Example 18's construction:

1. A BrainTrace `Linear` input projection drives an `Expon` synapse and `CUBA`
   current into `brainpy.state.LIF` neurons.
2. An `AlignPostProj` recurrent path uses a BrainTrace `SparseLinear`, another
   `Expon`/`CUBA` synapse, and exactly 16,384 nonzero directed edges.
3. A BrainTrace projection maps the 2,048-neuron state through a configurable
   bottleneck into separate height, width, and compact CP-factor color heads.

The sparse topology is drawn with `brainstate.random`, excludes self-edges by
default, is deterministic under its reported seed, and is checked after
construction for the exact nonzero count. The report records actual counts,
not requested counts alone.

The input synaptic state holds demonstration/query context on a slower time
scale. LIF voltage and spikes are the recurrent workspace. The report preserves
all three representations separately. A model call never writes a parameter;
only optimizer application after `etrace_grad` does.

### Static sequence and latent rollout

At episode start all model state is reset. A two-argument model call receives
the external row event and an explicit state-advance gate. This separates
invalid capacity padding, which freezes voltage and both exponential currents,
from latent ticks, whose external event is exactly zero while recurrence still
advances. The fixed context and latent schedule runs through a compiled loop.
Checkpoint 0 is snapshotted after the last query row; the following 32 recurrent
updates record spikes and voltage at every step.

No reset occurs between checkpoints. Outputs at 8, 16, and 32 therefore lie on
one continuous trajectory. Decoding a checkpoint does not feed its grid or
logits back into the network.

### Readout

The output head returns:

- 30 height logits for sizes 1–30;
- 30 width logits for sizes 1–30;
- 30×30×10 color logits.

The stored compact head contains 60 shape logits plus rank-specific row,
column, and color factors. Full color logits are expanded as a CP tensor only
for loss or analysis. At the default rank 16 this is 1,180 compact values
instead of a dense 9,000-value head. This is an explicit repository design
choice and a stricter representational bottleneck than an unrestricted ARC
color head; it is not attributed to the paper.

Training loss is height cross-entropy plus width cross-entropy plus the mean
cross-entropy over cells valid under the target shape. Output-cell padding is
masked from the loss. The target shape is never used to form a prediction.

## One-model mixed-effort pp-prop training

The model is compiled through `braintrace.compile` with
`braintrace.pp_prop(...)`; gradients come from `etrace_grad`. Training cycles or
samples terminal effort from 8, 16, and 32 using `brainstate.random`. These are
three compiled rollout lengths over the same model states and parameter objects,
not three independently initialized models. One optimizer state persists across
all updates. The report records update counts for each effort.

After training and training-source validation, parameters freeze. Each held-out
query is executed once through the full 32-step path, then the identical
snapshots are decoded at 0, 8, 16, and 32. All effort comparisons therefore use
the same task, event tensor, initial state, parameter bytes, decoder, and latent
trajectory prefix.

This example makes no claim that pp-prop matches BPTT. If a future change tests
a learning-rule property, it must follow the repository's finite-window oracle
rule rather than a vacuous whole-sequence VJP.

## Latent reasoning measurements

For each query and each latent step or required checkpoint, retain:

- provisional candidate grid(s) and exact/diagnostic scores when a target exists;
- number and fraction of candidate-one cells changed since the prior state;
- mean predictive entropy and top-two logit margin;
- spike count, firing occupancy, and a bounded raster sample;
- voltage mean, standard deviation, and norm;
- spike Hamming displacement and voltage L2 displacement;
- convergence status and flags for near-silence or near-saturation.

Aggregate metrics include distributions, not just means, so a universal
attractor cannot be hidden by average accuracy. Pairwise state hashes or
distances across distinct episodes are reported at the required checkpoints.

### Frozen causal controls

- **No context:** mask all demonstration events; preserve query, reset state,
  parameters, and decoder.
- **Shuffled demonstrations:** for tasks with at least two demonstrations,
  derange output grids across input grids. Every grid is retained exactly once.
- **Truncation:** decode the intact continuous trajectory at 0/8/16/32.
- **Slot ablation:** at a recorded boundary, zero the voltage/spike slice
  `[64s, 64(s+1))` for deterministic slot `s`, then continue with unchanged
  parameters and input.

Each control reports exact metrics and state distance from its matched intact
episode. If a perturbation yields byte-identical latent states, it is labelled
causally null at measured precision even if score equality alone would be
ambiguous.

## Files and public APIs

- `latent_workspace_task.py`: immutable ARC types, loaders, source manifests,
  fingerprints, leakage checks, augmentation, row-event encoding, smoke fixture.
- `latent_workspace_analysis.py`: logits/candidate types, deterministic decoder,
  exact metrics, trajectory metrics, control comparisons.
- `latent_workspace_model.py`: configuration, sparse topology, LIF/synapse
  network, compiled rollouts, snapshots/ablation, pp-prop loss plumbing.
- `21-latent-reasoning-in-context.py`: CLI, training/evaluation orchestration,
  controls, JSON/text report, Agg plot.

Every public class/function receives a NumPy-style docstring. Tests are sibling
`*_test.py` modules. There is no `tests/` directory and no `test_*.py` file.

## CLI and outputs

The entry point supports a reduced `--smoke` mode and a full mode taking source
manifest(s), output directory, seeds, optimizer/training counts, and device. Full
mode defaults to GPU and fails clearly when the requested backend is absent.

Outputs are:

- `result.json`, containing the complete structured evidence;
- `report.txt`, a plain-language interpretation with claim boundaries;
- `latent_reasoning.png`, using the noninteractive Agg backend;
- `data_manifest.json`, the resolved source and split evidence.

Reports, plots, checkpoints, and downloaded/generated datasets are run artifacts
and are not committed.

## Acceptance gates

### Source and test gates

- Strict OpenSpec validation passes.
- Co-located focused tests pass with more than 90 percent meaningful line
  coverage across changed production modules.
- The repository's normal example test gate passes.
- `ruff`/format/static checks used by the affected example pass.
- `main` remains unchanged; work is committed on
  `feat/example21-latent-reasoning`.

### Full structural qualification

A full structural run must prove from instantiated objects that it has exactly
2,048 LIF neurons and exactly 16,384 recurrent sparse edges, that pp-prop
compiles the event-plus-advance model, that context and latent repetitions use
BrainState transform primitives, and that a forward 0/8/16/32 trajectory
completes on the requested GPU.

### Full scientific qualification

A result may be labelled a full scientific run only when:

- at least one approved non-evaluation public training source is present;
- every training/evaluation overlap check is clean;
- held-out evaluation targets are present and were not used for tuning;
- one shared model received mixed 8/16/32 effort updates;
- the same frozen trajectories were scored at 0/8/16/32;
- every exact metric, diagnostic, trajectory measure, and causal control is
  present;
- no instability or missing-data condition is silently ignored.

There is no required accuracy threshold. A zero exact score, worsening with
effort, saturation, silence, or a causally null memory control is a valid
negative result when the gates above are met and the report states it plainly.

## Edge cases required in tests

- 1×1 and 30×30 grids, unequal demonstration dimensions, color 0 and color 9;
- ragged, empty, oversized, noninteger, boolean, and out-of-range cells;
- one and multiple demonstrations; one and multiple test queries;
- missing test outputs for inference versus required outputs for scoring;
- renamed duplicate tasks and train/evaluation fingerprint overlap;
- padding capacity exactly full and one demonstration beyond capacity;
- wrong predicted height/width and a single wrong cell;
- candidate-two duplicate and candidate-two-only exact success;
- fixed, changing, saturated, and silent latent trajectories;
- no-context equivalence, valid derangement, impossible one-demo derangement;
- first and last slot ablation plus invalid slot indices;
- deterministic topology and exact edge count at smoke and full sizes;
- same-seed reset/evaluation reproducibility and parameter immutability.
