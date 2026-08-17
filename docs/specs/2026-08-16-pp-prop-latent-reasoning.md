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

Repeated same-process, same-device evaluation is qualified per checkpoint and
per query. Spikes, decoded candidates, exact scores, and their scoring
diagnostics must be identical. The normative physical-state gate requires
float32 voltage, feedforward synaptic current, and recurrent synaptic current,
each with RMS across neurons at most `1e-6` for every checkpoint-query pair; a
batch-wide average is not sufficient. An additional decoder-state check applies
the same threshold across compact-logit features. This fixed threshold never
relaxes ARC scoring. Literal byte identity is reported separately. Slot
ablation must pass the same state, candidate, and metric checks at checkpoint 0
before later differences can be attributed to the intervention.

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
- repeated intact evaluation and the slot-ablation pre-intervention checkpoint
  pass the declared per-query reproducibility gates;
- no instability or missing-data condition is silently ignored.

There is no required accuracy threshold. A zero exact score, worsening with
effort, saturation, silence, or a causally null memory control is a valid
negative result when the gates above are met and the report states it plainly.

## Empirical qualification on 2026-08-16

The retained implementation revision is
`5934e5cfb2623d31250f60f86d6bafa7283d10cc`, with reported Example-21 source
digest `8ee80cca42c4df248a4e1430de5a57f4122eb9a2d506658967910294036fdda9`.
Both retained runs used an NVIDIA GeForce RTX 3080 Ti Laptop GPU and instantiated
2,048 LIF neurons, 16,384 recurrent sparse edges, 32 slots of 64 neurons, and
2,130,716 scalar parameters.

The structural-only artifact in `var/example21-full-structural-final-v5`
completed in 8.609 seconds and passed the full structural gate while correctly
remaining non-scientific. Its `result.json` SHA-256 is
`1e750fa18d2c29bbab0c5ec9662f07a33bc43a9e9cd10217fa91816b9e937478`.

The scientific artifact in `var/example21-full-scientific-final-v5` completed
in 165.160 seconds and passed both full gates. It loaded 399 ARC-AGI-1 training
tasks after explicitly excluding the byte-identical training/evaluation overlap,
then evaluated all 400 evaluation tasks and 419 test queries. Training performed
96 batch-one terminal pp-prop updates, balanced 32/32/32 across 8/16/32 effort,
and sampled 82 unique base tasks and 83 unique task/query pairs from the
399-task pool. All six parameter groups moved; feedforward and recurrent
synaptic weights used eligibility routes, while the four readout/head groups
received exact current-window reverse-mode gradients.

Exact query pass@1, query pass@2, strict task pass@1, and strict task pass@2 were
all zero at effort 0, 8, 16, and 32. Shape diagnostics were
`0.0883/0.0907/0.0883/0.0859`, and valid-cell pixel diagnostics were
`0.0650/0.0665/0.0660/0.0628`. The latent trajectory remained active rather
than silent or saturated: mean firing fell from `0.243874` at checkpoint 0 to
`0.142861` at checkpoint 32, while changed-cell fraction reached `0.869537`.
No-context, shuffled-demonstration, and slot-ablation controls changed measured
step-32 latent state for all 419 matched queries, but none changed exact success
from zero. This demonstrates latent computation and causal trajectory
sensitivity, not successful latent reasoning.

The scientific rerun was launched with
`XLA_FLAGS=--xla_gpu_deterministic_ops=true` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Repeat-intact evaluation preserved exact
spikes, compact logits, decoded candidates, and scores for all 419 queries; its
maximum per-query RMS was `3.739e-7` for voltage and `2.655e-7` for recurrent
current. Slot ablation matched checkpoint 0 for all queries, with corresponding
maxima `3.562e-7` and `2.433e-7`. Literal physical-state byte identity remained
false and is reported separately: these settings did not make the default
recurrent sparse backend byte deterministic, so the qualified claim is only the
declared `1e-6` bound. An earlier unprotected diagnostic run,
`var/example21-full-scientific-final-v4`, correctly failed qualification when
two repeat queries and one ablation checkpoint exceeded `1e-6`; the worst
voltage/recurrent RMS values were `8.212e-4/9.066e-4`. That failure was retained
as GPU sparse-accumulation reproducibility evidence and the tolerance was not
relaxed.

The retained scientific hashes are:

| Artifact | SHA-256 |
| --- | --- |
| `result.json` | `1344e32e64c502d9321e196efe7d0475f141c874f1c1651950a5ae55e479eaea` |
| `report.txt` | `a2ffea84eae782edf0615e4749a887302d400800fe743acf279b88cdc10439f4` |
| `latent_reasoning.png` | `3116e2f07e88081bc3a436bd855a3d563c8ac3c1d2692c1b998a3ec830f2b8ce` |
| `data_manifest.json` | `cb5340ad542d4587be8468f56524fb1e8262ad2c20c507a7a7c28419bf7d130a` |
| `process.log` | `56c6958e5b3451d107e85dfa0a30ef2adbd287ef135b17bb048feff126949f99` |
| `run-environment.txt` | `c34c21f2768b59c734947edcf7daf7c998e3d79836a53d15d181d91dc404bf84` |

The final focused gate passed 270 tests with 93 percent aggregate branch-aware
coverage across the four production modules (91/95/95/93 percent
individually). The normal repository example gate passed 676 tests with five
skips and 19 retained compiler warnings. Strict OpenSpec validation, scoped
Ruff check and format, and `git diff --check` passed. All downloaded data and
run artifacts are outside tracked source or ignored under `var/`; `main`
remained at `470fd66a13e140969300acb9539cbc108a1e2891`.

This was a full-scale protocol qualification, not a converged ARC training
campaign: 96 sampled updates are insufficient to falsify the architecture, and
the run makes no claim about the paper's private model or LOW/MEDIUM/HIGH
iteration counts.

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
- same-seed reset/evaluation reproducibility and parameter immutability;
- one excessive-noise query hidden inside an otherwise stable batch, decoded
  candidate mismatch, exact-metric mismatch, and matched ablation checkpoint.
