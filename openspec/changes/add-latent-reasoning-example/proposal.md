## Why

The current Example 21 reduces the paper's ARC experiment to single-symbol
lookup and replaces the repository's established neuron/synapse stack with a
custom state machine. It therefore cannot measure the published task, exact ARC
quality, or whether additional recurrent latent computation improves the same
frozen model on the same held-out problems.

Example 21 must instead combine the paper's observable ARC contract with the
actual LIF neurons, synapses, sparse recurrence, and pp-prop training patterns
already established by Examples 18–20.

## What Changes

- Replace the symbol-permutation task with lossless standard ARC episodes:
  variable rectangular grids, colors 0–9, multiple demonstration pairs,
  multiple held-out queries, and predicted output dimensions and cells.
- Load public training sources named by the paper when available—ARC-AGI-1
  training, RE-ARC, ConceptARC, ARC-Heavy, and ARC-GEN100K—through a
  provenance-checked adapter. Private paper data is explicitly unavailable and
  is never implied to be present.
- Keep ARC-AGI-1 evaluation and fresh generated evaluation tasks out of
  training and tuning; fingerprint all tasks to detect cross-split leakage.
- Replace the hand-written binary workspace with an Example-18-style recurrent
  spiking module using BrainPy LIF neurons, exponential current synapses,
  BrainTrace dense/sparse operators, and pp-prop terminal supervision.
- Use one trained model exposed to short, medium, and long latent rollouts, then
  evaluate the same frozen parameters and byte-identical tasks at 0, 8, 16,
  and 32 recurrent reasoning steps.
- Make the full configuration 2,048 LIF neurons arranged as 32 latent slots of
  64 neurons and 16,384 sparse recurrent reasoning edges. Reduced smoke sizes
  are testing aids and cannot supply the scientific result.
- Score exact ARC pass@1, pass@2, strict whole-task accuracy, and output-shape
  correctness; retain pixel accuracy only as a labelled near-miss diagnostic.
- Record provisional outputs, changed cells, confidence, spikes, voltage, state
  movement, convergence, runtime, and spike count across latent steps, plus
  no-context, shuffled-demonstration, truncation, and state-ablation controls.
- Replace the old OpenSpec design, delta spec, repository spec, tests, report,
  and plot contract while preserving the failed prototype in Git history.
- Add no library API and make no breaking change to `braintrace/`.

## Capabilities

### New Capabilities

- `pp-prop-arc-latent-reasoning`: trains and evaluates an ARC-format recurrent
  spiking reasoner under pp-prop, including data provenance, exact scoring,
  variable latent effort, causal controls, and trajectory evidence.

### Modified Capabilities

<!-- None. This remains an additive example and does not change library APIs. -->

## Impact

- Affected implementation: `examples/pp_prop/21-latent-reasoning-in-context.py`
  and its three co-located support modules and tests.
- Affected documentation: the Example 21 README entry, the active OpenSpec
  change, and `docs/specs/2026-08-16-pp-prop-latent-reasoning.md`.
- Runtime: the full run is GPU-only by default and targets the repository's
  documented 2,048-neuron sparse regime. Smoke and focused tests use reduced
  CPU configurations without being reported as model-quality evidence.
- Data: external datasets remain outside Git. Every run emits a manifest with
  source, role, path or URL, version, license metadata, hashes, counts, and
  deduplication outcomes.
- Compatibility: existing Example 21 module paths and CLI entry point remain,
  but their obsolete lookup-task result schema is intentionally superseded.
