## Why

The current Example 21 is too large and too slow for useful ARC iteration. Its
latent-workspace, synthetic-task, partial-score, candidate-routing, and large
diagnostic paths do not provide direct strict evidence that a recurrent brain
model solves ARC tasks.

Example 21 needs one small BrainCell Hodgkin-Huxley model that learns with
BrainTrace PP-Prop, produces one direct prediction, and reports exact observed
behavior within minutes.

## What Changes

- Replace the current Example 21 latent-workspace path with one recurrent layer
  of 2,048 BrainCell single-compartment Hodgkin-Huxley neurons.
- Start with 14,112 sparse input connections, 16,384 sparse directed recurrent
  connections, and signed trainable weights. Start with no Dale types, no E/I
  ratio, and no BrainCell chemical synaptic mechanisms.
- Use a compact lossless temporal encoding of real ARC practice tasks and one
  direct integer-grid decoder.
- Train every counted model arm with BrainTrace PP-Prop. Do not use BPTT,
  synthetic-task qualification, copy paths, rules, retrieval, forests,
  reranking, or latent-workspace modules.
- Use zero-tolerance strict task pass-at-1 as the sole routine ARC score.
- Limit the temporary proof to three minutes, each ordinary experiment to five
  minutes, the warmed decoder to 100 milliseconds per query, and the focused
  pytest selection to one minute.
- Keep each routine result at or below 256 KiB and store only direct prediction,
  target, exact query result, strict task result, and the strict task count.
  Print the selected backend and its direct timings as one separate line.
- Add observed 5% neuron, connection, and optional Dale-type stages only after
  the simple baseline works. Do not use random E/I assignment or random growth
  as the primary method.
- Defer AMPA, GABAa, NMDA, additional channels, compartments, morphology,
  neuromodulation, and persistent memory until separate measured stages.
- **BREAKING**: Remove the old Example 21 command surface, result schema, and
  latent-workspace implementation files. They are not a supported library API.

## Capabilities

### New Capabilities

- `pp-prop-braincell-arc`: Defines the direct real-ARC input, BrainCell
  recurrent model, PP-Prop training, exact decoder and scorer, bounded evidence,
  structural stages, output artifacts, documentation, and implementation gate.

### Modified Capabilities

None. No archived OpenSpec capability exists in this worktree.

## Impact

- Primary implementation:
  `examples/pp_prop/21-braincell-arc.py` and its co-located
  `examples/pp_prop/21-braincell-arc_test.py`.
- Removed implementation: the old `21-latent-reasoning-in-context.py` entry
  point and its Example 21 `latent_workspace*` production and test modules.
- Documentation: the Example 21 README entry, two implementation-truth
  documents, and the approved architecture recommendation document.
- Dependencies: pin `braincell==0.1.0` in the runtime image, development
  requirements, and applicable project extras. Keep the existing pinned
  Python, JAX, BrainState, BrainUnit, and GPU stack.
- Data: reuse the existing public ARC practice and evaluation data in the
  Example 21 image. Do not add generated tasks to model training or scoring.
- Library API: no change to the public `braintrace` package.
