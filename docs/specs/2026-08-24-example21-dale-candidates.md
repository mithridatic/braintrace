# Example 21 measured Dale candidates

Date: 2026-08-24
Status: implementation specification

## Contract

The candidate runner receives one accepted untyped parent checkpoint. It
measures each neuron's outgoing sign coherence, activity, pre-clip eligibility
gradient mass, task ownership, and lesion evidence. It ranks excitatory and
inhibitory candidates independently with stable index tie-breaking. Both arms
therefore use the same parent and the same candidate budget. No random ratio or
random regrowth is allowed.

Accepted signs are enforced by a local sparse `weight_fn` passed directly to
`braintrace.sparse_matmul`. Typed edges use
`sign * (softplus(raw) + 1e-6)` and untyped edges use the raw signed value.
Initialization uses inverse softplus so accepted non-zero magnitudes are
preserved. The transform is used by training, addition, pruning, and
compaction.

Chemical mechanisms are deferred. Construction defaults to no AMPA, GABAa,
NMDA, neuromodulation, or extra channels. Enabling any deferred mechanism in
the Dale runner raises a corrective error.

## Focused acceptance checks

- repeated measurements have identical rankings and candidate counts;
- excitatory and inhibitory candidates have separate stable tie order;
- both candidate sets carry the same parent identifier;
- the effective typed weights never change sign after an update or structural
  operation;
- zero type signs keep raw signed behavior;
- all optional biology remains disabled by default and is rejected when set.
