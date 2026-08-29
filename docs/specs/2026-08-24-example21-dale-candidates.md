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
`braintrace.sparse_matmul`. Typed edges use `sign * softplus(raw)` and untyped
edges use the raw signed value. Initialization uses
`inverse_softplus(max(abs(old_effective_weight), 1e-6))`, so accepted
non-zero magnitudes and the minimum typed magnitude are preserved. The same
transform is used by training, addition, pruning, and compaction. A structural
operation must validate every outgoing edge from each typed source after the
operation and must not create optimizer moments for new coordinates.

The Dale runner receives one accepted untyped parent checkpoint, snapshots its
serialized state once, and restores an independent child from that checkpoint
for each arm. It SHALL reject a child that aliases the parent or the other arm,
and SHALL reject any mutation of the checkpoint state. Each child runs exactly
64 PP-Prop updates in the compiled loop. The runner records the same direct
strict vector before and after each arm. An arm is accepted only when at least
one strict value changes from false to true and no strict value changes from
true to false; otherwise the parent remains active. Candidate selection and arm
execution are deterministic and do not use a random ratio or random regrowth.

The production Example 21 structural path SHALL invoke this checkpoint-backed
runner. Every new recurrent coordinate SHALL derive its initialization from
`topology.dale[source]`: typed sources use
`inverse_softplus(1e-6)` and untyped sources use raw zero. A caller flag SHALL
not override the source-neuron label.

The production Dale command SHALL load the accepted parent checkpoint, require
that every parent Dale code and mechanism code is zero, and derive all
candidate measurements from that fully untyped topology. It SHALL measure the
causal effect of blocking each candidate's outgoing recurrent connections and
include that lesion evidence in both stable excitatory and inhibitory score
vectors. Both arms SHALL run through the real PP-Prop update callback and
direct fixed strict screen. A supplied checkpoint SHALL be bound to the
accepted parent by its serialized topology, parameters, optimizer arrays, and
checkpoint digest; a matching identifier alone is insufficient. The runner
SHALL reject a different checkpoint state or a partially typed parent before
measurement or arm construction.

The causal lesion SHALL execute the model with each source's outgoing
recurrent coordinates blocked, compare the resulting task output with the
unblocked output, and normalize each task row independently before ranking.
Activity and pre-clip recurrent-gradient evidence SHALL also be normalized per
task before cross-task aggregation; recurrent-edge gradient mass SHALL first be
aggregated to source neurons separately for each task. Multiplying any one task
row by a positive constant SHALL not change either Dale candidate selection.
The production command SHALL write each promoted arm as a distinct array-only
child checkpoint, including its post-update model values and optimizer state.
Rejected arms SHALL not replace the accepted parent checkpoint.

Chemical and optional biological mechanisms are deferred. Construction
defaults to no AMPA, GABAa, NMDA, HCN, calcium-dependent adaptation, electrical
junctions, multiple compartments, morphology, neuromodulation, or persistent
episodic memory. Enabling any deferred mechanism in the Dale runner raises a
corrective error.

## Focused acceptance checks

- repeated measurements have identical rankings and candidate counts;
- excitatory and inhibitory candidates have separate stable tie order;
- both candidate sets carry the same parent identifier;
- the effective typed weights never change sign after an update or structural
  operation;
- the effective typed magnitude is `softplus(raw)` and the inverse encoding
  round-trips the old magnitude with a `1e-6` minimum;
- one update followed by one structural operation preserves every typed
  outgoing sign and leaves all new optimizer moments at zero. The update and
  structural operation SHALL use the same real typed candidate topology;
  helper-only or no-op updates do not satisfy this check;
- each Dale arm has exactly 64 updates, uses an isolated child checkpoint, and
  passes the strict false-to-true/no-regression gate before promotion;
- the production structural command invokes both Dale arms from one accepted
  checkpoint and records the real PP-Prop update and strict-screen evidence;
- a nonzero mechanism code or parent Dale code is rejected on the checkpoint
  and construction paths;
- block-lesion effects are measured from the accepted parent and affect both
  candidate ranking vectors;
- activity and source-aggregated pre-clip gradient evidence are task-normalized
  before aggregation, and positive rescaling of one task row leaves both
  selections unchanged;
- a checkpoint with the accepted identifier but different serialized state is
  rejected;
- zero type signs keep raw signed behavior;
- all optional biology remains disabled by default and is rejected when set.
