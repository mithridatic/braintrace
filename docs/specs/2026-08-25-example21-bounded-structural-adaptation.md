# Example 21 bounded structural adaptation

## Baseline and success metrics

The baseline is `BrainCellArcModel`: 2,048 Hodgkin-Huxley neurons, 441 sparse
inputs, eight recurrent edges per neuron, and a 360-value voltage readout. The
fixed task order is the eight `TRAINING_TASK_IDS` followed by the four
`VALIDATION_TASK_IDS` declared by Example 21. Seed 21 initializes input edges,
seed 22 recurrent edges, and seed 23 the readout. Structural selection itself
is deterministic and consumes no random numbers.

Learning quality is the ordered vector of direct `strict_task_pass_at_1`
Booleans for all fixed tasks. Adaptation speed is the number of PP-Prop updates
to the first strict false-to-true transition. Addition arms must use exactly 64
updates even if the transition occurs earlier. Engineering metrics are complete
arm wall time (limit 300 seconds), peak process resident memory, mutated item
count, and maximum resident connection-candidate tile size (limit 65,536
pairs). The evidence artifact records Python, JAX, backend, device, platform,
seeds, starting commit, command, and all metrics. A claim is valid only when its
artifact path and SHA-256 digest are reported.

The reproducible baseline command is:

```text
python examples/pp_prop/example21_structural.py baseline --output <artifact.json>
```

The arm command adds exactly one of `neuron-prune`, `connection-prune`,
`neuron-add`, or `connection-add`. One invocation rejects multiple arms.

## Scope

This specification implements OpenSpec tasks 7.1 through 7.6 as pure sparse
topology operations. A topology stores CSR row pointers, target indices, and
one-dimensional edge values. No operation materializes a neuron-by-neuron pair
array.

## Evidence and ranking

Task rows are normalized independently by their nonzero maximum. Neuron scores
average normalized direct readout effect, source-row recurrent transmission,
and incident pre-clip PP-Prop gradient mass. Connection scores average
normalized source-spike transmission and pre-clip PP-Prop gradient mass. Final
scores are task maxima. Stable index order breaks score ties. Owners contain all
tasks tied at a neuron's positive maximum; zero-score neurons are unowned.

Structural twins have equal input-source, recurrent incoming-source, recurrent
outgoing-target, Dale-label, and active-mechanism sets. Values do not affect the
twin class.

## Bounded mutation

Each pruning or addition changes `ceil(0.05 * active_count)` items. Pruning is
blocked while every validation strict Boolean is false. Neuron pruning removes
all incident sparse edges. Connection pruning changes recurrent edges only.

Compaction remaps every surviving sparse endpoint, neuron row, Dale label, and
Adam row or edge value. It preserves surviving moments and step counts, gives
new items zero moments, and reports that eligibility must reset. A caller must
compare prediction bytes and strict Booleans before promotion.

Neuron addition selects distinct high-score donors for the first failing
training task. Selected donors may not connect to one another. Each twin copies
input and recurrent incoming edges, duplicates outgoing edges while splitting
their values with the donor, and splits the donor readout row. It inherits owner,
Dale label, and mechanism. The complete sparse edge count must remain at or
below 1,024 times the new neuron count.

Connection addition ranks sources by mean spikes and targets by incident
gradient mass plus wrong-output readout evidence. It evaluates stable 256 by
256 source-target tiles, holds at most 65,536 resident candidate pairs, excludes
self and existing pairs, and retains the global best requested set. Untyped
edges start at raw zero; typed edges start at the inverse-softplus value whose
effective magnitude is `1e-6`.

## Promotion and execution

One process evaluates one arm. Addition arms run exactly 64 PP-Prop updates.
Every arm must finish within 300 seconds. Promotion requires at least one direct
strict false-to-true change and no true-to-false change. Pruning also requires
no regression. A failed candidate leaves the parent unchanged.

The model integration boundary snapshots pre-clip gradients directly from
`learner.etrace_grad` before `clip_gradient` is called. It accumulates absolute
mass per task and sparse parameter item. Spike evidence is accumulated from
`model.previous_spikes`; direct readout evidence is measured from the model's
voltage feature and readout rows. Physical compaction rebuilds model and learner
objects, remaps surviving Adam moments, and calls `reset_episode(learner)` so no
eligibility state crosses a topology change.

Mask-versus-compaction identity is evaluated on the same episode snapshot. The
masked parent zeros removed neurons and incident edges. The compact child uses
the remapped sparse topology. Promotion requires byte-identical decoded
prediction arrays and identical strict vectors before any post-compaction
training.

## Risk register

1. **High — compilation cost:** rebuilding a learner after a shape change can
   exceed the arm budget. Measure compile and execution time separately.
2. **High — semantic drift:** incorrect sparse endpoint or Adam remapping can
   change predictions. Require byte identity and hand-calculated remap tests.
3. **Medium — invalid evidence:** post-clip gradients hide contribution mass.
   Capture gradients at the `etrace_grad` return boundary and test values larger
   than the clip norm.
4. **Medium — memory growth:** dense pair ranking would exceed memory. Tile at
   256 by 256 and report the observed maximum resident pair count.
5. **Low — unstable ties:** backend ordering can change selected items. Resolve
   every tie by source then target index and test exact selections.

## Verification

Co-located tests use hand-calculated sparse fixtures for normalization,
contribution, ownership, twins, exact ceiling counts, pruning, compaction and
Adam remapping, connected twin addition, tiled connection selection, absence of
dense pair allocation, and strict promotion rules. Integration tests must check
prediction-byte equality across accepted mask compaction and the fixed 64-update
driver.
