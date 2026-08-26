# Example 21 bounded structural adaptation

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

## Verification

Co-located tests use hand-calculated sparse fixtures for normalization,
contribution, ownership, twins, exact ceiling counts, pruning, compaction and
Adam remapping, connected twin addition, tiled connection selection, absence of
dense pair allocation, and strict promotion rules. Integration tests must check
prediction-byte equality across accepted mask compaction and the fixed 64-update
driver.
