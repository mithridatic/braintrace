# Example 20 fixed-point compilation

## Goal

Reduce XLA compilation time for Example 20's post-training joint pruning without
changing its candidate order, mask evaluations, acceptance decisions, recorded
results, or fail-closed compaction checks.

## Change

Compile the neuron and edge phase as named, non-inline call boundaries instead
of repeatedly expanding both phase programs into the alternating-cycle
program. The two specializations share the existing phase implementation and
perform exactly the same stable ordering, cumulative candidate tests, pass
termination, and bookkeeping as before.

No probe batching, reduction reordering, candidate batching, tolerance change,
or model arithmetic change is permitted.

## Acceptance

- Co-located fixed-point tests preserve accepted indices, pass accounting, and
  convergence behavior.
- The existing Example 20 test file passes in full.
- A GPU end-to-end replay must retain the established 20-neuron, 19-edge result
  before the optimization can be treated as qualified.
- Compilation measurements must show a useful reduction; otherwise the change
  is rejected.

## Measured result

Two cold-process runs of the fixed-point regression, with baseline and optimized
order reversed between runs, measured 5.71 s -> 5.05 s and 5.83 s -> 4.87 s.
That is a 13 to 20 percent reduction on the available CPU compile proxy. The
Example 18-20 regression set remained unchanged at 70 passing tests.
