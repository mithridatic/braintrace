# Verified H01 network checks

Date: 2026-09-06. Branch: feat/h01-braincell.

## Export

The local population audit yields 104 nodes, 79 E/+1 and 25 I/-1, and three
verified anatomical contacts (EE, EI, IE). The exported contact-count matrix
is 104 by 104 and sums to three. Its constructible subset sums to two.
Every nonzero signed weight agrees with the sending node's Dale sign.
The node order is saved in the same NPZ, and the JSON retains original IDs.

The E-to-I contact 65017731 is on presynaptic component 55 of cell 3519995546,
0.2080843 um from its cable centerline. This component has no soma. Its
postsynaptic endpoint is on component 0 of cell 3680152874, 0.1947767 um away.
The contact remains in the anatomical graph and is explicitly excluded from
simulation. No fragment is joined to a soma and no soma emission is substituted.

## Tests

49 focused tests pass. The new h01_connectivity and h01_network modules each
have 100% measured line coverage. Checks include unknown/conflicting cell
labels, exact endpoint identity, nonzero spatial offsets, type/sign conflicts,
duplicate annotations, changed source hashes and placement, disconnected
components, multiple source sites, invalid weights and delays, and isolated
node retention. Connected and disconnected fixture networks run with compiled
time stepping. A direct fixture check observes a source event, delayed
conductance arrival, identical receiver voltage before arrival, and a voltage
change after arrival.

The first network fixture exposed an omitted population dimension: the cell
builder defaults to a scalar cell, but BrainCell Network requires pop_size=(1,).
The builder now supplies that dimension explicitly. The construction and
execution tests retain this regression check.

On this Windows environment, starting coverage before native-library imports
caused an Abseil SetTimeZone abort. Preloading braintrace and braincell before
starting pytest avoids it. This command completed with 49 passed in 37.74 s:

```powershell
.cache/validation/Scripts/python.exe -u -c "import braintrace, braincell; import pytest; raise SystemExit(pytest.main(['braintrace/datasets/h01_connectivity_test.py','braintrace/datasets/h01_network_test.py','--cov=braintrace.datasets.h01_connectivity','--cov=braintrace.datasets.h01_network','--cov-report=term-missing','-q','-s']))"
```

These checks establish connection handling and fixture delivery. They do not
establish human waveform accuracy, full-scale training, or firing at the real
H01 contacts with the current candidate profiles.

## Initial real construction attempt (superseded)

The four-neuron/two-projection construction was started with the prepared
topology, zero soma input, and max_cv_length_um=10. It was stopped after about
eight minutes without reaching the completed-construction message. No real
network simulation was launched, and no completed build artifact was produced.
This is a construction-cost limitation, not a passed execution check. Its cause
has not been isolated. No performance or physiology tuning was started.

The 104-node graph, all three verified synapses, source cable projections,
and sign-checked matrices are complete. The real four-cell BrainCell builder
had fixture validation only at that point.

## Completed real construction

The resumed construction passed on 2026-09-06. The exact source geometry,
candidate profiles, contact sites, 10 um mesh setting, and zero input were
retained. The result contains four neurons and two enabled synapses:

| Cell | Compartments |
| --- | ---: |
| 3955003482 | 33,965 |
| 4157825456 | 12,966 |
| 4188575291 | 13,989 |
| 5584343344 | 14,685 |
| Total | 75,605 |

Construction completed in 331.2011144 seconds. The
[build record](h01-verified-network-build.json) contains actual cell settings,
compartment counts, placements, assumed synapse parameters, and the blocked
fragment contact. The elapsed-time log is h01-network-construction.log.

The installed BrainCell implementation rebuilt a complete branch-index map
per edge and a complete branch tuple per compartment during channel placement.
The H01Cell subclass now caches these lookups within each synchronous
discretization call and restores the original instance methods in a finally
block. Other branch orders use the original lookup; topology-size changes
invalidate the cache. Neither geometry nor channel laws are cached or changed.
This is local package code, not an edit to the installed dependency.

Two regression tests failed before the corresponding fixes: nine index-map
builds on a three-branch fixture, and 44 ordered-branch tuple builds during
channel placement. Both now stay within the at-most-two-build bounds. Tests
also verify identical geometry, node and CV trees, and driven passive voltage
traces against the ordinary BrainCell Cell, including cloned initialization.
Nested scopes, nondefault order, topology additions, and exception cleanup
are covered. The final focused check passed 60 tests in 59.91 seconds, with
100% line coverage of h01_construction and h01_network.

Correction: the first fix covered index maps but missed ordered branch tuple
rebuilding. Keep both work-count regressions; a small functional test alone
does not expose quadratic work on large morphology trees. Test comparisons
must compare NumPy mapping arrays elementwise and place a probe before Cell.run.

This confirms full construction. Human waveform validation and Example 21
training remain separate, and the third anatomical contact still starts on a
fragment without a reconstructed connection to its soma.

## Execution check: blocked by solver memory

A second construction completed in 211.1 seconds with the same four cells,
two projections, and compartment counts. The requested 0.005 ms run at
dt=0.005 ms then failed during Network.init_state, before any simulation step.
The installed BrainCell build_cv_axial_operator constructs a dense node matrix
and dense blocks for eliminating algebraic nodes. NumPy raised MemoryError
in np.linalg.solve(algebraic_algebraic, algebraic_dynamic). The failure log is
[h01-network-execution.log](h01-network-execution.log). No trace file was produced.

This is a separate runtime memory limitation, not a construction timeout or
a failed human-response comparison. The real circuit is constructed but not
execution-qualified. Further execution work must address the dense axial
operator while preserving the electrical equations and contact placement.
The fixture parity and delivery tests do not establish large-cell execution.
