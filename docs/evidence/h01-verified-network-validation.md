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

## Real construction attempt

The four-neuron/two-projection construction was started with the prepared
topology, zero soma input, and max_cv_length_um=10. It was stopped after about
eight minutes without reaching the completed-construction message. No real
network simulation was launched, and no completed build artifact was produced.
This is a construction-cost limitation, not a passed execution check. Its cause
has not been isolated. No performance or physiology tuning was started.

The 104-node graph, all three verified synapses, source cable projections,
and sign-checked matrices are complete. The real four-cell BrainCell builder
has fixture validation only; its full-scale construction remains unverified.
