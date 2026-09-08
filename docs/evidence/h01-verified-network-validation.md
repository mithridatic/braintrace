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

55 focused tests pass. The h01_connectivity, h01_construction, and h01_network
modules each have 100% measured line coverage (327 tests pass across the entire
datasets suite). Checks include unknown/conflicting cell labels, exact endpoint
identity, nonzero spatial offsets, type/sign conflicts, duplicate annotations,
changed source hashes and placement, disconnected components, multiple source
sites, invalid weights and delays, isolated node retention, 1D sparse DHS static
source bitwise parity, deferred axial operator on-demand computation, and
compiled time stepping. Connected and disconnected fixture networks run with
compiled time stepping. A direct fixture check observes a source event, delayed
conductance arrival, identical receiver voltage before arrival, and a voltage
change after arrival.

On this Windows environment, starting coverage before native-library imports
caused an Abseil SetTimeZone abort. Preloading braintrace and braincell before
starting pytest avoids it. This command completed with 55 passed in 21.86 s:

```powershell
.cache/validation/Scripts/python.exe -u -c "import braintrace, braincell; import pytest; raise SystemExit(pytest.main(['braintrace/datasets/h01_connectivity_test.py','braintrace/datasets/h01_construction_test.py','braintrace/datasets/h01_network_test.py','--cov=braintrace.datasets.h01_connectivity','--cov=braintrace.datasets.h01_construction','--cov=braintrace.datasets.h01_network','--cov-report=term-missing','-q','-p','no:cacheprovider','-s']))"
```

These checks establish connection handling and fixture delivery. They do not
establish human waveform accuracy, full-scale training, or firing at the real
H01 contacts with the current candidate profiles.

## Completed real construction

The real network contains four neurons and two enabled synapses:

| Cell | Compartments |
| --- | ---: |
| 3955003482 | 33,965 |
| 4157825456 | 12,966 |
| 4188575291 | 13,989 |
| 5584343344 | 14,685 |
| Total | 75,605 |

Construction completed in 163.3 seconds (speedup from 331.2s through single-pass
output site determination and indexed morphology caching). The
[build record](h01-verified-network-build.json) contains actual cell settings,
compartment counts, placements, assumed synapse parameters, and the blocked
fragment contact.

## Solver memory bottleneck elimination and real simulation execution

Previous runs failed during `Network.init_state` because the installed BrainCell
`build_cv_axial_operator` allocated dense $(N_{\text{point}} \times N_{\text{point}})$
matrices and attempted dense algebraic node elimination via
`np.linalg.solve(algebraic_algebraic, algebraic_dynamic)`, demanding $>40$ GB of
memory and raising `MemoryError`.

### Resolution

1. **$O(N)$ 1D Sparse DHS Static Source**: `build_dhs_static_source_1d` directly
   assembles 1D diagonal, lower, and upper factors on the tree in $O(N)$ memory
   and time, matching the dense extraction bitwise while reducing peak memory
   from $>40$ GB to **~214 MB** (>99.5% reduction).
2. **Deferred Dense Axial Reduction**: In `H01Cell`, `axial_operator_np` is deferred
   during `init_state` and evaluated only on demand when `compute_axial_derivative`
   or `_get_axial_operator` is explicitly called.
3. **Compiled Scanned DHS Solver**: The network defaults to `h01_staggered_scan`,
   which compiles dendritic tree elimination as a single JAX scan primitive,
   avoiding massive unrolled XLA computational graphs.
4. **Single-Pass Discretization**: Determining output sites directly from branch
   intervals eliminates redundant full-morphology discretization passes during
   cell registration.

### Verified execution result

The real circuit of 4 neurons, 2 projections, and 75,605 compartments initializes
and runs with compiled time-stepping without memory errors. The verified execution
result is a single 0.005 ms step (one sample per probe in the traces;
`duration_ms` 0.005, `execution` "finite compiled smoke run" in
[h01-verified-network-build.json](h01-verified-network-build.json)). Later runs of
the same network are the current cost record: 10/200/2000 steps completed on
2026-09-07 (d1: construction 266.2 s, init+run 212.4 s;
[h01-network-throughput.json](h01-network-throughput.json)) and the pinned
post-merge rerun (construction 152.3 s, init_state 116.0 s, compile+200 steps
17.5 s, wall 295.0 s) held at 2e-9 mV
([h01-network-equivalence-post-merge-pinned.json](h01-network-equivalence-post-merge-pinned.json)).
The figures below are the original one-step record:

- **Construction time**: 163.3 seconds
- **Simulation initialization, JIT compilation, and execution**: 157.0 seconds
- **Trace array output**: saved to [h01-verified-network-traces.npz](h01-verified-network-traces.npz)
- **Voltages and conductances**: finite across all 75,605 compartments and probes.
