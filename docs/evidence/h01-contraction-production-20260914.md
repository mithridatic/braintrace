# Production H01 contraction qualification

The complete default 104-cell evolution in 30 seconds remains **not achieved**.
The exact command was attempted on Vast with the new implementation and an
external `timeout -k 5 30` cap; it exited 124 without completing. Its log was
empty, so no internal stage is inferred. A separate 104-cell manifest updated
only implementation hashes; all existing numerical settings were asserted
equal. The original manifest and the other session's running process were
preserved.

## Change

Static NumPy tree topologies now use parallel scalar Schur contraction inside
the existing implicit solve and transpose callbacks. Consecutive stages are
grouped by width, preserving elimination order and reducing padding. Retained
rows support reverse substitution without a stacked whole-system history.
Batching uses JAX vmap; unsupported input layouts retain the previous solver,
and `use_gpu=False` retains the reference scan path for qualification.

Tree validation rejects duplicate/missing children, invalid indices, cycles,
disconnected components and non-integral topology. The independent sentinel
row is solved explicitly. The static schedule cache checks content, shape and
dtype, owns edge arrays weakly, and retains at most 32 MiB of schedule payload
and 128 entries. Oversized schedules are usable without cache retention.

Neither dt, float64 precision, spatial discretization, training updates, nor
the linear system changes. New runs need manifests pinned to this implementation;
existing checkpoint continuation remains pinned to its original code.

## Measured real execution

RTX 4090 on Vast, Python 3.14.7/JAX 0.11.0. Four real anatomical cells, 75,605
compartments, dt=0.000625 ms and 160 substeps per event. cProfile and synchronized
phase timing; the bounded production probe completed two finite synthetic
one-event Muon updates. It used the default integrated path, without experimental
callback replacement.

| Measurement | Prior production GPU probe | Integrated contraction |
| --- | ---: | ---: |
| Cold event | 14.456 s | 14.189 s |
| Warm event | 4.479 s | 0.718 s |
| Sparse graph construction | 3.356 s | 4.187 s |
| First compile/update | 107.653 s | 97.944 s |
| Warm one-event update | 24.862 s | 4.649 s |
| Peak host RSS | 4,728,436 KiB | 4,751,940 KiB |

Warm event and learning speeds improved 6.2x and 5.3x respectively. Host peak
RSS is 0.5% higher in these samples, not lower. Activation retains the substantial
runtime gain while leaving the RAM-reduction objective open. The unrolled
prototype's peak was 5,579,628 KiB. These are small samples on a shared GPU,
not isolated statistical estimates. The earlier grouped-prototype run overlapped
tests; the integrated learning probe did not overlap additional tests.

Maximum differences versus the preceding production probe were 1.836e-10 mV
in recorded voltages, 1.986e-12 in synthetic losses, and 2.064e-13 in gradient
norms. These are not full ARC episodes or full ARC quality evidence.
Construction and initialization still took 32.960 and 21.749 seconds for four
cells, so even startup remains above the full-command objective.

## Validation

- Six-file integrated gate: 67 passed in 78.18 s with `pytest -n 2` and coverage.
  Covers dense/implicit forward and reverse derivatives, RHS batching/vmap,
  input non-mutation, sentinel behavior, calcium/network dynamics, ARC execution,
  and branched-cell parameter/optimizer/eligibility/voltage learning parity.
- Final topology/cache gate: 16 passed in 6.70 s with `pytest -n 2`; new module
  statement coverage is 128/128 (100%). Validation now runs before cache lookup,
  so a cached integer node count cannot accidentally admit an equal float.
- Local Windows CPU JAX 0.11.1: compiled production contraction matches a dense
  solve, including nonzero sentinel RHS and irrelevant sentinel edge coefficients.
- Grouped opt-in numerical/physical gate: 15 passed in 20.54 s with xdist.

Text and JSON profiles are synced in the adjacent learning folders. Raw `.prof`
files remain under `/tmp/h01-contraction-production-learning` on Vast. The next
work must address the remaining full-workload cost; neither these tests nor
the successful small probe establish completion of the 104-cell evolution.
