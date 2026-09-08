# H01 40-cell construction continuation

2026-09-08. **Construction and initialization PASS; simulation pending.**

The repaired loader constructed 40 cells, 165,084 compartments and two
projections in 240.850 seconds (245.793 seconds process wall time, exit 0).
Both formerly failing cells in this prefix, 5965472721 and 5805562981, are
included. There are 36 isolated cells; disconnected fragments remain separate.

Evidence: [build summary](h01-ready-40-build-summary.json),
[launch and completion](h01-ready-40-build-launch.json).
The code is committed as `c47e870` on `feat/h01-braincell`.
The summary preserves per-cell provenance and settings; its raw-evidence field
records the SHA-256 and local path of the complete electrical interval arrays.

This measurement replaces the earlier failed build as the current construction
result. It does not erase that historical failure or promote any donor model.
The init-only run completed with exit 0: construction 257.549 seconds,
initialization 226.172 seconds, process wall time 493.329 seconds, and peak
resident memory 2,466.867 MiB. All 40 expected population IDs occur in the
initialization timings. Cell settings, donor assignments, contacts, compartment
counts and population IDs match the preceding construction result exactly.
Evidence: [initialization summary](h01-ready-40-init-summary.json) and
[launch/completion](h01-ready-40-init-launch.json). The previous 450-second
initialization estimate is superseded by this measurement.

Runtime: Python 3.13.14, BrainCell 0.1.0, brainstate 0.5.4, brainunit 0.5.2,
JAX/JAXlib 0.11.1, NumPy 2.5.2. The next 1 ms compiled `ei` smoke has its live
record in `h01-ready-40-ei-1ms-launch.json`, with a 1,800-second wall cap and
600-second silence cap. Its measured setup estimate is about 484 seconds;
compile plus stepping remains projected at 60 seconds until measured.

Remaining: compiled multi-step simulation, all four matched controls,
then staged 104-cell measurements sized from the observed memory and timing.
The 1 ms smoke precedes the assumed pulse onset at 2 ms and cannot qualify
spiking or delivery. Physiological and functional-inhibition gates remain open.
