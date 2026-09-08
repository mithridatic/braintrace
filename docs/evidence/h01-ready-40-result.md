# H01 40-cell construction continuation

2026-09-08. **Construction PASS; initialization and simulation pending.**

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
The next init-only run is supervised under a 1,800-second wall cap and
600-second silence cap, with its live state in `h01-ready-40-init-launch.json`.
Its estimate is 240.85 seconds measured construction plus 450 seconds projected
initialization from the prior 12-cell point. No initialization duration has yet
been measured for these 40 cells.

After initialization: compiled multi-step simulation, all four matched controls,
then staged 104-cell measurements sized from the observed memory and timing.
The 1 ms smoke precedes the assumed pulse onset at 2 ms and cannot qualify
spiking or delivery. Physiological and functional-inhibition gates remain open.
