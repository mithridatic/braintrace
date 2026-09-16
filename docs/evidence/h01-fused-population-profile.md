# H01 fused population: before/after on the 17 kept cells (2026-09-16)

Spec: [2026-09-16-h01-fused-population.md](../specs/2026-09-16-h01-fused-population.md).
Question: what does one forest population (`build_network(fused=True)`, one
`H01ForestCell` over all 17 cells, one kernel per mechanism, one axial solve) change in
the forward ARC event, the compile, the per-mutation recompile and the learner, against
the per-cell populations of the same commit, on the same manifest
(`docs/evidence/h01-keep-drop/arc-manifest/manifest.json`, 17 cells, 107,537 compartments,
0 contacts). Script: [h01_evolve_event_profile.py](h01_evolve_event_profile.py) `arm`
with and without `--fused` (branch `campaign/h01-fused-population-20260916`). Receipts:
[h01-fused-population-profile/](h01-fused-population-profile/); JSON:
[h01-fused-population-profile.json](h01-fused-population-profile.json).

Executor: Vast 50616476 (RTX 4090), `/workspace/braintrace-fused`, `/workspace/venv314`,
`taskset -c 160-223`, persistent XLA cache. **Every arm here ran while two keep-test chains
(`transfer-all-*-ramp`) and the dt spike-window arm shared the GPU** (utilisation 99-100 %
before any arm started). Wall seconds are therefore contended and are compared only within
this session; the uncontended per-cell numbers of the profile campaign
([h01-evolve-event-profile.md](h01-evolve-event-profile.md)) are quoted separately.
GPU-busy and launch counts per substep come from the XLA profiler and are less sensitive to
contention, but the per-cell busy times below are inflated by it too (25.4 ms contended
versus 8.3 ms uncontended at dt 0.005): read the fused busy times as upper bounds.

## Result in one line

The forest runs the 17 cells as 8 while loops and 327 stream launches per cable substep
instead of 136 and 5,104, with 2.9-4.7 ms of GPU time per substep instead of 8.3 ms
(uncontended per-cell) or 19.7-25.4 ms (contended per-cell); the forward event costs
0.129 s at dt 0.005 and 1.018 s at the pinned dt under contention (per-cell uncontended:
0.180 s and 1.357 s; per-cell under the same contention: 0.874 s and 6.89 s). Voltages agree
with the per-cell path to 7e-10 mV over 21 events at both timesteps. The forward compile
falls from 27-42 s to 5-6 s, a clone's compile from 40 s to 5.4 s, the learner's graph
analysis from 7.6-9.9 s to 0.6 s and its 2-event update compile from 314-348 s to 36 s.
What remains: 90-94 % of the fused substep is still unattributed command-buffer time
(236 launches: the per-mechanism channel kernels), the fused path delivers no contacts yet,
and the sparse pp-prop eligibility layout grows 17x (455 MB, 17 colours) on the forest.

## 1. Forward path, same session (contended)

| Arm | s/event (median of 20) | build + init s | forward compile + 1 run s | GPU busy ms/substep | launches/substep | while loops (substep HLO) | peak device MiB | peak host RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| per-cell, dt 0.005 / 20 | 0.874 | 20.3 + 24.9 = 45.2 | 27.5 | 25.37 | 5,104 | 136 | 63 | 4.6 |
| fused, dt 0.005 / 20 | 0.129 | 48.3 + 5.5 = 53.9 | 5.4 | 4.69 | 327 | 8 | 90 | 4.1 |
| per-cell, dt 0.000625 / 160 | 6.891 | 20.4 + 25.2 = 45.6 | 33.8 | 19.72 | 5,104 | 136 | 63 | 5.4 |
| fused, dt 0.000625 / 160 | 1.018 | 48.1 + 5.4 = 53.5 | 6.0 | 2.95 | 327 | 8 | 89 | 4.0 |

Uncontended per-cell reference (profile campaign, same manifest, same code path): 0.180 s
and 1.357 s per event, 8.34 ms busy per substep at dt 0.005, 5,104 launches, forward
compile 42.2 s (cold) / 16.8 s (cache hit). Against that reference the fused forward event
is 1.4x (dt 0.005) and 1.33x (pinned) faster while running under contention; against the
per-cell path under the same contention it is 6.8x. The uncontended fused number is not
measured here: it is queued for when the chains finish (the fused wall per substep, 6.4 ms
at both dt, is 1.4-2.2x its GPU-busy time, so part of the gap is waiting on the shared
device).

Build + init: on the fused path `build_network` initializes the 17 source cells (that is
the 48 s) and the forest's own `init_state` is 5.5 s (lowering 107,537 CVs once plus
copying parameters and states). The total is 54 s against 45 s: the source cells are
still built and initialized one at a time; fusing costs 8-9 s on top.

Per-substep GPU classes, fused (dt 0.005 / pinned): `other` (command buffers) 4.40 / 2.66 ms
with 236 launches; loop control 0.13 ms, 26 launches; axial solve 0.075 ms, 42 launches;
implicit calcium 0.065 ms, 15 launches; cell runtime 0.008 ms, 2 launches; memcpy 6
launches. Per-cell (contended, dt 0.005): `other` 21.7 ms / 3,710 launches; axial 2.75 ms /
701; loop control 0.47 ms / 303; calcium 0.35 ms / 255; memcpy 118; cell runtime 17. The
axial solve went from 701 launches to 42 (one forest contraction, `h01_dhs_forest`), the
calcium solve from 255 to 15, and the channel kernels from 17 sets to one; what is left is
one kernel per mechanism per gate over 107,537 points, which the profiler still cannot
attribute past the command buffers (the `pinned-nocb` arm remains unrun).

## 2. Parity

Soma voltages of the fused arm against the per-cell arm at the same dt, same 21 ARC events
(`voltages_mv.npy` of each arm): max |dV| 1.7e-10 mV at dt 0.005 (per-cell maxima
6.6e-12 to 1.7e-10) and 7.0e-10 mV at the pinned dt (2.0e-12 to 7.0e-10); all voltages
finite. The unit test (`h01_forest_cell_test.py`) drives two synthetic cells with different
densities and phase factors through a spike: forest voltages equal the per-cell voltages to
0.0 and 5.7e-14 mV over 400 substeps, spike outputs identical. The spiking-window gate of
the spec (same 17 cells over the 40 ms window) is not yet run on the fused path.

## 3. Mutation and learner (fused, dt 0.005, contended)

| Arm | Cells / compartments | build + init s | forward compile + 1 run s | s/event | learner |
| --- | ---: | ---: | ---: | ---: | --- |
| `fused-clone` (`clone` of cell 0) | 18 / 109,145 | 47.8 + 5.6 | 5.4 | 0.137 | not run |
| `fused-learner` | 17 / 107,537 | 48.9 + 5.4 | 5.1 | 0.128 | `compile_graph` 0.63 s; 2-event update compile + run 36.2 s, warm 3.85 s (per-cell uncontended: 9.9 s, 313.7-347.8 s, 3.6 s) |

A clone still changes the forest's shapes (the static-capacity lever is not implemented),
so it recompiles: 5.4 s for the forward program against 40.0 s on the per-cell path
(`mut-clone`, cache miss). The `add_contact` recompile arm could not run: the fused path
raises on any active contact (`h01_runtime._fuse_populations`), so every number above is a
contact-free network and the fused path is not yet runnable on a connected topology.

Learner: the sparse pp-prop layout on the forest is 56,858,914 eligibility elements,
454.9 MB, 17 colours, 19 state blocks, against 3,344,642 elements, 26.8 MB, 1 colour, 323
blocks per-cell. The forest's state blocks are 17x larger and the sparse-influence analysis
now colours per cell, so the factor allocation is 17x and sits at the 512 MB
`factor_limit_bytes`; peak device memory 2.3 GB (per-cell learner at dt 0.005: 0.72 GB).
The warm 2-event update (3.85 s contended) is not faster than the per-cell one (3.6 s
uncontended): the forward substeps got cheaper but the eligibility update got bigger. The
learner's sparse layout must be told the forest's per-cell block structure
(`forest_offsets`) before the fused path is used for training; until then the fused
population is a forward-path (scoring) lever only.

## What was not run

The uncontended fused forward arm (wait for the chains); the spiking-window gate on the
fused path; the `add_contact` arm (blocked by the missing fused delivery); the `pinned-nocb`
attribution; precision 32.
