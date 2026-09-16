# H01 fused population: one forest, static capacity, one kernel set (2026-09-16)

Goal (J): code fast enough that grow-and-prune can clone the full-fidelity human cells
into thousands and then millions without dumbing anything down. No reduced models,
no fewer compartments, no coarser cells. Every lever below keeps every compartment,
every channel and every donor parameter of every cell.

Baseline (branch `campaign/h01-evolve-profile-20260916`,
[h01-evolve-event-profile.md](../evidence/h01-evolve-event-profile.md)): 17 cells /
107,537 compartments, each cell its own `braincell` population; forward 1.357 s per 0.1 ms
event at dt 0.000625 (160 substeps), 0.180 s at dt 0.005 (20 substeps); 9.0 ms wall per
substep with 4,300-5,100 stream launches; the single-substep HLO holds 136 while loops =
17 cells x 8 DHS scans; `cell-runtime` shows 17 launches per substep; 75 % of substep busy
time in command buffers, axial solve 12.7 %, throughput 12.7 M compartment-updates/s =
7e-5 of fp64 peak. Cold compile ~320 s per topology; a mutation (`add_contact`, `clone`)
changes array shapes and is a cache miss. The GPU is launch-bound: 17 separate kernel sets
run one after another, each too small to fill the device.

## 0. What BrainCell 0.1.0 allows (read on the box venv, `braincell/`)

- `Morphology` is one tree by construction: only `Morphology.__init__` registers a node
  with `parent_id=None` (`morph/morphology.py:306-309`); `attach` always needs an existing
  parent. `_build_cv_tree` raises `"Expected exactly one root CV"`
  (`_discretization/base.py:514-518`). `node_build.py:151-154` would silently glue a second
  parentless branch onto branch 0's proximal node, which couples the cells: never use it.
- The DHS kernels are forest-agnostic: every row with `parent_rows == -1` gets zero
  lowers/uppers and is routed to the sentinel row `n_point` (`quad/_staggered.py:307-316`);
  peel levels are computed per component (`_compute/scheduling.py:155-178`);
  `root_node_id`/`root_cv_id` are read only by `vis/`. braintrace already owns this path
  (`H01Cell.init_state` builds `build_dhs_static_source_1d`; `h01_dhs_scan._voltage_step`
  and `h01_calcium_solver` are the registered integrator).
- `CellRuntimeState.from_cell` reads only `cell.node_tree`, `cell.cvs`, `cell.pop_size`
  (`_compute/runtime.py:280-288`); `Discretization`, `CVTree`, `NodeTree` are plain frozen
  dataclasses with no validation on construction. The morphology is not consulted by the
  runtime, the solver, the probes or the current assembly after discretization.
- Mechanism layouts group by declaration value `(class, params, name, coverage, ion
  selectors)` (`runtime.py:290-305`, `mech/_density.py:181-206`); every dense layout is one
  runtime node over all `n_point` points with `g_max` masked to zero off its region
  (`runtime.py:2447-2456`). Parameters are per-point buffers `pop_size + (n_point,)`
  (`runtime.py:1060-1085`); `set_state(layout_id, var, array)` accepts a full-shape array and
  `setattr`s it on the node, calling `_on_param_updated` if present (`runtime.py:2471-2488`).
  Ions are one runtime object per name with per-point `E` scattered from their layouts
  (`runtime.py:1602-1659`).
- Two hard preconditions: with one named `Ion` per family every channel binds
  unambiguously (`runtime.py:2284-2293`); `MechanismProbe` raises if two layouts of one
  `instance_name` share a point (`probes.py:134-138`).

Consequence: a forest of N different cells can be one `braincell` cell if its
`Discretization` is built directly from the N per-cell discretizations (a vendored
subclass in this repository; no BrainCell change), and one kernel per mechanism follows
if the N donor declarations are replaced by one canonical declaration per
`(class, name)` with per-point parameter vectors written after `init_state`.

## 1. Lever 0: cable timestep from step 0

Check: [h01-dt-spike-window.md](../evidence/h01-dt-spike-window.md) /
[.json](../evidence/h01-dt-spike-window.json), script
[h01_dt_spike_window.py](../evidence/h01_dt_spike_window.py). The 17-cell ARC model
(`build_network` -> `init_h01_network_states` -> `H01ArcModel`, the manifest path) is driven
for 40 ms with each cell's donor primary current (`kept-currents.json`, the keep-test probe)
held through the model's `drive` state from 2 ms; soma voltage recorded every substep.
Contract: max |dV| <= 1 mV on every cell against the pinned dt 0.000625 / 160 and identical
spike counts per cell. Result and the session default it sets: section 6. Substeps per event
scale 1:1 with the forward and learner event cost (160 -> 20 substeps: 1.357 -> 0.180 s).
Changing `dt_ms`/`substeps` in `numerical_settings()` edits `h01_session.py`, which is in its
own `implementation_sha256` list: every manifest is rebuilt (`examples/h01_arc_manifest.py`)
and the qualification in `h01-timestep-ladder.json` (`# qualified step` comments) is
superseded by the spike-window evidence, which must be said in the ladder record.

## 2. Levers 1+2: one forest cell, one kernel set

New module `braintrace/datasets/h01_forest_cell.py` (+ `_test.py`), class
`H01ForestCell(H01Cell)`; builder `fuse_cells(cells, records) -> (forest, map)`.

Forest discretization (`h01_forest_discretization.py`): concatenate the N per-cell
`Discretization`s: CV ids, node ids, branch ids and `CVEdge`/`NodeEdge` endpoints offset by
cumulative counts; `cv_to_mid_node_id` concatenated; `branch_to_cv_ids` concatenated;
`root_cv_id`/`root_node_id` = cell 0's (informational). Each CV keeps its own `area`, `cm`,
`ra`, `r_axial_*`, `v` (donor rest) and point mechanisms (soma clamp, probes, contact
synapses) with offset ids. Compartment count is exactly the sum: nothing is merged or dropped.

Canonical layouts: each CV's `density_mech` tuple is rewritten so that every donor
`Channel(class, name=..., g_max=..., m_open=..., h_slope=...)` becomes the one canonical
`Channel(class, name=name, ion_name=family, g_max=0)` per `(class, name)`, and every donor
`Ion(class, name=..., E=..., decay=..., gamma=...)` the one canonical `Ion(class, name=name)`.
Leak `IL` keeps per-CV `E` the same way. After `init_state` (H01Cell's: `from_cell`,
scheduling, `build_dhs_static_source_1d`, `_prepare_levels`) the forest cell writes the
per-point vectors gathered from the N cells' runtime nodes into the canonical nodes with
`runtime.set_state(layout_id, var, vector)`: `g_max`, `E`, `h_slope`, `m_open`, `m_close`,
`h_open`, `h_close`, calcium `decay`, `gamma`, `depth`. `_PVChannel` gains
`_on_param_updated` so that `phase_factors` follows the vectors (the rate accessor already
broadcasts arrays: `h01_pv_channels.py:113-126`). Gate, calcium and synapse `DiffEqState`
values are copied point-for-point from the N initialized cells, so the forest starts in
exactly the per-cell state. Expected: one runtime node per `(class, name)` (about 12
channel nodes, 3 ions, 1 leak, 1 clamp layout with 17 active points) instead of 17 x that.

Forest solve (`h01_dhs_forest.py`): the existing contraction schedule requires one tree
(`h01_dhs_contraction._build`: "connected and acyclic", `edges.shape == (n-1, 2)`), so a
forest variant builds the same Schur stages with every root kept active and terminates when
only roots remain; `solve` writes `rhs/d` for every root row. Registered integrator
`h01_forest_calcium_implicit` = `h01_calcium_solver._implicit_step` with a `_voltage_step`
that takes the forest pack from `runtime.h01_dhs_forest_pack`. Stage count = the max over
cells, so the scan depth is that of the deepest cell, run once for all cells.

Spike output: `_ForestSpike(base, output_cv_ids, n_cv)` masks to the N output CVs; per-cell
spikes = `spike[..., output_cv_ids]` (one gather, no `any` reduction per cell).

Model integration: `H01NetworkStep` sees one population; `H01ArcModel` routes `drive` as one
`.at[..., soma_points].add(drive * nA)` on the fused clamp layout and reads `_soma()` as one
gather `V[..., soma_cv_ids]`; `readout_features`, `reset_episode` unchanged.
`build_network(topology, archive, ..., fused=True)` builds the N cells as today (their
discretizations are the input), fuses them, registers one population `forest`, and returns
records with per-cell offsets (`cv_offset`, `point_offset`, `soma_cv`, `output_cv`).
`numerical_settings()` gains `fused_population: True`; `False` keeps the per-cell path
selectable for the diff.

Expected per substep: launches from ~4,300-5,100 to the per-mechanism count (one exp-Euler
kernel per gate, one current kernel per channel, one forest elimination/substitution pair
per contraction group, one calcium solve): about 150-300; 136 while loops -> 8; the
`cell-runtime` class 17 -> 1. Speedup bound: the 17 x serial kernel sets become one set over
107,537 points, which is still small for the device, so the substep should approach the
single-cell substep time of the largest cell (a few ms) rather than 9 ms x 17/17; expect 3-6x
on the forward event before the command-buffer attribution (section 6) refines it.

Gate: same 17 cells, same manifest, `fused_population` True against False, the spiking
window of lever 0 at the session dt: max |dV| <= 1 mV over the window on every cell (expected
< 1e-9 mV: same arithmetic, same order within each cell; only the forest elimination visits
rows in a different order) and identical spike counts. Unit test: two synthetic 3-compartment
cells with different densities -> forest voltages equal the per-cell voltages to 1e-12 mV
over 100 substeps; the same with one contact.

Cloning: a clone adds no kernel launches (the fused kernels are over all points already);
it still changes array shapes until lever 3, so it recompiles once per mutation as today.

## 3. Lever 3: static capacity (no recompile per mutation)

Capacity is set once from the evolve caps: `max_neurons` (4,096) -> compartment capacity
`C_cv` = `max_neurons x mean compartments per source cell` rounded up to a multiple of 1,024,
point capacity `C_pt` likewise, and `max_recurrent_edges` (65,536) -> contact capacity
`C_syn`. The forest is allocated at capacity from the start: padded CVs are isolated
(no edges, DHS row `diag=1, lowers=uppers=0, solve=0`, the existing sentinel treatment),
`g_max=0` on every channel, `cm` and `area` positive so `I/C` is finite, `V` at rest, masked
out of spike output, probes and readout. Every array a mutation writes becomes runtime
input instead of a traced constant:

- `H01ForestState`: per-point parameter vectors (`g_max`, `E`, phase factors, `h_slope`,
  calcium constants), `C`, `cv_area`/`point_area`, the DHS pack (`diag`, `lowers`, `uppers`,
  contraction stage tables, backsub jumps, `dynamic_rows`, `row_capacitance`), the clamp
  points, `soma_cv_ids`, `output_cv_ids` and the active masks, all held as
  `brainstate.ShortTermState`s of fixed shape. The H01 channel classes (`_PVChannel`,
  `_L2Channel`, `PVCalcium`, and `H01SodiumFixed`/`H01PotassiumFixed` registered in this repo)
  read their parameters through properties backed by those states, so a write is a device
  `.at[].set` and the compiled step is unchanged. Contraction stage tables are padded to
  capacity width per stage with sentinel entries (`valid=False`, parent = sentinel).
- Contacts (`h01_forest_delivery.py`): one table of `C_syn` rows (`pre_cell`, `post_point`,
  `delay_substeps`, `weight`, `reversal`, `tau`, `active`) and one ring buffer
  `(max_delay_substeps, C_syn)`; per substep: gather pre-cell spikes -> enqueue at the delay
  slot -> pop the current slot -> segment-add the conductance jumps onto the post points of
  one `ExpSyn` layout of `C_syn` active points; masked rows deliver zero. This replaces the
  `braincell` network delivery blocks (which allocate per contact and per population) for the
  fused path only. `H01ArcModel.recurrent_weight` is the `weight` column (capacity length,
  masked), so the learner's parameter shapes are fixed too.
- `clone(parent)`: copy the parent's compartment rows (`cv_offset .. +n_cv`) and point rows
  into the next free rows of every state, add the offset to the copied edge endpoints in the
  stage tables and the backsub jumps, append the clone's soma/output/clamp ids, mark active.
  `add_contact`: write one contact row. `prune`: clear the active flag. Zero kernel launches
  are added, zero recompiles happen: checked by `jax.log_compiles` around one forward event
  after each mutation (0 compile messages) and by the per-substep launch count before/after
  a clone (equal). `H01Topology`/`H01Session.mutate` still produce the child topology and the
  checkpoint; the fused runtime applies the mutation in place instead of rebuilding.

Gate: the lever 2 gate on the padded forest (identical voltages to the unpadded forest, max
|dV| <= 1e-9 mV), plus: after `clone`, the clone's soma trace equals its parent's under the
same drive to 1e-9 mV; after `add_contact` with weight 0 nothing changes; the padded-capacity
forward event costs no more than 1.1x the unpadded one at 17 cells (masked rows are real
rows: this is the price of static shapes; capacity is sized to the evolve caps, not more).

## 4. Lever 4: precision 32 (later, its own check)

Not part of this campaign. Check: the lever 0 window at precision 32 against 64 on the same
forest, max |dV| <= 1 mV and identical spike counts; the DHS and calcium solves are the
first suspects. Expected 2x on memory-bound kernels, 1x on launch-bound ones.

## 5. Files and manifest consequence

New: `braintrace/datasets/h01_forest_discretization.py`, `h01_forest_cell.py`,
`h01_dhs_forest.py`, `h01_forest_solver.py`, `h01_forest_delivery.py`, `h01_forest_state.py`
(lever 3), each with a co-located `_test.py`; `docs/evidence/h01_dt_spike_window.py`.
Changed: `examples/pp_prop/h01_runtime.py` (`build_network(..., fused=)`),
`h01_session.py` (`fused_population`, dt), `h01_arc_model.py` (fused drive/soma routing),
`braintrace/datasets/h01_pv_channels.py` (`_on_param_updated`), `h01_pv_calcium.py` (per-point
constants already supported). Every one of these is in `implementation_sha256`
(`h01_session.py`, `_numerical_settings_cached`: all `braintrace/datasets/h01*.py`), so any
edit forces a manifest rebuild; the kept manifest is rebuilt once with
`examples/h01_arc_manifest.py` after the last edit of the campaign, and the pinned
comparison run is the per-cell path under the same rebuilt manifest.

## 6. Results of this campaign

Filled in by the evidence files as they land:

- Lever 0: [h01-dt-spike-window.md](../evidence/h01-dt-spike-window.md).
- Command-buffer attribution (which kernels fill the 75 %): `pinned-nocb` arm of the profile.
- Levers 1+2 before/after: [h01-fused-population-profile.md](../evidence/h01-fused-population-profile.md).
- Lever 3 recompile check: same file.
