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
`H01ForestCell(cells)` (an `H01Cell` over initialized source cells); `build_network(...,
fused=True)` in `examples/pp_prop/h01_runtime.py` builds and initializes the N cells as
today, fuses them and registers one population `forest` (records carry the offsets).

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
only roots remain; `solve` writes `rhs/d` for every root row. The pinned integrator
`h01_staggered_calcium_implicit` is unchanged; `h01_dhs_scan._voltage_step` takes the forest
schedule from `runtime.h01_dhs_forest` when the forest cell has set it and otherwise runs
the single-tree contraction as before. Stage count = the max over cells, so the scan depth
is that of the deepest cell, run once for all cells.

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

Contacts on the fused path (implemented, `h01_forest_delivery.py`): one BrainCell
`DeliveryBlock` per contact whose target is the contact's placed synapse layout in the forest;
the presynaptic spike is the forest spike at the pre cell's output CV, the same value
`population_spike` returns for that cell on the per-cell path. Ring buffers, arrivals and the
model's trainable magnitudes are the per-cell code. Parity: 0 / 3.9e-14 mV on the synthetic
pair, 2.2e-6 mV on the 17 cells with 3 synthetic contacts over 33 spikes.

Learner on the forest (implemented): the sparse pp-prop layout gets a declared segmentation
(`SparseInfluence.build_segmented`, `model.sparse_structure`): the block-level program analysis
still decides which blocks and ETP outputs interact; the model declares that cable blocks split
per cell at `forest_offsets` and that each contact's ring buffer reads `{pre, contact}` and
feeds `{post, contact}`, its synapse states `post`. Segments and blocks carry `(reads, feeds)`
ports; a parent reaches a child only where feeds meet reads. Factors are stored per position
through a slot table; JVP colours are the per-cell conflict structure. On the 17 cells the
layout is element-for-element the per-cell one (3,344,642 / 26.8 MB / 1 colour; 27.2 MB at 18
cells, 53.5 MB at 34: 1.57 MB per cell, the 512 MB limit at about 325 cells) and the
gradients equal the per-cell learner's (1.4e-17 on `input`, 0 elsewhere). A changed contact
graph needs a learner recompile (`sparse_structure_signature`); a clone into a declared slot
does not.

## 3. Lever 3: static capacity (implemented as pre-provisioned slots)

Rather than padding compartment arrays with anonymous rows and re-parameterising them at
clone time, every source cell is laid out `slots` times (`H01ForestCell(cells, slots=K)`,
`build_network(fused=True, slots=K, contact_capacity=C)`): slot 0 is active, the other slots
are dormant copies of the same anatomy, parameters and initial state, integrated every
substep and masked out of the drive, the spike output and the readout by the forest's
`active` state. A clone (`H01ArcModel.clone(parent)`) activates the next dormant slot of the
parent's source cell in place. Contacts live in `ForestContactTable` (`h01_forest_contacts.py`):
two shared BrainCell synapses per cell at the soma (`contact_exc` 0 mV / 2 ms, `contact_inh`
-80 mV / 5 ms, the two kinds `H01Topology.add_contact` creates; anatomical contacts must be
one of them and target the post soma), a table of `C` rows (pre cell, target point, kind,
delay, weight, active; all states) and one ring buffer `(depth, C)`; `add_contact(pre, post,
kind)` writes one row. Mutations write through the host with the array's device placement
preserved (`host_write`), so the jitted step keeps its input signature. `H01ArcModel.
recurrent_weight` is the table's weight column (length `C`, dormant rows zero gradient), the
encoder and readout span all `N x K` cells (dormant rows masked, zero gradient), and the
segmented learner declares the table's rows as ring-buffer segments.

What this buys and costs: zero recompiles and constant launches per substep for any clone
or contact within capacity (measured: 337-339 launches at 17, 17+1 and 34 cells; 0 compiles
after mutation), at the price that the substep always integrates every slot (34 laid out
costs what 34 active costs: 0.225 s/event against 0.103 s at 17, dt 0.005). Capacity is a
setting per run, not the evolve caps (4,096 cells would be 26 M compartments, 15 GB of
state). Not done: `H01Session.mutate` still rebuilds from the topology; the in-place path is
reachable at the model level and needs the session/checkpoint integration (remap of the
optimizer state is unnecessary when shapes do not change, so that integration is smaller
than the current one).

Gate (met): static forest against BrainCell per-population projections 1e-12 mV; dormant
copies equal their source; after `activate` + one table write the jitted substep and the
jitted ARC event run with 0 XLA compiles and unchanged shapes; model forward and gradients
equal the per-cell model (chain 0 -> 1 -> 2).

## 4. Lever 4: precision 32 (later, its own check)

Not part of this campaign. Check: the lever 0 window at precision 32 against 64 on the same
forest, max |dV| <= 1 mV and identical spike counts; the DHS and calcium solves are the
first suspects. Expected 2x on memory-bound kernels, 1x on launch-bound ones.

## 5. Files and manifest consequence

New (on the branch): `braintrace/datasets/h01_forest_discretization.py`, `h01_forest_cell.py`,
`h01_dhs_forest.py`, each with a co-located `_test.py`; `docs/evidence/h01_dt_spike_window.py`.
Changed: `examples/pp_prop/h01_runtime.py` (`build_network(..., fused=)`),
`h01_session.py` (`fused_population`, default False), `h01_arc_model.py` (fused drive/soma
routing), `braintrace/datasets/h01_dhs_scan.py` (forest schedule hook in `_voltage_step`),
`h01_pv_channels.py` (`_on_param_updated`). Deferred to lever 3: `h01_forest_delivery.py`
(contacts on the fused path) and `h01_forest_state.py` (capacity-padded states). Every
changed or added file is in `implementation_sha256` (`h01_session.py`,
`_numerical_settings_cached`: all `braintrace/datasets/h01*.py`), so the kept manifest
`docs/evidence/h01-keep-drop/arc-manifest/manifest.json` was rebuilt with
`examples/h01_arc_manifest.py --cells docs/evidence/h01-keep-drop/decision.json` on the
branch (topology unchanged; only the hashes and the new `fused_population: false` key
differ); `H01Session.build` validates against it. Any further runtime edit repeats that
rebuild.

## 6. Results of this campaign

- Lever 0 ([h01-dt-spike-window.md](../evidence/h01-dt-spike-window.md)): dt 0.005 fails the
  contract across the spikes: spike counts identical on all 17 cells (30 / 30) and spike
  times shifted by at most 0.064 ms, but that shift on a spike upstroke is up to 17.8 mV of
  instantaneous |dV| (per cell 0.49-17.8 mV). The ladder: dt 0.0025 shifts spikes by <= 0.029 ms
  (max |dV| 7.9 mV), dt 0.00125 by <= 0.0087 ms (2.65 mV), counts identical at every rung, so
  the pointwise 1 mV band (a spike-time band of about 3 us) fails everywhere and its edge lies
  below 0.00125 ms. **Decision (2026-09-17, every state variable in lockstep, section "Every
  state variable" of the evidence):** the rule put to the ladder was "dt 0.005 / 20 substeps
  becomes the session default only if no state variable drifts between spikes beyond its own
  dt-half difference (dt 0.005 against 0.0025)". Every variable does: between spikes, the
  dt 0.005-against-0.000625 difference is 1.42-1.76x its dt-half difference on all 22 traced
  quantities (V 2.98 against 1.71 mV; calcium 7.7e-6 against 5.2e-6 mM; SK z 9.9e-3 against
  6.6e-3; NaTs m 5.0e-2 against 3.4e-2; Ca_HVA h 2.4e-4 against 1.7e-4; Nap h 1.3e-4 against
  8.8e-5; Ih m 2.1e-4 against 1.4e-4; the two contact conductances whose dt-half pair is 1e-19
  are the pre spike landing on the same coarse step, not agreement), which is first-order
  convergence (0.875 / 0.5 = 1.75 in the limit), and the slow variables that do not reset on
  a spike (calcium, SK z, Ca_HVA h, Nap h) carry their maximum at the end of the window: a
  drift that halves per halving of dt and disappears at no rung above the pinned one. So
  `dt_ms` stays 0.000625 / 160 substeps, `fused_population` stays `False` by default, the kept
  manifest is not rebuilt, and the same is recorded in
  [h01-causal-model.md](../h01-causal-model.md). Caveat carried with the numbers: the +-1 ms
  spike mask leaves the AHP tail of a shifted spike in the "between" column (the 2.98 mV on V
  is 1.2 ms after a spike), so the between-spike V number is an upper bound; the ratio is set
  by the convergence order and does not move with the mask, and the slow-variable end-of-window
  drift is not a mask artefact. What would admit dt 0.005 or 0.00125 is a different contract
  (spike counts plus a spike-time band, as the keep tests' dt-half repeats used): J's call, not
  a measurement.
- Levers 1+2 ([h01-fused-population-profile.md](../evidence/h01-fused-population-profile.md),
  steps 1-2 of [h01-fused-population-steps.md](../evidence/h01-fused-population-steps.md)):
  implemented (`H01ForestCell`, `build_network(fused=True)`, `fused_population` setting,
  default False). 5,104 -> 327 launches per substep, 136 -> 8 while loops, GPU busy 8.3 ->
  <= 2.9-4.7 ms per substep, forward 0.180 -> 0.129 s/event at dt 0.005 (fused measured under
  GPU contention, so a lower bound on the gain), voltages equal to the per-cell path to
  7e-10 mV (no contacts) and 2.2e-6 mV over 33 spikes with 3 synthetic E contacts at the
  pinned dt (`h01_forest_delivery.py`; the I kind is checked at unit level only). Forward
  compile 42 -> 5-6 s, clone compile 40 -> 5.4 s, learner graph analysis 9.9 -> 0.6 s, learner
  compile 314-348 -> 36 s. The segmented sparse layout (`forest_offsets`, `sparse_structure`)
  puts the learner's eligibility back at the per-cell size (26.8 MB / 1 colour at 17 cells,
  27.2 MB at 18, 53.5 MB at 34; the 512 MB limit at about 325 cells) with gradients equal to the
  per-cell learner (1.4e-17), so the fused path is a training lever, not only a scoring one.
  Open: 90-94 % of the fused substep is unattributed command-buffer time (the per-mechanism
  channel kernels, lever "one kernel", `pinned-nocb` unrun); the uncontended fused timing and
  the spiking-window gate on the fused path at the session dt wait for the keep chains; a
  chain of contacts widens the segmented factor store to the depth of the chain (7 colours,
  188 MB for a 3-deep chain on 17 cells).
- Lever 3 (step 3 of the steps evidence): implemented as pre-provisioned slots (section 3):
  339 launches per substep at 17, 17 + 1 clone and 34 cells (12 over the dynamic path, constant
  under mutation), 0 XLA compiles after a clone and a contact written in place, forward and
  gradients equal to the per-cell model on a 0 -> 1 -> 2 chain. Cost: 34 laid out costs what
  34 active costs (0.2245 s/event at 17 + 1 clone against 0.2267 at 34, dt 0.005, contended).
  `H01Session.mutate` still rebuilds from the topology: the in-place path stops at the model.
