# H01 to Example 21 adapter: interface contract (SP9.2, spec only)

Status: contract only. No code is written under this spec; the user decision of 2026-09-07
(`docs/specs/2026-09-07-h01-population-programme.md`, "User decisions": Example 21 adapter =
"Deferred: interface-contract spec only, no code") stands. A future implementer builds against
this document; anything not stated here is not promised. Every identifier below was checked
against the cited file at commit 312e391 of `feat/h01-braincell`.

Sources: `examples/pp_prop/21-braincell-arc.py` (model), `examples/pp_prop/example21_arc_adapter.py`
(coordinator bridge), `examples/pp_prop/example21_structural.py` (`SparseTopology`),
`braintrace/datasets/h01_network.py` (`make_h01_network`), `braintrace/datasets/h01_construction.py`
(`H01Cell`), `braintrace/datasets/h01_ei_cell.py` (`make_h01_ei_cell`), `examples/h01_verified_network.py`,
`docs/evidence/h01-network-throughput.md`, `docs/evidence/h01-population-status.md`, `AGENTS.md`.

## 1. Inputs

The adapter takes exactly the two return values of `make_h01_network(topology, archive, annotations, ...)`
(`h01_network.py`): a `braincell.Network` named `"h01_verified"` whose populations are named
`"cell_<source ID>"`, and the `evidence` dict with keys `nodes`, `simulated_cell_ids`, `cell_order`,
`cells`, `donors`, `contacts`, `blocked_contacts`, `control`, `disconnected`, `enabled_contacts`,
`removed_contacts`, `isolated_cells`, `qualification`, `topology_audit_sha256`. It also takes the
`control` literal, one of `CONTROLS = ("ei", "e_only", "i_only", "disconnected")`.

The adapter emits one `SparseTopology` (`example21_structural.py`, fields `input_source`,
`input_target`, `input_value`, `recurrent_source`, `recurrent_target`, `recurrent_value`, `readout`,
`dale`, `mechanisms`, `owner_codes`, `neuron_ids`; `neuron_count` is `len(dale)`). Field mapping and
provenance:

| `SparseTopology` field | Source in H01 outputs | Measured or assumed |
| --- | --- | --- |
| `neuron_ids` | `int(cell_id)` for each entry of `evidence["simulated_cell_ids"]`, in that order; index i in every other array refers to position i here | Measured (H01 proofread IDs) |
| `dale` | `topology["nodes"][*]["dale_sign"]` (+1 or -1; `make_h01_network` raises if it conflicts with `cell_sign(annotations.metadata(id).tags)`) | Measured tags. Note: never 0, so Example 21's "untyped" code does not occur |
| `owner_codes` | none; emit `None` (the model then leaves it `None`); checkpoint writers fill -1 = inactive per `checkpoint_arrays` | Not measured |
| `mechanisms` | one tuple per cell naming `evidence["donors"][cell_id]` and `evidence["cells"][cell_id]` record fields | Assumed (borrowed donor profile; see `h01-population-status.md`, "Physiology per cell"). Constraint: `_topology_optimizer_from_arrays` and `load_parent_checkpoint` reject nonzero `mechanism_codes`, so this field cannot round-trip through the format-1 checkpoint; the adapter must keep donor provenance in its own evidence JSON |
| `recurrent_source`, `recurrent_target` | one row per entry of `evidence["contacts"]` with `enabled == True` (equivalently `enabled_contacts`), `pre_cell`/`post_cell` mapped to indices in `neuron_ids` | Measured endpoints (`endpoints_verified`); which rows are present is set by `control`, not by anatomy |
| `recurrent_value` | `signed_weight_us` of the same contact rows | Assumed (`excitatory_weight_us=.01`, `inhibitory_weight_us=.02` defaults; literature pins exist only for PV-to-pyramidal, `h01-ie-synapse-literature.json`) |
| `input_source`, `input_target`, `input_value` | the encoder's feature-to-cell assignment (Section 3); 441 rows = `N_INPUTS` | Assumed; no H01 datum constrains it |
| `readout` | `(neuron_count, N_READOUT)` initial array or `None` (`topology_from_checkpoint` also uses `None`) | Assumed |

`neuron_count` is the number of built cells (4 today; at most 104), not `N_NEURONS = 2048`.
`BrainCellArcModel(topology)` accepts any `neuron_count`; the format-1 checkpoint requires
`input_indptr` of shape `(442,)` (`load_parent_checkpoint`, `topology_from_checkpoint`).

## 2. State ownership

Example 21 today: `BrainCellArcModel` (`21-braincell-arc.py`) owns `brainstate.ParamState`
`input_weight`, `recurrent_weight`, `readout_weight`, `readout_bias`; `brainstate.HiddenState`
`previous_spikes`; and one `CompatibilityHodgkinHuxley(neuron_count)` (`braincell.SingleCompartment`,
`solver="ind_exp_euler"`, `V_initializer` -65 mV, `V_th` 0 mV) whose `V` and `spike` states are read
in `_advance`, `readout_features`, `reset_episode`, and `decoder_boundary_intervention`. Its trainable
sparse operations are `braintrace.sparse_matmul` and `sparse_dale_matmul`, which the compiler links
to those hidden states.

H01 replacement: the `braincell.Network` from `make_h01_network` owns one `H01Cell`
(`h01_construction.py`, subclass of `braincell.Cell`, sparse DHS axial operator in `init_state`) per
population, each with 4,000 to 30,000+ compartments (75,605 total at 4 cells). Membrane and channel
states live in the cell's `state_trees` and are reached through `init_state`, `reset_state`,
`get_state`, `set_state`, `get_cv_state`. They are BrainCell hidden state; the adapter does not
re-declare them. The adapter owns exactly:

- `ParamState`: `input_weight` (input CSR values), `recurrent_weight` (per enabled contact; a
  multiplicative scale on the projection `weight` in microsiemens, initial 1.0), `readout_weight`,
  `readout_bias`. Nothing inside a cell (channel densities, donor profile, receptor parameters) is a
  `ParamState`; those are frozen physiology under the transfer directive (Section 6).
- `HiddenState`: `previous_spikes` (one float per cell, from the `output_voltage` probe crossing or
  the population spike record), and a soma-voltage buffer read from the `StateProbe(field="v",
  name="voltage")` that `make_h01_ei_cell` places at the soma (`h01_ei_cell.py:149`) and the
  `output_voltage` probe placed by `_register_cell`.

Loop ownership: `braincell.network.Network.run(*, dt, duration, delay_quantization, event_backend,
brainevent_backend, spike_recording)` compiles its own internal `for_loop`/`scan` over the whole
duration and returns a `NetworkRunResult` (verified in the installed source). That loop is closed; the
Example 21 compiler cannot see a step inside it. Therefore the adapter MUST expose a single-step
callable, `update(event)`, that advances the network by one biological interval and returns the
per-cell soma voltage array, so that `compile_pp_prop_model` (`braintrace.compile(model,
braintrace.pp_prop, zeros((N_INPUTS,)), batch_size=None, decay_or_rank=TRACE_DECAY,
vjp_method="single-step")`) can trace it, exactly as `BrainCellArcModel.update` is traced today.
Whether BrainCell exposes such a step for a `Network` (as opposed to `Network.run`) is an open
implementation question; if it does not, the adapter is blocked and says so. `step`, `interval`,
`readout`, `readout_features`, `reset_episode` keep their current signatures; `run_event_sequence`,
`_score_event_sequence`, `run_pp_prop_sequence` continue to drive `model.step` under
`brainstate.transform.for_loop`/`scan`/`jit` (AGENTS.md rule 10) and are not changed.

## 3. Hooks

Injection. Today `_advance` computes `bounded_population_current(input_drive, recurrent_drive)`
(0.02 tanh + 0.01 tanh, in mA/cm^2) and calls `self.cell.update(current)`. The adapter replaces this
with an encoder that maps the 441-feature event to one soma current per cell, in nanoamperes, and
delivers it through the same soma pulse channel `make_h01_ei_cell` uses for `current_na`
(`make_h01_network(currents_na=...)`, "Pulses start at 2 ms and last 3 ms"). Contract: the encoder is
`input_drive -> nA` with a stated bound; it is the only path by which an event reaches a cell;
recurrent drive is NOT injected as current but arrives through the placed `ExpSyn` projections
(`_place_contact`), scaled by `recurrent_weight`. This is a change of mechanism from Example 21,
where recurrent drive is an additive current; it must be stated in every result.

Readout. `readout_features()` today returns `tanh((V_mV + 65) / 20)` per neuron from
`self.cell.V`. The adapter returns the same expression per cell from the soma `voltage` probe
(`h01_ei_cell.py:149`), giving `(neuron_count,)` features; `readout()` stays `features @
readout_weight + readout_bias` with `N_READOUT = 360` logits, consumed unchanged by
`_supervised_request_loss`, `decode_prediction`, and `direct_query_metrics`. The E model rests near
-84 mV (`h01-population-status.md`, pair row), so the -65 offset is not centred for H01 cells; the
offset is a constant the implementer may change, recorded as such.

## 4. Timing

Example 21 advances one event per `dt_ms=0.1` (`_advance`), optionally split by `interval(...,
substeps=n)` via `integration_substep_events`; `matched_integration_check` compares one step against
two half-steps. H01 runs at `--dt-ms 0.005` (default in `examples/h01_verified_network.py`; the
transfer pages use 0.005 and 0.0025). One Example 21 event therefore costs 20 H01 substeps; the
adapter's `update(event)` runs 20 internal substeps at 0.005 ms and the dt-halving control runs 40 at
0.0025 ms. `integration_substeps` in the checkpoint must stay 1 (`_topology_optimizer_from_arrays`
rejects other values); the 20 substeps are internal to `update`, not a checkpoint field.

Measured cost (`docs/evidence/h01-network-throughput.md`, SP1, 4 cells, 2 projections, 75,605
compartments, `h01_staggered_scan`, precision 64): a = 245.39 s init + compile, b = 0.05764 s per
0.005 ms step, 11.528 s per simulated ms; three points linear within the two-repeat limit.

Derived, not measured (label every use "derived"):

- one Example 21 event (0.1 ms) at 4 cells: 20 * 0.05764 = 1.15 s;
- one 705-event episode (the fixed length in `SupervisedQuery`): 70.5 ms = 813 s plus compile;
- the 128-update `ORDINARY_UPDATES` schedule (`_EXPECTED_UPDATES = 128`): 9,024 simulated ms =
  about 104,000 s (29 h) at 4 cells, excluding scoring passes over 400 tasks;
- 104 cells: the programme's SP1 sizing rule assumes cost proportional to compartments, about 11x
  (skeleton nodes 2,804,549 vs 245,769), so about 127 s per simulated ms, 12.7 s per event, 9,000 s
  per episode, and about 320 h for 128 updates. The 11x factor is unvalidated until the 12-cell
  build in SP8.

Consequence: an ordinary 128-update run is not feasible at this cost, and the run-duration policy
forbids quoting unmeasured durations or running past the approval threshold without a measured
estimate. The 8-update `PROOF_UPDATES` schedule with `PROOF_DEADLINE_SECONDS = 180.0` is also out of
reach at 4 cells (8 episodes about 6,500 s derived). Reduced-order options exist and are named only,
not chosen: shortened event episodes; coarser dt after a dt-halving pair; a fixed-input surrogate per
cell (rate or spike response fitted to the multicompartment run, checked against it); fewer cells
(`--cells N`); smaller `max_cv_length_um` mesh reduction after a matched-mesh gate.

## 5. Not promised

- Any ARC score. Example 21 records 0 pass@1 on the current model (memory note); nothing here
  claims the H01 substrate changes that.
- Any eligibility-trace or learning-rule correctness claim about the adapter that is not measured
  through the finite-window oracle `chunked_online_param_gradients`
  (`braintrace/_testing/oracle.py:140`). Per `AGENTS.md`, a whole-sequence multi-step VJP returns
  BPTT for every algorithm and makes such an assertion vacuous; assertions about the compiler seeing
  the adapter's ETP primitives may use the whole-sequence path.
- Any physiology claim beyond the representation table (`docs/tutorials/h01_braincell.rst`, "What is
  preserved and what is assumed" and "The measured E/I pair: what its behaviour represents"): cells
  are borrowed donor profiles, synapse numbers are assumed, inhibition is unverified, and the E
  transfer is unqualified. The adapter inherits `evidence["qualification"]` ("Verified anatomical
  subset with assumed synapse dynamics and unqualified candidate cells.") verbatim.
- Determinism of the H01 substrate under `brainstate.transform.jit`, a fixed 705-event episode
  fitting in memory (RSS 1.33 to 1.35 GB at 4 cells and 10 ms, `h01-network-throughput.md`), or
  that `Network` exposes a single step (Section 2).

## 6. Qualification before any training claim

All of the following must hold, each with its decision JSON, before an adapter run is described as
training rather than construction and delivery:

1. Transfer gate Y2 (`docs/specs/2026-09-07-h01-transfer-isolation.md`, programme SP2): the I cell
   and, for any E cell used, the E cell pass the matched-mesh BrainCell-vs-NEURON gate; SP2 currently
   reads `time_level_open` (`h01-population-status.md`, "I transfer"). The standing directive (no
   serial physiological tuning while transfer is open) applies to the adapter too.
2. Usable tier per cell: every simulated cell's donor has a printed usable-tier verdict at the inputs
   it will receive (SP3, SP4, SP6d); borrowed types are labelled in the adapter evidence.
3. Functional inhibition (SP5): one E spike shifted or one cycle lengthened beyond the DLF limit
   with the placed receptor; today both controls show no E spikes.
4. Population run with controls (SP8): the four `CONTROLS` at the final N with identical cells,
   inputs, dt, duration, all traces finite, a per-cell spike table, and the adapter's own
   `disconnected` run showing no conductance arrival.

## 7. Acceptance tests for the future implementation

Mirroring the programme rules (prediction and rejection registered before each run; a killed run is
untested; both tiers stated), co-located `*_test.py` (AGENTS.md rule 9):

1. Interface compile: `compile_pp_prop_model(adapter_model)` returns a learner; the compiler links
   `input_weight` and `recurrent_weight` to the soma-voltage hidden state; `readout()` has shape
   `(360,)`; `readout_features()` has shape `(neuron_count,)`.
2. Topology mapping: for each `control`, `recurrent_source`/`recurrent_target` equal the
   `enabled_contacts` mapped through `neuron_ids`; `dale` equals `nodes[*]["dale_sign"]`;
   `validate_topology_dale` passes; `canonicalize_topology_and_optimizer` is idempotent on it.
3. One update: one `PPPropEpisodeTrainer.update_episode` on a shortened episode leaves every state
   finite (`optimizer_is_finite()`, all probes finite as `h01_verified_network.py` checks) and
   changes `recurrent_weight` (`recurrent_weight_changed` in `_real_workflow_report`); wall time
   recorded against the Section 4 derived figure, mismatch reported against the prediction.
4. Controls: `disconnected` shows zero conductance in every `syn_*_g` probe; `e_only` and `i_only`
   differ from `ei` only at receivers of removed contacts; the encoder-off run (zero event) leaves
   all cells at rest within the dt-halving limit.
5. Learning-rule claims, if any are made: measured with `chunked_online_param_gradients`, never with
   a whole-sequence VJP.
6. State preservation: `evaluate_forward` raises on no change to parameters or hidden state after a
   scoring pass; `_score_runtime` parameter-equality check passes.
7. Cost gate: any test predicted over 15 min carries a measured estimate and approval
   (programme "Compute" decision); otherwise it uses `--cells` and a shortened episode.
