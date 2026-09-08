# H01 population builder (SP8, builder part)

Extends `make_h01_network` and `examples/h01_verified_network.py` so a staged population build
(4 -> 12 -> 40 -> 104 cells) can be constructed with matched projection controls. This spec
covers construction only; no simulation is run under it. Physiology, dt, solver, and synapse
defaults are untouched.

## Inputs already fixed by earlier specs

- Topology: output of `prepare_connectivity` (`docs/evidence/h01-verified-network.json`):
  104 nodes, contacts with `construction_ready`, pinned placements. Today 2 contacts are
  construction-ready, so 4 cells are *incident* (`3955003482`, `4157825456`, `4188575291`,
  `5584343344`); the programme text's "6 incident cells" counts the blocked contact
  `65017731`, which the builder never constructs (its pre endpoint sits on component 55).
- Component inventory: `docs/evidence/h01-population-components.json` (`cells` rows with
  `cell_id`, `nodes`, `components`, `largest_component`, `largest_share`, `largest_has_soma`).
- Donor registry: `donor_for_tags(tags, polarity)` in `h01_cell_types.py`; the per-cell donor
  key is recorded in evidence `donors[cell_id]` and `cells[cell_id]["donor"]` exactly as for
  incident cells.

## API

```python
make_h01_network(topology, archive, annotations, *, include_isolated=False, control="ei",
                 cells=None, components=None, disconnected=False, ...unchanged...)
```

### `include_isolated`

- `False` (default): only cells incident on construction-ready contacts are built; the
  "Network requires nonempty contacts with resolved placements" error stands.
- `True`: every topology node that is not incident is an *isolated cell*, built from its
  **largest component**, which must be soma-bearing (`largest_has_soma` true in `components`);
  a node whose largest component lacks a soma, or which has no `components` row, raises
  `ValueError` (nothing is fabricated, no other component is substituted). Fragments are never
  joined. The cell gets no receptors and no projections; its output site is the soma
  (`anatomy().soma_location()`), exactly the existing path for a cell that is not a presynaptic
  source. Inputs (`currents_na`) may address isolated cells.
- With `include_isolated=True` the empty-contacts error does not fire; a build with zero
  constructible contacts is a population of isolated cells with zero projections.
- Evidence `isolated_cells[cell_id]` records
  `"isolated, largest component <c> of <M> nodes, fraction <f>"` where `<c>` is the component
  suffix, `<M>` the node count of that component (`round(largest_share * nodes)`), `<f>` the
  share to three decimals, plus the structured fields (`component`, `component_nodes`,
  `cell_nodes`, `fraction`, `components`).
- `components` is the parsed inventory: either the whole JSON object (`{"cells": [...]}`), the
  `cells` list, or a `{cell_id: row}` mapping. It is required when `include_isolated=True`
  and ignored otherwise.

### `control`

`control in {"ei", "e_only", "i_only", "disconnected"}`, mirroring
`h01_ei_circuit.py:113-115`:

| control | projection kept when |
| --- | --- |
| `ei` | always |
| `e_only` | presynaptic `dale_sign == +1` |
| `i_only` | presynaptic `dale_sign == -1` |
| `disconnected` | never |

Only projections change: cells, receptor placements, probes, and inputs are identical across
controls. Evidence gains `control`, `enabled_contacts` (annotation ids kept, in contact order),
`removed_contacts` (annotation ids dropped), and each `contacts[i]["enabled"]` follows the
table. `disconnected=True` remains a deprecated alias of `control="disconnected"` (docstring
note only, no warning); passing `disconnected=True` together with a `control` other than
`"ei"`/`"disconnected"` raises `ValueError`. Evidence `disconnected` stays as
`control == "disconnected"` for existing readers.

### `cells` ordering and the `--cells N` prefix rule

Evidence `cell_order` lists every candidate cell in build order:

1. incident cells, ascending integer id (the existing `simulated_cell_ids` order);
2. then isolated cells, ascending **largest-component node count** (`round(share * nodes)`),
   ties by ascending integer id.

`cells=N` builds the first `N` entries of `cell_order`. `N` must satisfy
`n_incident <= N <= len(cell_order)`; smaller would drop a contact endpoint, larger has nothing
to build, both raise `ValueError`. `cells=None` builds all of `cell_order`.
`simulated_cell_ids` is the built prefix; `cell_order` is the full order so a later stage can
be checked against an earlier one (each stage is a prefix of the next).

For the evidence topology the 12-cell stage is the 4 incident cells plus
`2530864375` (1,540 nodes), `3178243558` (2,748), `3751392341` (3,499), `2451406889` (3,750),
`678539249` (4,480), `4420044370` (4,624), `4260528559` (4,790), `2820958695` (4,864):
276,064 largest-component nodes against 245,769 for the 4-cell SP1 build (1.12x).

### Example flags

`examples/h01_verified_network.py` gains `--control {ei,e_only,i_only,disconnected}`
(default `ei`; `--disconnected` still accepted and mapped to `disconnected`),
`--include-isolated`, `--cells N`, and `--components PATH` (default
`docs/evidence/h01-population-components.json`). `--current-na` still applies to every built
cell (incident and isolated) so the four control arms of SP8 share identical inputs.

Timing: `braincell.network.Network.init_state()` exists and `Network.run` calls it
idempotently, so cell initialization is separable from stepping. The build JSON records
`init_state_seconds` (all `Cell.init_state` calls, includes the sparse DHS source assembly)
and `compile_and_run_seconds` (the `run` call: jaxpr tracing, XLA compile, stepping). XLA
compilation cannot be separated from stepping through the public API, so the JSON says
`timing_separation: "init_state separable; compile measured together with stepping"`.
`initialization_and_run_seconds` (their sum) stays for the SP1 parser.

## Memory and time stop rule (from SP1)

Measured at 4 cells (75,605 compartments, machine alone): peak RSS 1.35 GB, a = 245 s
init + compile, construction 240-280 s. Both are assumed proportional to compartment count
(DHS is O(N)), so the 12-cell stage is predicted at about 1.12x: roughly 1.5 GB and 275 s
construction. This is re-measured at 12 cells (construction only, under load) before 40 is
approved; the 40 and 104 projections are derived from the 12-cell measurement by the same
proportional rule, and a 12-cell RSS that projects 104 cells past available memory reduces N.
Abort limit for the 12-cell construction check: 900 s wall; a kill records "untested".

## Tests (sibling `h01_network_test.py`, fixtures only)

- each `control` value: projection count, `enabled_contacts`/`removed_contacts`, per-contact
  `enabled`; `disconnected=True` alias; alias conflict rejected; unknown control rejected.
- `include_isolated`: isolated node `14` built from `components`, no projections to it,
  `isolated_cells["14"]` string and fields, soma output site, donor recorded; missing row and
  soma-less largest component rejected; zero constructible contacts with `include_isolated`
  builds without the empty-contacts error; `cell_order` ordering by node count then id.
- `cells`: prefix, below-incident and above-total rejected.
- Coverage on `h01_network.py` >= 90 %.

## Addendum 2026-09-07 (registered before the second 12-cell check, SP8 staged build)

Predictions for the repeat 12-cell construction check (`--build --include-isolated --cells 12
--control ei`, measured under load), written before the run:

1. The `Invalid electrical region interval.` failure on isolated cell `2451406889` does not
   recur (the 1e-9 snap in `h01_ei_cell._snap_interval` covers the 2e-16 overshoot); all 12
   cells of `cell_order` build.
2. Construction completes under the 900 s abort limit. Point estimate from the SP1
   proportional rule: ~275 s alone; under load the first attempt reached the incident cells'
   `make_h01_ei_cell` end at 318 s, so the load-inflated expectation is 320-450 s.
3. Compartments about 1.12x the 4-cell 75,605, i.e. ~85,000 (derived from the largest-component
   node ratio 276,064 / 245,769 = 1.123; nodes and compartments are assumed proportional).
4. Peak RSS about 1.12x the SP1 1.35 GB, i.e. ~1.5 GB (derived); rejection if over 6 GB.

If construction completes, two 1 ms runs at dt 0.005 ms follow (control `ei`, then
`disconnected`): all traces finite; the `disconnected` run's conductance probes (`*_g`) are
identically zero while the `ei` run's are not; `init_state_seconds` and
`compile_and_run_seconds` are recorded. SP1's a = 245 s (init + compile at 4 cells) scales to
~275 s at 12 cells by the same rule; a 12-cell point is "inside 2x the SP1 linear prediction"
when construction, init + compile, and RSS each fall within [0.5x, 2x] of these derived values.

### Addendum 2026-09-07, 16:26 (after the construction check, before the run-phase retry)

The build-only check completed at 883.3 s construction under load (within the 900 s abort). The
first 1 ms `ei` run was killed by the same 900 s construction abort with 3 of 12 cells left to
discretize (the run's construction pace was ~55 s behind the build-only pace under varying
load); it is recorded as untested, not negative. For the two run-phase attempts that follow the
construction abort is set, before launch, to 1,350 s (1.5x the measured 883 s point); the 600 s
silence kill and a 2,700 s total wall cap stay. Predictions for the run phase are unchanged
(finite traces; disconnected conductance probes zero; init + compile + 200 steps ~276 s scaled
from SP1, expected higher under load; RSS ~1.5 GB, reject over 6 GB).

### Addendum 2026-09-07, 17:05 (init_state instrumentation; registered before the quiet-machine 12-cell runs)

Instrument change (no model or numerical change). `braincell.network.Network.init_state` exposes
no hook: it loops over `network.populations` and calls `Cell.init_state` once per uninitialized
cell, and both `Network.init_state` and `Network.run` skip initialized cells. So
`braintrace/datasets/h01_network_init.py` performs that same loop itself:

- `init_h01_network_states(network, progress, heartbeat_seconds=60)` emits
  `Initializing cell_<id> (i/n, k compartments)` and `Initialized cell_<id> in s s` per
  population, then calls `network.init_state()` (a no-op afterwards). It returns
  `init_state_seconds`, `init_seconds_by_population`, `initialized_populations`, `peak_rss_mb`.
- `heartbeat(label, emit, seconds)` runs a daemon thread that emits
  `heartbeat: <label> elapsed <t> s, RSS <m> MB` every `seconds` while the wrapped call is
  silent (a heartbeat is a progress line for the 10-minute silence rule). The example wraps
  each `Cell.init_state` and the whole `network.run` (jaxpr tracing, XLA compile, stepping) in it.
- `examples/h01_verified_network.py` gains `--init-only` (construct, initialize cell by cell,
  record `init_state_seconds`, `init_seconds_by_cell`, `init_peak_rss_mb`, write the build JSON
  with `execution = "constructed and initialized; not compiled or simulated"`, exit before
  compile and stepping; excludes `--duration-ms`) and `--heartbeat-s` (default 60). Run mode
  records the same init fields plus `run_peak_rss_mb` (`psutil` `peak_wset` where reported).

Tests: `h01_network_init_test.py` (heartbeat cadence, stop, psutil-less path, per-cell lines,
skip of initialized cells, network `init_state` call; 100 % line coverage) and
`h01_network_test.py::test_per_cell_init_progress_then_run_is_idempotent` on the fixture network
(real `Cell.init_state`, second call initializes nothing, `run` still finite).

Predictions for the quiet-machine 12-cell runs (`--include-isolated --cells 12`, first
`--init-only --control ei`, then 1 ms at dt 0.005 ms for `ei` and `disconnected`), written
before launch; "quiet" means `docker ps` shows only `synapse` and no python process other than
the C3 rescan worker uses more than 300 MB:

1. Construction near the SP1 rate scaled by 1.11x compartments (84,097 / 75,605): SP1 alone
   measured 240-280 s, so **270-310 s (derived)**. The under-load 883 s point is not the
   reference.
2. `init_state` is unmeasured; expected **between 1x and 3x the 4-cell a = 245 s**, i.e.
   245-735 s, with the 33,965-compartment cell `3955003482` the largest single term. Per-cell
   seconds are recorded so a superlinear-in-compartments cost is visible directly.
3. All traces finite in both 1 ms runs; the `disconnected` run's `*_g` conductance probes are
   identically zero and the `ei` run's are not.
4. Peak RSS near 1.5 GB (derived); the process peak is read from inside the real interpreter and
   the tree sum from the watcher.

Rejection: a nonfinite trace, RSS over 6 GB, or `init_state_seconds` over 1,500 s. Kill rules for
these launches: 10 minutes without a progress or heartbeat line, or the 1,500 s init_state abort;
no separate construction abort is applied on the quiet machine (the heartbeat and per-stage
lines cover it). The "inside 2x the SP1 linear prediction" test of the earlier addendum
applies to construction (275 s), init + compile + 200 steps (276 s) and RSS (1.48 GB).
