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
