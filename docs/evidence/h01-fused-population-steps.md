# H01 fused population, steps 1-3: contacts, segmented learner, static capacity (2026-09-16/17)

Spec: [2026-09-16-h01-fused-population.md](../specs/2026-09-16-h01-fused-population.md).
Continues [h01-fused-population-profile.md](h01-fused-population-profile.md) on branch
`campaign/h01-fused-population-20260916`. Receipts:
[h01-fused-population-profile/](h01-fused-population-profile/) (`*-report.json` per arm,
`contact-parity-pinned.json`, `plots/`). Executor as before: Vast 50616476, worktree
`/workspace/braintrace-fused`, `taskset -c 160-223`; **the two keep-test chains ran throughout,
and from 23:10 UTC the dt state check shared the GPU too**, so every wall time is contended
and comparable only within this session (per-cell references from the profile campaign are
uncontended and are quoted as such). GPU-busy and launch counts per substep are profiler
readings.

## Step 1: contact delivery inside the forest

`braintrace/datasets/h01_forest_delivery.py`: one BrainCell `DeliveryBlock` per contact whose
target is the contact's own placed synapse layout in the forest; the presynaptic spike is the
forest's spike at the pre cell's output CV (`forest_presynaptic_spikes`), which is the value
BrainCell's `population_spike` returns for that cell on the per-cell path (every other CV is
masked). Ring buffers, arrivals and the ARC model's trainable magnitudes are the per-cell code
unchanged. `build_network(fused=True)` accepts any contact.

Parity, unit (`h01_forest_delivery_test.py`, three synthetic cells, one E and one I contact
from the spiking cell, 500 substeps at dt 0.005): forest against BrainCell's per-population
projections **0.0 / 3.9e-14 / 0.0 mV** on the three cells while the contacts move the post
cells by 77.7 and 11.4 mV.

Parity, real cells (`h01_dt_spike_window.py parity`, [contact-parity-pinned.json](h01-fused-population-profile/contact-parity-pinned.json),
[plots/contact-parity-pinned-soma.png](h01-fused-population-profile/plots/contact-parity-pinned-soma.png)):
kept 17 cells plus 3 synthetic contacts (0.05 uS, E 0 mV / 2 ms, 0.5 ms, cells 0->1->2->3, all
delivered: the spike count rose from 30 to 33), 40 ms window, dt 0.000625, fused against
per-cell: **max |dV| 2.18e-6 mV, 33 / 33 spikes, spike shift 0.000 ms**; the difference trace
is a bipolar 1e-10 to 2e-6 mV pulse on each spike upstroke (float-order, amplified by dV/dt)
and flat elsewhere. Same check with every state variable: section "Full-state parity" below.

`add_contact` recompile (`fused-contact` arm, dynamic contacts): 17 cells + 1 synthetic contact,
forward compile + run 5.3 s (per-cell `mut-contact`: 39.0 s), `compile_graph` 0.83 s, learner
2-event update compile + run 37.9 s (per-cell: 358.8 s), warm calls 5.38 / 5.24 s with 0 compile
messages, 88 on the first call. A contact still changes array shapes on the dynamic path; the
static path (step 3) removes that.

## Step 2: the learner on the forest (segmented sparse layout)

`braintrace/_compiler/sparse_influence.py::SparseInfluence.build_segmented`,
`sparse_io.py` slot-table factor paths, `SparseIOGraph` hook `model.sparse_structure`,
`H01ArcModel.sparse_structure`. The block-level program analysis is unchanged (it proves which
blocks and ETP outputs interact); the model declares how each cable block splits into per-cell
segments (`forest_offsets`) and which cells each ring buffer / synapse block reads and feeds
(contact pre and post). Segments and blocks carry `(reads, feeds)` ports; a parent reaches a
child only where feeds meet reads; an output seeds only where its labels meet reads. Factors
are stored per position with a slot table (width = widest segment row), so JVP colours are the
per-cell conflict structure, not one per cell.

| Arm | Cells / compartments | Eligibility elements / bytes / colours | learner 2-event update compile + run / warm | peak device |
| --- | ---: | ---: | ---: | ---: |
| per-cell (profile campaign, uncontended) | 17 / 107,537 | 3,344,642 / 26.8 MB / 1 | 313.7-347.8 s / 3.6 s | 0.72 GB |
| fused, whole blocks (`fused-learner`) | 17 / 107,537 | 56,858,914 / 454.9 MB / 17 | 36.2 s / 3.85 s | 2.30 GB |
| fused, whole blocks, 18 cells (`fused-clone-learner`) | 18 / 109,145 | 61,089,732 / 488.7 MB / 18 | 36.3 s / 5.59 s | 2.15 GB |
| **fused, segmented (`fused-learner-seg`)** | 17 / 107,537 | **3,344,642 / 26.8 MB / 1** | 35.6 s / **1.69 s** | 0.84 GB |
| fused, segmented, 18 cells (`fused-clone-learner-seg`) | 18 / 109,145 | 3,393,874 / 27.2 MB / 1 | 38.0 s / 2.24 s | 0.85 GB |
| fused, segmented, 34 cells (`fused-cloneall-learner-seg`, every cell cloned once) | 34 / 215,074 | 6,689,284 / 53.5 MB / 1 | 58.2 s / 4.63 s | 1.66 GB |

The segmented forest layout is element-for-element the per-cell layout (one colour, 26.8 MB at
17 cells) and grows linearly with cloning (27.2 MB at 18, 53.5 MB at 34); the 512 MB
`factor_limit_bytes` that stopped the whole-block forest at 19 cells is now reached at about
325 cells (1.57 MB per cell). Warm update 1.69 s at 17 cells against 3.85 s (17 colours) and the
per-cell 3.6 s uncontended; 4.63 s at 34 cells.

Gradient parity (`examples/pp_prop/h01_arc_model_forest_test.py`): three synthetic cells with two
contacts, per-cell learner against forest learner, 2-event update, squared-logit loss, all four
parameter groups: max abs difference **1.4e-17 on `input` (max gradient 0.21), 0.0 on
`readout_weight`, `readout_bias`, `recurrent`**; layouts have identical rows and colours
((0,1,1,2,2): cell 0 alone, cells 1 and 2 each with their contact). The static-capacity model
(step 3) passes the same test against the per-cell model with its dormant slots and unused
contact rows carrying exactly zero gradient.

What the declaration is: cable states segmented per cell; contact `c`'s ring buffer reads
`{pre(c), contact c}` and feeds `{post(c), contact c}`; its synapse states read and feed
`post(c)`; every other block is unrestricted. What it excludes (two cells coupled other than
through a declared contact) is not proved by the compiler; the gradient parity test is the check.
When the contact graph changes (`add_contact`), the learner must be recompiled
(`H01ArcModel.sparse_structure_signature` names the structure); a clone into a pre-declared slot
does not change it.

## Step 3: static capacity

`H01ForestCell(cells, slots=K)`: every source cell is laid out `K` times; slot 0 is active and
the others are dormant copies (same anatomy, parameters and initial state, integrated every
substep, masked from the drive, the spike output and the readout through the `active` state).
`braintrace/datasets/h01_forest_contacts.py::ForestContactTable`: two shared BrainCell synapses
per cell at the soma (`contact_exc` 0 mV / 2 ms, `contact_inh` -80 mV / 5 ms, the two kinds
`H01Topology.add_contact` makes) and a fixed table of `capacity` rows (pre cell, target point,
kind, delay, weight, active) with one ring buffer `(depth, capacity)`. `build_network(fused=True,
slots=K, contact_capacity=C)`; `H01ArcModel.clone(parent)` and `.add_contact(pre, post, kind=...)`
write into the padding through the host (`host_write` keeps the array's device placement so the
jitted step's input signature is unchanged). Anatomical contacts must target the post soma and
be one of the two kinds; the kept manifest has none.

Unit (`h01_forest_contacts_test.py`, `h01_arc_model_forest_test.py`): static forest with 2 slots
and 4 contact rows against BrainCell per-population projections **1e-12 mV**; a dormant copy of
the undriven-by-contacts cell equals its source exactly; after `activate` + one table write the
jitted substep and the jitted ARC event run with **0 XLA compiles** and unchanged state shapes;
model forward (2 events) and gradients equal the per-cell model (dormant slots and unused rows:
zero gradient); a full slot table raises.

Box arms (fused, dt 0.005, contended):

| Arm | Cells active / laid out / compartments | forward s/event | GPU busy ms/substep | launches/substep | while loops | forward compile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fused-dt0005` (dynamic, no contacts) | 17 / 17 / 107,537 | 0.129 | 4.69 | 327 | 8 | 5.4 s |
| `static17` (slots 1, capacity 64) | 17 / 17 / 107,537 | 0.103 | 3.81 | 339 | 8 | 5.0 s |
| `static17-clone1` (slots 2, one clone + one contact written) | 18 / 34 / 215,074 | see receipts | | | | |
| `static34` (slots 2, every cell cloned) | 34 / 34 / 215,074 | 0.180 | 8.71 | 337 | 8 | 8.1 s |

The static table adds 12 launches per substep over the dynamic path (339 against 327) and
they do not grow with contacts or clones; at capacity 34 the substep integrates 215,074
compartments always, so "17 + 1 clone" costs what 34 costs (that is the price of static
shapes; capacity is a setting). `static17-mutate-after` clones one cell and writes one contact
*after* the forward program compiled: see the receipt for compiles after the mutation (0
expected) and the seconds per event after it.

## Full-state parity and the timestep (J's amendment)

`docs/evidence/h01_dt_state_check.py`: see [h01-dt-spike-window.md](h01-dt-spike-window.md)
(lockstep of every state variable at four timesteps and, in `--mode parity`, of the per-cell
model against the fused model at the pinned timestep; per-compartment difference frames,
per-site traces and plots).
