# Example 21 — Neuron types for the latent workspace (staged design)

Date: 2026-08-21
Status: Piece 1 (survey + design) — awaiting Stage A implementation
Branch: `feat/ex21-neuron-types`

## 1. Problem

Example 21's substrate (`examples/pp_prop/latent_workspace_model.py`) is a
single homogeneous LIF population:

- One `brainpy.state.LIF` group of `neuron_count` neurons with identical
  `tau`, `V_th`, `V_reset`, `V_rest` (`LatentWorkspaceModel.__init__`,
  line ~1726).
- Recurrent wiring from `build_sparse_topology` (line ~981): exact-edge,
  no-self, seeded-random topology whose values are i.i.d. signed Gaussians
  (`randn * recurrent_gain / sqrt(mean_degree)`). A single neuron therefore
  emits a mixture of positive and negative outgoing weights, and training can
  flip any sign freely.
- No cell types, no E/I split, no Dale's law, no per-type connection
  statistics.

Biological recurrent circuits are strongly typed. The question this line of
work answers piece by piece: does giving the workspace even the coarsest type
structure (a Dale-constrained E/I split) change trainability, spike-rate
statistics, or stability — and is a deeper type taxonomy worth pursuing?

## 2. Survey findings

### 2.1 Existing E/I machinery in braintrace (`04-neurons-coba-ei-rsnn.py`)

`examples/pp_prop/_shared.py::COBAEICell` (line 259) is the house pattern:

- Dense `braintrace.nn.Linear` over `[input ‖ recurrent]` rows.
- **Soft Dale, enforced by initialisation only**: the first `n_exc` recurrent
  rows are `abs(KaimingNormal)`, the last `n_inh` rows are `-abs(...)`.
  The example's module docstring claims pp_prop "trains the full recurrent
  matrix without violating the sign constraint because the signs are fixed by
  the initialiser, not by the gradient step" — this is only true for small
  steps; nothing projects the weights back, so signs *can* cross zero under
  enough training. Ex21's Stage A should be honest about this and enforce the
  constraint explicitly (§4.2).
- 75/25 E/I ratio in the example (`n_exc=48, n_inh=16`); despite the "COBA"
  name it drives a CUBA LIF (`out=brainpy.state.CUBA(scale=1.)`), i.e. the
  E/I structure lives entirely in the weight signs, not in conductance
  reversal potentials. That maps directly onto Ex21's CUBA wiring.

### 2.2 Human-connectome schema vocabulary (`../config/*.yaml`)

The parent repo's schemas give a ready-made, versioned type vocabulary:

- `human_connectome_vocabularies.schema.yaml`
  - `physiological_effect`: `excitatory | inhibitory | modulatory | mixed |
    context_dependent | no_effect_detected | unknown` — the natural Stage A
    axis (binary E/I is its two-value projection).
  - `broad_chemical_phenotype`: `glutamatergic | gabaergic | glycinergic |
    cholinergic | dopaminergic | ...` — a Stage C refinement axis.
  - `intrinsic_firing_pattern`: `regular_spiking | fast_spiking | bursting |
    adapting | ...` — maps to per-type neuron parameters (tau, adaptation),
    a Stage B/C physical axis.
  - `major_cell_class`, `neuron_projection_extent`, `taxonomy_level`,
    `connection_kind`, `directionality`.
- `human_connectome_annotation_schema.yaml` additionally enumerates
  `neuron_subclass_axes`: cortical excitatory families (IT/ET/CT/near-
  projecting/L4/L6b) and inhibitory families (pvalb incl. chandelier, sst,
  vip, lamp5/neurogliaform, basket forms), plus per-region principal types.
- `human_connectome_record.schema.yaml` shows how a cell record carries
  `major_cell_class` (required) and optional phenotype/firing-pattern fields.

Stage C hooks should speak this vocabulary (`physiological_effect`,
`broad_chemical_phenotype`, `intrinsic_firing_pattern`) rather than inventing
a parallel one.

### 2.3 What has to change in Example 21

The recurrent path is
`spikes -> braintrace.nn.SparseLinear(CSR) -> Expon -> CUBA -> LIF`:

- `braintrace.nn.SparseLinear.update` (braintrace/nn/_linear.py:204) computes
  `y = x @ CSR` via the ETP `sparse_matmul` (braintrace/_op/sparse.py:528).
  With `x` = spike vector, the CSR **row index is contracted with the input**,
  so the CSR row is the *presynaptic* neuron in the computation. The CSR is
  assembled in `_topology_to_csr` with `indptr` from `topology.rows` and
  column indices `topology.columns`.
  **Caution**: the `SparseTopology` docstring calls `rows, columns` "post- and
  presynaptic endpoint arrays" — opposite to the executed contraction. Dale's
  law must be applied along the executed presynaptic axis, i.e. per
  `topology.rows` value (conveniently, edges are lexsorted by `rows`, so a
  per-presynaptic-neuron constraint is a contiguous per-CSR-row constraint on
  the nnz vector). Stage A tests must pin this orientation behaviourally
  (perturb one presynaptic neuron's type, observe all its outgoing edges).
- Edge values come from `build_sparse_topology` as one i.i.d. Gaussian vector;
  typing changes only the sign pattern (Stage A) and later the per-type
  degree/gain statistics (Stage B).
- The trainable recurrent parameter is
  `model.rec_syn.comm.weight.value["weight"]` (an nnz-vector with unit mA).
  The optimizer step lives inside the compiled `train_one`
  (`21-latent-reasoning-in-context.py::_train_model`, line ~2468:
  `optimizer.update(clip_grad_norm(...))`) — a post-step projection must be
  inserted *inside the traced function* right after `optimizer.update`, and
  equally inside the task-local adaptation path if it is ever combined with
  typing (out of scope for Stage A; `ei_dale` + `task_local_adaptation` will
  be rejected or simply left unprojected-and-documented — decision: **reject**
  to fail closed).
- Feed-forward input weights, readout heads, and the memory projections are
  *not* population-recurrent synapses; Dale's law does not constrain them in
  Stage A (documented scope).
- Entry-point surface: `ExperimentConfig` (+ `smoke_config`, `to_dict`),
  `_parser`/`_config_from_args`, `_model_config`, and the `model_report`
  assembly in `run_experiment` (line ~6161) for type counts.
- The entry point hashes its own five sources (`_implementation_report`,
  line ~5302): never edit them while a run is in flight.

## 3. Staged design

### Stage A — binary E/I split with Dale's law (this branch)

**Knob.** `ModelConfig.neuron_typing: Literal["none", "ei_dale"] = "none"`,
`ModelConfig.excitatory_fraction: float = 0.8` (validated in `(0, 1)`,
meaningful only under `ei_dale`; supplying a non-default fraction with
`neuron_typing="none"` is an error — fail closed). Mirrored by
`ExperimentConfig.neuron_typing` + CLI `--neuron-typing {none,ei_dale}` and
`--excitatory-fraction`.

**Type assignment.** Deterministic from the seed: draw a permutation with
`brainstate.random.RandomState(seed + 7)` (a fresh, documented stream so the
existing `seed`/`seed+1`/`seed+101..105` streams stay untouched) and mark the
first `round(excitatory_fraction * neuron_count)` neurons of the permutation
excitatory, the rest inhibitory. Assignment is exposed as an int8
`+1/-1` per-neuron vector on the model (`neuron_type_signs`).

**Sign constraint (init).** In `build_sparse_topology` output post-processing
(a separate pure function, `apply_dale_signs(topology, type_signs)` — the
builder itself stays untyped so its contract and tests are untouched):
`values <- sign[rows] * abs(values)`. Presynaptic = `rows` per §2.3.

**Sign constraint (training).** Projection after the optimizer step, inside
the compiled update: `w <- where(sign_edge > 0, max(w, 0), min(w, 0))`
elementwise on the nnz vector, with `sign_edge = type_signs[topology.rows]`
staged as a constant. Choice rationale, documented here once:

- *Projection (chosen)*: keeps the optimizer unmodified (Muon/AdamW/Adam all
  work), keeps the parameter in its natural coordinates, is idempotent, and
  its only cost is one `where` per update. A clipped weight sits at exactly 0
  and can re-grow in the legal direction only.
- *Reparameterisation* (`w = sign * softplus(theta)` via `sparse_matmul`'s
  `weight_fn`, which auto-composes the derivative): mathematically cleaner
  (no projection bias) but changes the loss landscape, interacts with the
  eligibility-trace coordinate, and makes `"none"`-mode bit-exactness harder
  to argue. Recorded as a Stage B/C candidate if projection shows optimization
  pathology (e.g. mass pile-up at 0).

**Reporting.** `model_report["neuron_typing"]` = mode, E/I counts, realized
excitatory fraction, count of edges clamped at init (sign flips), and a
recomputed post-training Dale-violation count (must be 0).

**Back-compat.** Default `"none"` must remain bit-exact with today's model:
pinned by a test comparing topology values, parameter snapshot digests, and a
short `run_sequence` output between a `"none"` model and a control built from
the current code path (i.e. the `"none"` branch must not consume any extra
random draws — the `seed + 7` stream is only created under `ei_dale`).

### Stage B — per-type connection statistics (not in this branch)

Replace the single `recurrent_gain` with a 2x2 (later KxK) block matrix of
gains and connection densities (E->E, E->I, I->E, I->I), defaulting to values
that reproduce Stage A exactly when uniform. Optionally per-type `tau`/`V_th`
(`intrinsic_firing_pattern` axis: fast-spiking inhibitory tau < regular-
spiking excitatory tau). Requires extending `build_sparse_topology` to
per-block edge budgets while keeping the exact-edge-count contract.

### Stage C — schema-derived taxonomy hooks (not in this branch)

A `NeuronTypeTaxonomy` value object loaded from a small YAML speaking the
parent schemas' vocabulary (`physiological_effect`,
`broad_chemical_phenotype`, `intrinsic_firing_pattern`, fractions, block
statistics), validated against the vocabularies file. `ei_dale` becomes the
built-in two-row taxonomy. Only worth building if Stage A/B measurements show
type structure moving the needle.

## 4. Stage A acceptance tests (TDD targets)

1. Default pin: `neuron_typing="none"` model equals current model bit-exactly
   (topology values, parameter digest, short-run outputs).
2. Sign orientation: under `ei_dale`, for every edge,
   `sign(value) == type_signs[rows]` at init (zero values allowed only if
   drawn exactly zero — measure count).
3. Determinism: same seed -> same type vector; different seed -> different
   (statistically) vector; realized E count == `round(fraction * N)`.
4. Projection: after a synthetic optimizer step that pushes weights across
   zero, the projected vector has no violations and legal weights are
   untouched.
5. Config validation: bad mode string, fraction outside (0,1), fraction
   supplied with `"none"`, `ei_dale` + `task_local_adaptation` rejected.
6. Report: type counts and zero post-training violations appear in
   `model_report` (smoke-scale integration).

## 5. Piece 3 experiment plan (CPU smoke only)

Docker image `braintrace-gpu:0.11.0-py314`, `--device cpu`, smoke dataset,
256–512 neurons, few updates; arms `none` vs `ei_dale` at the same seed.
Measure training-loss trajectory, spike-rate mean/std (and E vs I rates),
clamp counts per update, NaN/instability flags. Short runs only; quote only
measured durations. GPU is reserved by another agent — no GPU runs.

## 6. Open questions

- Does projection-at-zero freeze a large weight mass at 0 under Muon's
  orthogonalized updates? (Measure clamp counts per update in Piece 3.)
- Should the feed-forward input projection eventually be typed too
  (biologically it is glutamatergic, i.e. nonnegative)? Out of scope until
  recurrent typing shows an effect.
- `SparseTopology` docstring's rows/columns naming contradicts the executed
  orientation — fix the docstring in Stage A or leave for a separate cleanup?
  (Stage A will fix the docstring; it is one of the hashed files, so only
  between runs.)

## 7. Measured results

To be filled by Piece 3/4.
