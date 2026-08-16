## Context

See `proposal.md` — Why. Requirements are in
`specs/pp-prop-latent-reasoning/spec.md`; this document covers only how they are
met.

Three properties of the existing codebase shape the approach:

1. **`ControlFlowPolicy.scan_unroll_limit` defaults to 16.** An ETP-relevant
   inner `scan` longer than that stays opaque, and weights used inside an opaque
   control-flow equation raise `NotImplementedError`. Any nested loop over
   demonstrations or latent iterations that touches a `ParamState` is therefore
   capped or broken.
2. **ETP operators mark parameter-times-input operations.** `braintrace.matmul`,
   `lora_matmul`, and friends take a `ParamState.value` as the trainable operand
   (Example 10). A contraction between two *hidden states* is not an ETP
   operation; the compiler absorbs it into the hidden-to-hidden transition.
3. **Examples 18 and 19 set the house style** for arms, plain-English reports,
   one Agg PNG, and one-shot NumPy analysis after the run.

The source work publishes Eqs (1)–(4) and states in §3.4 and §4.1 that
dimensions, update rules, and the training recipe are proprietary. Eqs (1)–(4)
are therefore the entire implementable surface.

## Goals / Non-Goals

**Goals:**

- Instantiate the four published equations as one temporal rollout that pp_prop
  can drive end to end, so the eligible recurrent and readout parameters receive
  terminal supervision across the full episode. The release memory-write
  projections remain fixed random and are not evidence of a learned write path.
- Make the two-state separation *falsifiable* at example scale: the report must
  be able to show that the latent workspace added nothing over the memory.
- Keep every loop inside `brainstate.transform` primitives.

**Non-Goals:**

- Reproducing the source system, its benchmark score, or its cost figure.
- Any statement about pp_prop's gradient estimate — no oracle comparison, no
  bounded-deviation claim. This example measures accuracy and geometry only.
- Grid-structured or multi-token outputs. The query decodes to one symbol.

## Decisions

### D1. One flat time axis with arithmetic phase gating

An episode is a single contiguous sequence; the demonstration, query, and latent
phases are spans of it. A per-tick phase vector multiplies the sub-terms of the
step function, selecting which map is active.

*Why:* it sidesteps constraint (1) entirely — there is no inner loop to unroll,
so neither `K` nor `R` is capped at 16 — and it puts the whole episode under the
existing sequence driver, so the eligibility trace spans ingestion through
readout, which is the point of the example.

*Alternatives:* nested `brainstate.transform.scan` per phase (works only while
`K, R ≤ 16`; beyond that the scan stays opaque and the compiler rejects weights
inside it); `cond`-based dispatch (the
canonicalizer if-converts `cond` to `select_n` by default, so it would function,
but arithmetic masks are cheaper and produce no diagnostic surface).

### D2. Contextual memory is a hidden state, read by a plain contraction

The mathematical memory is a pair of factors `A, B ∈ ℝ^{n_lat × M}`, where `M`
is the slot capacity. The implementation stores their transposed, slot-major
logical views with shape `(batch, M, n_lat)` rather than allocating two separate
`HiddenState` objects. The abstract read `A @ (Bᵀ @ H_r)` is an ordinary `jnp`
contraction, **not** an ETP operator, because both operands are hidden states
(constraint 2). The compiler absorbs it into the hidden-to-hidden transition.

The compiled layout keeps those two slot-major factor views and the binary spike
workspace in one row-major `HiddenState` of physical shape
`(batch * (2M + 1), n_lat)`. The
latent voltage and pure query encoding are separate `HiddenState` objects with
that same compiler-aligned physical shape; only their workspace row is active.
Reshaping exposes logical value rows and key rows of shape
`(batch, M, n_lat)` plus logical workspace, voltage, and query views of shape
`(batch, n_lat)`. Projection inputs use the same grouped row axis as their
leading native-batch axis, so ETP dispatches to `etp_mm` without flattening rows
into features or wrapping the model in `vmap`.

*Consequence, stated plainly:* the memory interaction is bilinear in hidden
state, and pp_prop's factorized trace handles it approximately. This is exactly
why the example asserts nothing about gradient fidelity — it reports task
behavior, geometry, and movement of the eligible parameters only.

*Alternative rejected:* making the memory a `ParamState` so the read becomes an
ETP op. That would make the memory a trained weight rather than inference-time
state, destroying the requirement that ingestion leaves parameters unchanged.

### D3. Slotted write via a one-hot mask

`k_t = W_k r_t`, `v_t = W_v r_t` are computed from the ingestion population's
rate. Their four scaled tick contributions accumulate linearly; rectifying each
tick would change the aggregate projection because
`sum_t ReLU(W r_t) != ReLU(W sum_t r_t)`. The initialized aggregate code
projections are nonnegative and row-normalized by construction. The write is
`A ← A + v_t ⊗ m_t`, `B ← B + k_t ⊗ m_t`, with `m_t` a one-hot over slots,
itself gated to zero outside the demonstration phase. This is
`S_t = S_{t-1} + v_t k_tᵀ` held factored — Eq (1)'s named linear-attention case.

*Why one-hot slotting over accumulation:* capacity becomes an explicit number,
the shuffled control becomes a column permutation that provably preserves shape
and magnitude, and overflow (`K > M`) is detectable rather than silent. Pure
arithmetic, no `dynamic_update_slice`.

A decaying-accumulation variant (`A ← λA + v_t ⊗ u_t` with a learned write
address) was considered but is not implemented by this change.

The feasibility spike did not establish that this write path learned: the
brief compiled CPU run measured exactly zero movement in `W_v`. The release
default therefore selects the fixed-random-write fallback for the memory
projections. The report must say that the write path was fixed rather than
describing the passing accuracy gate as evidence that pp_prop learned the
write. A future learned-write arm is admissible only as a separately labelled
experiment with a nonzero parameter-delta check.

### D4. The latent workspace is a separate population

Ingestion runs two feedforward threshold/reset LIF banks with temporal membrane
state; the latent phase runs a *distinct* subtractive-reset LIF population `H`
with its own recurrence `W_f`. They are not the same units.

*Why:* the comparison the example exists to make — decodability from `S_K` alone
versus from `H_R` — is vacuous if `H` and the memory-writing population are the
same state. The paper's §3.4 is explicit that `S_t` and `H_r` have different
conceptual roles; collapsing them would make the null result untestable rather
than merely likely.

### D5. The latent phase must sustain activity without input

`R` silent ticks on a population with Example 17's fast constants would decay to
zero in two or three steps and the sweep would measure nothing. The latent
population therefore uses a `160 ms` membrane constant and `W_f` initialized to
spectral radius `0.9`. The feasibility spike retained `0.751055` of the `r = 0`
firing rate at the largest swept depth. The corrected production
subtractive-reset LIF measured `r0 = 0.353027`, `r8 = 0.279785`, retention
`0.792531`, and achieved recurrent spectral radius `0.9000002`, again clearing
the `0.25` gate. This result required the coupled update named by Eq. (3):
recurrent drive and the contextual memory read remain active together at every
latent tick. A recurrence-only, decoupled latent rollout is not the measured
design.

### D6. Train once per latent depth, then freeze and intervene

Latent depth is architectural, so a model is trained per `R ∈ {0, 1, 2, 4, 8}`
on a mixed binding-count distribution. **All** controlled interventions —
binding count, supported vs short context, intact vs shuffled memory — are then
run against frozen models with no retraining.

*Why:* it collapses the evaluation grid from ~80 trainings to 5, and it mirrors
the source work's own protocol ("We generated fresh ARC-like tasks after freezing
the model", §6.2).

### D7. Probes decode the memory *read*, not the raw factors

The memory-only probe takes `A @ (Bᵀ q)` for the accumulated query encoding
`q`, computed after the query phase with no workspace feedback, rather than
flattened `A`, `B`. That is the fair comparison against `H_R`: same
dimensionality, same semantic role. Raw factors are reported as a secondary
line.

The output-only `R = 0` classifier also decodes this analog pure contextual
read after incorporating the final query tick. It does **not** replace the
recurrent state: `SequenceResult.workspace` records the binary subtractive-reset
LIF `H_0`, and every workspace-geometry measurement uses that binary `H_0` and
binary `H_1` onward.

Probes are linear, fit on one disjoint episode set and scored on another, with
both counts printed. Analysis is one-shot NumPy after the run (Example 19
precedent).

To keep analysis storage linear in episodes and latent width and avoid dense
width-square factorizations, ridge probes use a matrix-free conjugate-gradient
solve over centered features. Each iteration is linear in the feature matrix; it
stops on a residual tolerance under a dimension-scaled iteration limit capped at
2,048. Participation ratio avoids an `n_lat × n_lat` covariance: through width
16 it uses a complete 16-row Walsh/Rademacher design and is exact; above width
16 it uses 16 fixed SplitMix64-derived Rademacher probes as a deterministic
Hutchinson estimate with a nontrivial nullspace. The structured geometry result
records the method, probe count, exact-width threshold, and limitation.

### D8. File layout and budgets

| File | Budget | Holds |
|---|---|---|
| `21-latent-reasoning-in-context.py` | ~350 | CLI, arms, sweep driver, report, PNG |
| `latent_workspace_task.py` | ~200 | episode generator, oracle, supported/short split |
| `latent_workspace_model.py` | ~250 | phase-masked step, memory write/read, populations |
| `latent_workspace_analysis.py` | ~250 | participation ratio, trajectory, probes |

Each gets a co-located `*_test.py`. Example 18 reached 1,882 lines in one file;
these budgets exist so 21 does not repeat that.

These were soft decomposition targets, not acceptance limits. The implemented
source snapshot exceeds all four: the entry point is 1,191 lines, the task module
565, the model module 810, and the analysis module 748. No individual file
repeats Example 18's 1,882-line monolith, but the overage is recorded rather than
presented as budget compliance.

### D9. Sizing defaults are set by measurement, not by guess

The feasibility spike selected these defaults:

| Quantity | Default | Measured reason |
|---|---:|---|
| symbol count `C` | `10` | leaves two symbols outside the largest eight-binding context |
| symbol code width | `24` | produced an unambiguous code at the selected rate |
| encoding probability | `0.25` per tick | BrainState Bernoulli draw at `codebook_seed=313320`; realized rate `0.248958333`, 10/10 unique symbols, augmented rank 10, minimum Hamming distance 29/96 |
| demonstration budget `T_d` | `4` ticks | key and value are presented in parallel on each tick |
| memory slots `M` | `8` | represents every binding in the normative two-through-eight sweep |
| latent width `n_lat` | `32` | measured recommendation from the compiled spike |
| key/value projection width | `32` | matches the latent workspace and memory-factor row width |

With the untrained fixed-write mechanism, supported-query accuracy for
`K = 2..8` was respectively `0.989532`, `0.622009`, `0.391968`, `0.280212`,
`0.217133`, `0.169830`, and `0.142273`. The monotone curve clears the capacity
gate at both endpoints without changing the normative binding range. A brief
compiled CPU training gate then measured `0.9060` at `K = 2` and `0.5015` at
`K = 8`. These are feasibility-skeleton measurements only: the corrected
subtractive-reset production LIF does not inherit those accuracy scores.

**Corrected-production baseline requalification — failed.** On an RTX 3080 Ti
with `width=32`, `batch=4`, `M=8`, `R=8`, four symbol ticks, eight terminal
pp_prop updates, and 512 fresh supported evaluation episodes per endpoint,
`K = 2` was `0.08984375` both before and after training, failing the `0.9` gate;
`K = 8` was `0.099609375` both before and after training, clearing the `0.6`
upper gate. The model predicted
class 9 for every evaluation episode and its terminal workspace saturated near
`0.95`. `W_k` and `W_v` had exactly zero delta as required by fixed-random mode;
`W_f` and `W_o` had L2 deltas `0.0008848` and `0.0007841`, respectively, with
no compiler warnings. This baseline does not qualify release.

**Corrected-production `R = 0` requalification — passed.** A single model on an
NVIDIA GeForce RTX 3080 Ti Laptop GPU (`cuda:0`, JAX `gpu` backend), using
Bernoulli `codebook_seed=313320`, `projection_seed=210848`, recurrent
`seed=2108`, `width=32`, `batch=4`, and `M=8`, received eight terminal pp_prop
updates. On 512 fresh supported held-out episodes per endpoint, `K = 2`
accuracy was `0.994140625` both before and after training, and `K = 8` accuracy
was `0.1640625` both before and after training. The io-factorized/coupled
compile completed with no warning or error diagnostics. Fixed `W_k` and `W_v`
had exactly zero delta; `W_f` also had zero delta at zero latent depth, while
`W_o` moved by L2 delta `0.000603494`.

At `R = 0`, query-terminal logits decode the analog pure contextual read
`A @ (Bᵀ @ q_next)` after the final query tick. That output is not the
workspace `H_0`: the recorded workspace and all `H_0..H_R` geometry remain
binary subtractive-reset LIF states. With the same trained parameters and one
actual LIF latent tick, terminal `K = 2` and `K = 8` accuracies fell to
`0.203125` and `0.103516`, respectively. Latent iteration therefore degraded
this endpoint result; the example reports that outcome without claiming that
iteration helped.

The CLI defaults to `--device gpu` and fails closed when no GPU is visible. Full
runs use the repository's CUDA-enabled Docker environment operationally, but the
CLI enforces the requested JAX platform rather than detecting the container.
The first run is expected to pay XLA compilation cost; the larger
neuron/synapse regime is the reason for keeping GPU as the full-run default.
Fast tests select CPU explicitly rather than inheriting whichever accelerator
happens to be visible. At the native default `(batch=4, M=8, n_lat=32)`, the
coupled hidden group has three `(68, 32)` state members and materializes
`42,614,784` float32 Jacobian elements (`170,459,136` bytes). The model therefore
passes the explicit `1 << 26` element ceiling to the compiler.

## Risks / Trade-offs

- **The latent phase decays to silence, and the `R` sweep is flat by
  construction.** → D5, verified empirically in Task 1 before any model code is
  written. Gate: mean latent firing rate at `r = R` is at least 25 percent of
  its value at `r = 0`, at the largest swept depth.
- **The task sits at ceiling or floor across 2→8 bindings, so nothing is
  measurable.** → Task 1 swept symbol count and encoding against the explicit
  gate: supported-query accuracy ≥ 0.9 at two bindings and ≤ 0.6 at eight. The
  feasibility skeleton passed, the original corrected-production `R = 8`
  baseline failed at two bindings, and the corrected `R = 0` query-terminal
  path passed both endpoints on fresh GPU data. The one-tick LIF result degraded
  sharply, so it is retained as a reported finding rather than hidden behind the
  passing no-iteration control.
- **pp_prop's factorized trace credited the ingestion path too weakly to show
  learned value writing in the spike.** → `W_v` movement was exactly zero, so
  D3's fixed random write projections are the release default. The trained CPU
  accuracy gate is reported alongside that parameter-delta result and is not
  presented as evidence that the write path learned.
- **The two-state separation may buy nothing at this scale.** → This is a
  designed-for outcome, not a failure: the spec requires the comparison be
  reported plainly either way.
- **Example-gate runtime.** → `--smoke` exercises every phase, arm, and
  measurement at reduced size; the full sweep is opt-in.

## Migration Plan

Additive; nothing to roll back but the files themselves. Order: spike →
task module → model module → analysis module → entry point → README rows →
`docs/specs/2026-08-16-pp-prop-latent-reasoning.md`. Each module lands with its
co-located tests passing before the next begins.

## Open Questions

- Whether to add an optional BPTT baseline arm (Examples 12 and 14 precedent).
  Deferrable: it changes no requirement, no module boundary, and no task in the
  breakdown — it would be one more arm behind a flag.
