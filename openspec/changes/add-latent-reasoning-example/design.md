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
  can drive end to end, so credit for memory-writing parameters must travel the
  full episode.
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
`K, R ≤ 16`, and silently degrades past it); `cond`-based dispatch (the
canonicalizer if-converts `cond` to `select_n` by default, so it would function,
but arithmetic masks are cheaper and produce no diagnostic surface).

### D2. Contextual memory is a hidden state, read by a plain contraction

The memory is a pair of `brainstate.HiddenState` arrays `A ∈ (n_lat, M)` and
`B ∈ (n_lat, M)`, where `M` is the slot capacity. The read is
`A @ (Bᵀ @ H_r)` — an ordinary `jnp` contraction, **not** an ETP operator,
because both operands are hidden states (constraint 2). The compiler absorbs it
into the hidden-to-hidden transition.

*Consequence, stated plainly:* the memory interaction is bilinear in hidden
state, and pp_prop's factorized trace handles it approximately. This is exactly
why the example asserts nothing about gradients — it reports whether the thing
*learns*, not whether the estimate is faithful.

*Alternative rejected:* making the memory a `ParamState` so the read becomes an
ETP op. That would make the memory a trained weight rather than inference-time
state, destroying the requirement that ingestion leaves parameters unchanged.

### D3. Slotted write via a one-hot mask

`k_t = ReLU(W_k r_t)`, `v_t = ReLU(W_v r_t)` are computed from the ingestion
population's rate. The write is
`A ← A + v_t ⊗ m_t`, `B ← B + k_t ⊗ m_t`, with `m_t` a one-hot over slots,
itself gated to zero outside the demonstration phase. This is
`S_t = S_{t-1} + v_t k_tᵀ` held factored — Eq (1)'s named linear-attention case.

*Why one-hot slotting over accumulation:* capacity becomes an explicit number,
the shuffled control becomes a column permutation that provably preserves shape
and magnitude, and overflow (`K > M`) is detectable rather than silent. Pure
arithmetic, no `dynamic_update_slice`.

A decaying-accumulation variant (`A ← λA + v_t ⊗ u_t` with a learned write
address) stays available behind a flag but is not the default.

### D4. The latent workspace is a separate population

Ingestion runs a recurrent LIF population; the latent phase runs a *distinct*
population `H` with its own recurrence `W_f`. They are not the same units.

*Why:* the comparison the example exists to make — decodability from `S_K` alone
versus from `H_R` — is vacuous if `H` and the memory-writing population are the
same state. The paper's §3.4 is explicit that `S_t` and `H_r` have different
conceptual roles; collapsing them would make the null result untestable rather
than merely likely.

### D5. The latent phase must sustain activity without input

`R` silent ticks on a population with Example 17's fast constants would decay to
zero in two or three steps and the sweep would measure nothing. The latent
population therefore gets a slow membrane constant and `W_f` initialized to a
spectral radius near one. The exact values are fixed by the spike (Task 1), not
guessed here.

### D6. Train once per latent depth, then freeze and intervene

Latent depth is architectural, so a model is trained per `R ∈ {0, 1, 2, 4, 8}`
on a mixed binding-count distribution. **All** controlled interventions —
binding count, supported vs short context, intact vs shuffled memory — are then
run against frozen models with no retraining.

*Why:* it collapses the evaluation grid from ~80 trainings to 5, and it mirrors
the source work's own protocol ("We generated fresh ARC-like tasks after freezing
the model", §6.2).

### D7. Probes decode the memory *read*, not the raw factors

The memory-only probe takes `A @ (Bᵀ q)` for the query's encoding `q` — what the
memory would return for this query — rather than flattened `A`, `B`. That is the
fair comparison against `H_R`: same dimensionality, same semantic role. Raw
factors are reported as a secondary line.

Probes are linear, fit on one disjoint episode set and scored on another, with
both counts printed. Analysis is one-shot NumPy after the run (Example 19
precedent).

### D8. File layout and budgets

| File | Budget | Holds |
|---|---|---|
| `21-latent-reasoning-in-context.py` | ~350 | CLI, arms, sweep driver, report, PNG |
| `latent_workspace_task.py` | ~200 | episode generator, oracle, supported/short split |
| `latent_workspace_model.py` | ~250 | phase-masked step, memory write/read, populations |
| `latent_workspace_analysis.py` | ~250 | participation ratio, trajectory, probes |

Each gets a co-located `*_test.py`. Example 18 reached 1,882 lines in one file;
these budgets exist so 21 does not repeat that.

### D9. Sizing defaults are set by measurement, not by guess

Symbol count `C`, slot capacity `M`, latent width `n_lat`, demonstration tick
budget `T_d`, and the encoding rate are left unfixed here and are set by the
spike (Task 1) against the gates in Risks below. They are recorded back into
this document once measured, so the design states what was chosen and why
rather than a number nobody tested.

## Risks / Trade-offs

- **The latent phase decays to silence, and the `R` sweep is flat by
  construction.** → D5, verified empirically in Task 1 before any model code is
  written. Gate: mean latent firing rate at `r = R` is at least 25 percent of
  its value at `r = 0`, at the largest swept depth.
- **The task sits at ceiling or floor across 2→8 bindings, so nothing is
  measurable.** → Task 1 also sweeps symbol count and encoding against an
  explicit gate: supported-query accuracy ≥ 0.9 at two bindings and ≤ 0.6 at
  eight. If no configuration satisfies it, the binding range is retuned and the
  spec's range updated before implementation proceeds.
- **pp_prop's factorized trace may credit the ingestion path too weakly for
  `W_k`, `W_v` to learn at all.** → Fall back to fixed random write projections,
  which is the standard construction for associative memory anyway, and report
  that the write path was not learned. Documented as an outcome, not hidden.
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
