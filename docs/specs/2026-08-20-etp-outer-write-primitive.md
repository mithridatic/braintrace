# ETP outer-product write primitive (`etp_outer_write`)

Status: approved 2026-08-21 ("execute the spec"); implementation in progress
Date: 2026-08-20
Branch: `investigate/ex21-learned-memory-keys` (continues the learned-memory-coding line)
Depends on: `2026-08-20-example21-learned-memory-coding.md` (structural finding)

## Motivation

The Example 21 elimination chain leaves exactly one trainability gap between
the implemented system and BDH-CQ's learned ingestion operator `U_θ`:

1. Store capacity refuted twice (width 32→512, load K=4→8: binding perfect).
2. Retrieval-path key learning refuted: keys train hard (l2_delta 2.05) but
   ARC pairing sensitivity stays exactly ≈ 0 in both arms.
3. The **write path is outside the differentiable program**. pp-prop rejects
   any trainable operator upstream of the outer-product write
   `S_t = λS_{t-1} + g_t · (k_t ⊗ v_t) ⊙ Σ`, so *what gets stored* cannot
   carry gradient. Values are untrainable outright; keys only trained
   through retrieval.

This spec defines the Layer 1–3 feature that closes the gap: a fused ETP
primitive whose output is the write matrix itself, owning the key/value
projections as trainable invars, with per-primitive trace rules derived
below.

Regime caveat (honesty about the evidence): BDH-CQ closes this gap with
*offline* pretraining of a large `U_θ` on an ARC-style mixture — no
parameters update at inference; in-context adaptation flows through `S`
only. This program learns the write *online per task* through eligibility
traces, a regime the paper does not speak to. A null result here therefore
would not contradict BDH-CQ; the elimination-chain logic is structural, not
evidential.

References: BrainScale — Wang et al., *Model-agnostic linear-memory online
learning in spiking neural networks*, Nat. Commun. (2026),
s41467-026-68453-w (published version of bioRxiv 2024.09.24.614728v2;
source of the D-RTRL and `ε ≈ ε_f ⊗ ε_x` factorizations this spec builds
on). BDH-CQ — arXiv 2608.09888. Delta-rule / TTT write formulations —
arXiv 2603.15031 §6.1.

## Why the existing machinery rejects the write (precise statement)

Two independent guards fire, and both are correct given today's rules:

- **Position-preserving tail** (`vjp_base._assert_factorized_tails_preserve_positions`
  → `prove_position_preserving`): pp-prop stores an output-shaped trace
  `ε_f ∈ R^{B×O×n_state}` (`_mm_init_pp`) whose positions must map 1:1 onto
  hidden positions. The tail `y → cos → k ⊗ v → S` sends key position `i`
  into the whole row `S[i, :]` through a `dot_general` — "equation
  'dot_general' mixes hidden positions."
- **Non-parametric-tail invariant** (`hid_param_op._emit_no_relation_diag`):
  a weight reaching hidden state only *through another trainable ETP
  primitive* is excluded — ETP cannot decompose weight→weight→hidden
  without double-counting. This forecloses the naive fix of marking the
  outer product as its own trainable-adjacent op with the projections kept
  as separate upstream `etp_mm`s.

Conclusion: the projections and the outer product must live inside **one**
primitive, so the compiler sees a single `weights → y` arrow whose `y` is
already hidden-shaped, and the tail from `y` to `S` is genuinely
elementwise (decay, gate, scale, add).

## Design options considered

| option | shape | verdict |
|---|---|---|
| A. Gradient-enabled outer op (an `etp_elemwise`-style broadcast op), projections stay separate `etp_mm`s | y=(B,K,V) | Rejected: violates the non-parametric-tail invariant (weight→weight→hidden); would require relaxing a load-bearing compiler invariant. |
| B. **Fused primitive owning the projections** (`etp_outer_write`) | y=(B,K,V) | **Recommended.** One weights→y arrow; closed-form rules; zero changes to the position prover — y is hidden-shaped and its tail is elementwise. |
| C. Extend the position framework with "factored broadcast tails" (teach the prover that broadcast duplication is not mixing, and teach pp-prop to contract the duplicated axis) | — | Deferred: touches the prover, the executor contraction, and every algorithm's tail check; strictly more general but far larger blast radius. Reconsider if a second broadcast-tail use case appears. |
| D. TTT-style write-as-gradient-step (`W_t = W_{t−1} − η∇ℓ(W_{t−1}; x_t)`, arXiv 2603.15031 §6.1) — learn the write objective, never differentiate the Hebbian outer product | — | Rejected for this spec: replaces the S_K architecture rather than making it trainable, so it cannot answer the U_θ question; noted as an alternative memory design if the decision rule's null branch fires. |

## Primitive definition (option B)

```
y = etp_outer_write(x_key, x_value,
                    key_weight, key_bias, value_weight;
                    key_nonlinearity="cos_rff", value_nonlinearity="tanh",
                    key_scale: float)
# y[b,i,j] = key_scale * φ_k(x_key[b] @ key_weight + key_bias)_i
#                      * φ_v(x_value[b] @ value_weight)_j
# shapes: x_key (B, A_k), x_value (B, A_v),
#         key_weight (A_k, K), key_bias (K), value_weight (A_v, V),
#         y (B, K, V)
```

- Trainable invars: `{key_weight: 2, key_bias: 3, value_weight: 4}`
  (via `trainable_invars_fn`, mirroring `_mm_trainable_invars`).
- `x_invar_index`: the primitive has *two* data inputs; register `x_key` as
  the primary x-invar and carry `x_value` as a secondary input the rules
  close over (the registry's single-x assumption is per-primitive API, not
  math; the rules receive the eqn's invars). If the registry hard-requires
  one x, the fallback is passing `x = (x_key, x_value)` packed by the
  user-facing op and unpacked in `impl` — decided at implementation time,
  noted here so it is not "discovered."
- Gating, decay, `memory_write_scale`, and the side-validity masks stay
  **outside** the primitive as today's elementwise ops on `y` — they are
  position-preserving on (B,K,V) and already proven.
- `"frozen"` semantics are recovered exactly by not using the primitive at
  all (existing code path untouched, bit-exact guarantee preserved).

### Nonlinearity placement

φ_k is applied *inside* the primitive (`cos(γ·phase + b)` folded as in the
learned-keys work: γ into `key_weight` init, bias trained). Keeping φ inside
is what makes ∂y/∂W closed-form per primitive; the VJP-based `xy_to_dw`
handles it exactly (same mechanism `weight_fn` uses in `_mm_xy_to_dw`).
v1 supports exactly the two nonlinearities Example 21 needs
(`cos_rff`, `tanh`), enumerated in eqn params, each with a registered
forward — not arbitrary callables, so the jaxpr stays serializable.

## Per-primitive rules (the actual math)

Notation: `p = x_k @ W_k + b_k` (pre-activation, (B,K)),
`k = φ_k(p)`, `v = φ_v(x_v @ W_v)`, `y[b,i,j] = c·k_i·v_j`.

- **`xy_to_dw`** (instantaneous ∂h/∂W, cotangent `hidden_dim` = ∂h/∂y of
  shape (B,K,V)): one `jax.vjp` over the dict-valued fused forward, exactly
  the `_mm_xy_to_dw` pattern. Closed forms it computes:
  - ∂/∂W_k[a,i]: `c · x_k[b,a] · φ'_k(p)_i · Σ_j hidden_dim[b,i,j]·v_j`
  - ∂/∂b_k[i]:   `c · φ'_k(p)_i · Σ_j hidden_dim[b,i,j]·v_j`
  - ∂/∂W_v[a,j]: `c · x_v[b,a] · φ'_v(·)_j · Σ_i hidden_dim[b,i,j]·k_i`
- **`init_pp` — per-weight traces, NOT one shared trace.** For `etp_mm` a
  single y-shaped `ε_f` suffices because there is one weight–x pair and the
  forward is linear in x. Here the three trainable invars carry three
  *different* instantaneous postsynaptic factors:
  `D_f^{W_k} = D_f^{b_k} = c·φ'_k(p^t) ⊗ v^t` and
  `D_f^{W_v} = c·k^t ⊗ φ'_v(q^t)`. Each gets its own y-shaped filtered
  trace `ε_f ∈ R^{B×K×V×n_state}` with the factor injected **per step**
  (matching the published scheme's per-step `diag(D_f^t)` injection), plus
  two x-side traces (filtered `x_k`, filtered `x_v`; `b_k`'s x-factor is
  the filtered constant 1). Memory at Example 21 scale: ≈ 3 × 131 KB per
  state — still negligible.
- **`dt_to_t`** (pp/D-RTRL trace propagation): the hidden→y chain factor.
  The hidden group containing `S` is (B,K,V)-shaped and its transition is
  elementwise, so the executor's `hidden_dim` arrives (B,K,V)-aligned;
  `dt_to_t` is a pure elementwise multiply on the y-shaped trace — simpler
  than the matmul case (no δ-column bookkeeping).
- **`init_drtrl`** (exact D-RTRL variant): weight-shaped trace per hidden
  state. For W_k this must retain the value axis:
  `ε_{W_k} ∈ R^{B×A_k×K×V×n_state}` (and symmetrically for W_v). At
  Example 21 scale with A_k = 424: 32·424·32·32·4 B ≈ 55 MB per hidden
  state per trace — feasible but heavy; D-RTRL support is therefore
  **optional in v1** (pp-prop is the algorithm Example 21 runs). If v1
  ships pp-only, the registry entry raises `NotSupportedError` with an
  actionable message for D-RTRL, matching existing per-algorithm gaps.
- **Fast-path kernels** (`_DENSE_FAST_PATH` analogue): deferred; the
  generic VJP path first, kernels only if profiling shows need.
- **snap rules**: `snap_anchor=False` v1 (no SnAp support claimed).

### Approximation class — and why the `_mm_xy_to_dw` pattern must NOT be reused

The obvious implementation — defer everything to a solve-time VJP over the
fused forward, as `_mm_xy_to_dw` does — is **wrong for this primitive**.
The solve path (`io_dim_vjp.py:628–652`) evaluates `xy_to_dw` at the
**α-smoothed x trace**. For `etp_mm` that is exact in the factorized sense
because `y = xW` is linear in x. Here the forward is nonlinear in x, so a
deferred VJP would recompute `φ'_k(p(x̄_k))` and `φ_v(x̄_v W_v)` from
*smoothed* inputs, introducing (i) a Jensen-gap error through cos/tanh that
no existing primitive has, and (ii) — decisive for this experiment —
destruction of the within-timestep `k_t ↔ v_t` pairing correlation, because
`x_k` and `x_v` are filtered separately. Since the experiment's primary
readout is pairing sensitivity, that error mode could manufacture a false
null.

Therefore the rules inject the full per-step, per-weight y-shaped `D_f^t`
factors into the traces at trace-update time (see `init_pp` above), and the
solve-time combine contracts the filtered per-weight `ε_f` with the
filtered x-side factor **without re-deriving φ'/v from smoothed x**. The
residual approximation is then pp-prop's `ε ≈ ε_f ⊗ ε_x` factorization
itself — with one honest caveat: BrainScale justifies that rank-1 collapse
by sign-consistent pre/post quantities (spikes, conductances ≥ 0), and this
primitive's factors are sign-alternating by construction (`φ'_k = −sin`,
signed tanh values, real latents). The factorization is thus *outside its
empirically validated envelope* here, which the validation plan measures
directly rather than assumes away.

Per AGENTS.md, gradient assertions about these rules are learning-rule
properties: they MUST be measured through the finite-window oracle
(`chunked_online_param_gradients`), never the whole-sequence VJP.

## Weight sharing with the query path (design decision)

Today the *same* key encoder serves write and query. Under option B the
write-side weights live inside `etp_outer_write`; if the query path also
consumed them (via `etp_mm` or plain ops) that is mixed parameter ownership,
which the compiler rejects by design.

**v1 unties them**: `write_key_projection` (inside the primitive) and the
existing retrieval-path `memory_key_projection` are separate ParamStates,
both initialized from the same frozen basis. BDH-CQ's public description is
agnostic to tying (everything below its Eq. 1 is proprietary), and untying
is what makes the experiment cleanly answer "does a learnable write help"
independent of retrieval. Tying
(a two-output primitive emitting `y_write` and `k` for the query side) is
recorded as follow-up, contingent on v1 showing signal.

## Model integration (Example 21)

New `memory_coding="learned_write"`:
- write path: `context_memory += gate ⊙ etp_outer_write(...) ⊙ scale ⊙ decay`
  (masks/gate/scale unchanged, outside the primitive);
- query path: keeps `learned_keys` behavior (retrieval-trained
  `memory_key_projection`), so `learned_write` ⊃ `learned_keys`;
- init reproduces frozen behavior exactly at step 0 (same folded-γ trick,
  proven by the same init-equivalence test pattern);
- `associative_memory_report`: `key_map="learned_rff_cosine_write_and_retrieval"`,
  `value_map="learned_tanh_projection"`,
  `write_component_type="braintrace.outer_write"`.

## Validation plan

1. **Op unit tests** (`braintrace/_op/outer_test.py`, co-located): forward
   equivalence vs `einsum('bi,bj->bij', φ_k(...), φ_v(...))`; parameter
   registration; batching rule (no silent vmap decomposition — register the
   batched counterpart or assert the warning).
2. **Oracle**: finite-window `chunked_online_param_gradients` vs BPTT on a
   small S-carrying model, per the AGENTS.md finite-window rule; also the
   op-rule oracle harness (`op_rule_oracle.py`) which existing ops use.
   Additionally, cosine-similarity-vs-BPTT (BrainScale's own metric) on a
   *pairing-order-sensitive* sequence pair — two sequences differing only
   in demonstration pairing whose BPTT gradients differ — asserting the
   pp-prop gradient distinguishes them too. This is the direct test that
   the factorization did not erase the pairing signal.
3. **Compiler**: relation discovered with `all_direct` classification into
   the (B,K,V) hidden group; no position-prover involvement (tail is
   elementwise); mixed-ownership rejection still fires if a test shares the
   weights outside the primitive.
4. **Example 21 integration**: init-equivalence, trainability
   (write-projection l2_delta > 0 through the *write*, verified by zeroing
   the retrieval path in a diagnostic config), frozen bit-exactness
   untouched, full suite in Docker without `--gpus all`.
5. **Experiment**: smoke, then 100-task GPU pilots `frozen` vs
   `learned_keys` vs `learned_write`, evaluation controls, seed 2108.
   Primary readout stays the shuffled-demonstrations deviation. All wall
   clocks measured; no run projected past the pilot scale without approval.

Decision rule: `learned_write` moving the shuffled-demonstrations deviation
away from 0 (in either direction, beyond the repeat-intact nondeterminism
band) is the first evidence the model *can* use pairing; improvement on
intact shape/pixel then decides scale-up. If the deviation stays pinned at
0, the result is NOT immediately interpretable — the pp-prop factorization
is outside its sign-consistency envelope here (see Approximation class), so
a pinned null must first be disambiguated on a small config by an exact
path: finite-window BPTT via `chunked_online_param_gradients`, and/or the
deferred D-RTRL weight-shaped trace (its ≈55 MB/state cost is affordable at
diagnostic scale). Only if the exact path *also* shows no usable pairing
gradient does the conclusion become "the binder is not the coding," and the
next spec targets the memory *format* — where the cheapest next suspect is
the delta-rule write `S_t = (I − β k kᵀ)S_{t−1} + β k vᵀ` (targets
superposition interference directly, keeps a single S), ahead of
multi-trace or slotted designs.

## Implementation deviations (recorded 2026-08-21, before writing code)

Three facts found while grounding the design in the actual code changed it.
They are deviations from the text above, not refinements of it.

**(a) A new algorithm-layer hook is required; registry recognition is *not*
automatic.** The IO-dim algorithm exposes exactly three per-primitive hooks:
`init_pp` (trace shape), `xy_to_dw` (solve-time contraction) and
`pp_x_repr` (x-trace representation at step time). None of them can inject a
per-step factor that depends on the weights, and the required factor
`φ'_k(p^t)_i · v^t_j` is `(i,j)`-indexed so it cannot ride the x-side either.
`_contract_hidden_jacobian` forces the trace to carry the group's
`(*varshape, n_state)` layout, so contracting `j` away early is also
unavailable (it would only be valid where the hidden Jacobian happens to be
uniform over `j`, which is not a property a rule may assume).

v1 therefore adds a fourth hook, symmetric with `pp_x_repr`:

```
ETP_RULES_PP_DF_FACTORS[p] = fn(x, weights, **eqn_params) -> {name: y-shaped array}
```

invoked at trace-update time on the **raw current-step** `x` and the current
weight values, returning one multiplier per trace name; the injected
`D_f^t` is multiplied by it before smoothing. Consequences inside
`io_dim_vjp.py`: the pp df trace becomes a **dict of y-shaped arrays** for
primitives that register the hook (`jax.tree.map` makes the array case
byte-identical, so no branch is needed on the solve path), and
`_update_IO_dim_etrace_scan_fn` gains `weight_vals` — already threaded to
`_make_etrace_stepper` on both the fused-scan and single-step paths, where
this algorithm had so far declared it unused. Blast radius is bounded:
`ETP_RULES_INIT_PP` has exactly one consumer in the package.

**(b) Two factor arrays, not three.** `b_k`'s postsynaptic factor is
identical to `W_k`'s (`c·φ'_k(p^t) ⊗ v^t`); only their x-side factors differ
(`x_k` vs. the constant 1). The trace dict is therefore keyed by *factor
group* — `{'key', 'value'}` — and `xy_to_dw` maps `'key' → {key_weight,
key_bias}`, `'value' → {value_weight}`. Cost is ≈ 2 × 131 KB per state, not 3.

**(c) The "registry single-x assumption" risk resolves by packing.** The
user-facing op passes `x = concat([x_key, x_value], axis=-1)` and the rules
split it at a static `key_features` param. This is exact rather than a
workaround: the x-trace low-pass filter is linear and elementwise, so
filtering the concatenation equals concatenating the filtered halves, and
the solve path needs *both* filtered inputs anyway — which a single
registered x-invar could not have supplied.

Unchanged by these deviations: the position prover stays uninvolved (it
checks only the operands *reachable from* `y`, so the side-validity mask and
`memory_write_scale` broadcasts in the tail are never classified), and
D-RTRL remains out of scope — `init_drtrl` / `dt_to_t` raise
`NotSupportedError` rather than silently mis-tracing.

## Scope and sequencing

1. Layer 1: `pp_df_factors` registry hook, primitive + rules + op tests +
   oracle (pp-prop only).
2. Layer 4: dict-valued pp df traces and factor injection in `io_dim_vjp`;
   both `fast_solve` and legacy-vmap solve paths exercised.
3. Layer 2/3: verify compiler recognition, add the relation test.
4. The finite-window pairing-discrimination test — written **before** the
   Example 21 wiring, since it is the test that validates why the machinery
   exists.
5. Example 21 wiring behind `learned_write` + integration tests.
6. Smoke + pilots + results recorded here.

Out of scope for v1: D-RTRL weight-shaped traces (55 MB/state), SnAp,
fast-path kernels, tied write/query encoders, arbitrary nonlinearities,
`einsum`-general N-ary outer products.

## Risks

- **Registry single-x assumption**: the two-data-input shape is the main
  unknown in Layer 1 plumbing (mitigation named above).
- **vmap decomposition**: the batching warning seen for `etp_conv` shows
  unbatched primitives silently decompose under vmap and drop out of ETP —
  the primitive must be registered `batched=True` from the start.
- **Trace correctness under gating**: the write is gated per-example;
  gate=False steps must contribute exactly zero to the trace (covered by
  the oracle on a gated sequence).
- **Key-scale drift**: the folded-γ init pins the key scale only at step 0;
  once `key_weight` trains, nothing constrains it (attention-residual
  ablations show unnormalized keys let large-scale components dominate).
  v1 monitors per-step key-norm statistics in the associative diagnostics;
  an RMSNorm-style guard on `k` is the prepared mitigation, off by default
  because it changes frozen-init equivalence.
- **Null result**: pinned-at-zero pairing even with a differentiable write
  is a real possibility; the decision rule above makes that outcome
  informative rather than wasted.

## Results

### Implementation (2026-08-21)

Shipped on `investigate/ex21-learned-memory-keys`:

| commit | contents |
|---|---|
| `f01d197` | Layer 1: `etp_outer_write_p` + `outer_write`, the `pp_df_factors` registry hook, `assert_factored_rules_match_vjp` oracle helper |
| `3bdae2c` | Layer 4: factored (dict-valued) pp df traces, per-step factor injection, `weight_vals` in the stepper, scan-descent rejection |
| `276c267` | Example 21 `memory_coding="learned_write"`, `encode_memory_write`, `apply_context_memory_write` split, report fields, CLI flag |

Two compiler questions the spec left open were settled empirically, not by
reading:

- **The position prover never fires.** It classifies only the operands
  *reachable from* `y`, so the side-validity mask `(B,1,1)`, the `(K,V)`
  `memory_write_scale` broadcast and the gate's `select_n` are invisible to it.
  The intended tail compiles unchanged.
- **The write projections do reach `context_memory`.** In the real model the
  fused relation is classified `all_direct` and its hidden paths include
  `('context_memory',)` — the thing that was structurally impossible for
  `learned_keys`, where the relation deliberately excluded it.

### Measured: per-step injection vs. the deferred design

The rejected design (recompute `φ'` and the codes from the low-pass-filtered
`x` at solve time) was implemented as a temporary mutant and measured on the
same panel — three seeds × two chunk sizes, six-step sequences, cosine
alignment of the finite-window pp-prop gradient with BPTT:

| case | per-step injection | deferred VJP |
|---|---:|---:|
| worst of panel | **0.868** | 0.342 |
| mean of panel | **0.943** | 0.800 |
| worst relative deviation | **0.570** | 1.047 |

A relative deviation above 1.0 means the deferred estimate is further from
BPTT than the zero vector is. The review's Finding 1a was therefore not
theoretical: the two designs are separated by a wide margin on exactly the
quantity the experiment reads. `test_multi_step_windows_stay_aligned_with_bptt`
pins the threshold at 0.80, between the two, and was verified to fail under the
mutant.

Both designs are one-step exact (with no history the filter has averaged
nothing), so the op-level oracle alone would *not* have caught the difference —
only the finite-window panel does, as AGENTS.md's rule predicts.

### Suites and smoke (2026-08-21, all measured)

`braintrace/` 3,098 passed (27:05); `examples/pp_prop/` 2,415 passed (18:47);
both CPU-only containers, zero failures. Smoke
(`--smoke --device cpu --memory-coding learned_write`): 15.4 s, exit 0.

Smoke parameter movement — the structural headline:

| parameter | l2_delta | note |
|---|---:|---|
| `write_value_weight` | 0.1023 | **had no gradient path at all before this primitive** |
| `write_key_weight` | 0.0236 | |
| `write_key_bias` | 0.0054 | |
| `memory_write_scale` | 0.0270 | the elemwise relation was not displaced |

### GPU pilots (2026-08-21) — 100 evaluation tasks

Seed 2108, 4096 neurons / 4096 edges, width 32, batch 32, 260 updates, chunk 5,
lr 1e-3, `--evaluation-controls`, `row_refinement`. All three arms re-run on
this commit (the earlier table's frozen arm was a reused width-sweep artifact
from a different commit and is *not* comparable). Non-scientific by
construction: task cap.

| arm | shape e30 | shape e60 | pixel e30 | pixel e60 | wall |
|---|---:|---:|---:|---:|---:|
| frozen | 0.5096 | 0.5096 | 0.3807 | 0.3608 | 296 s |
| learned_keys | 0.5288 | 0.5385 | 0.3845 | 0.3703 | 357 s |
| learned_write | **0.5673** | **0.5481** | **0.3925** | 0.3688 | 349 s |

Control-minus-intact deltas (shape, effort 60). `repeat_intact` is **exactly
0.0** in every arm, so the nondeterminism band is zero and one query is 0.0096:

| control | frozen | learned_keys | learned_write |
|---|---:|---:|---:|
| no_context | −0.2115 | −0.3077 | −0.2308 |
| shuffled_demonstrations | +0.0192 | 0.0000 | +0.0096 |
| slot_ablation | 0.0000 | −0.0096 | 0.0000 |

Trainability and attribution:

| quantity | learned_keys | learned_write |
|---|---:|---:|
| `memory_key_projection` l2_delta | 2.0490 | 2.2324 |
| `write_key_weight` l2_delta | n/a | 1.7979 |
| `write_value_weight` l2_delta | n/a | 1.9282 |
| write↔retrieval key cosine | n/a | 0.99992 |
| write↔retrieval key relative l2 | n/a | 0.0124 |

### Verdict

**The mechanism works; the readout did not move.**

What the primitive was built to do, it does: the write projections train hard
(l2_delta ≈ 1.8–1.9, comparable to the retrieval projection's 2.2), the
relation is `all_direct` into `context_memory`, and the value coding is
trainable for the first time in this program.

Intact accuracy improves monotonically across the trainability ladder —
frozen 0.5096 → learned_keys 0.5288 → learned_write 0.5673 at effort 30 (six
queries out of 104, on a deterministic run) — so a learnable write is not
inert.

But the **primary readout is unmoved**. The shuffled-demonstrations deviation
is +0.0096 in `learned_write` versus +0.0192 in `frozen`: it did not grow, and
its *sign is wrong* in every arm — shuffling the demonstrations very slightly
*helps*. That is the signature of a model not using pairing at all, not of a
model whose pairing was damaged.

Two candidate explanations are now **ruled out** rather than merely unlikely:

- *The gradient does not carry pairing.* Refuted at the rule level by the
  alignment panel above: the finite-window pp-prop gradient through this
  primitive tracks BPTT at cosine 0.87–0.98 and separates pairing-permuted
  sequences. (Measured on a 2×2 toy memory, so it establishes the *rule* is
  sound, not that the ARC task's pairing signal survives at width 32.)
- *The read drifted out of the code the memory was written in* — the untying
  risk that motivated `memory_coding_divergence`. Refuted directly: after 260
  updates the write and retrieval key encoders are still at cosine 0.99992,
  relative l2 0.0124. They co-adapt rather than diverge.

Per the decision rule this is therefore *not* yet "the binder is not the
coding": what remains untested is whether the ARC model **at pilot scale** has
a usable pairing gradient, as opposed to the toy. The next step the rule
prescribes is the exact-path disambiguation on a small ARC config
(finite-window BPTT via `chunked_online_param_gradients`, and/or the deferred
D-RTRL weight-shaped trace at ≈55 MB/state, affordable at diagnostic scale).
If that also shows no usable pairing gradient, the conclusion becomes "the
binder is not the coding" and the next spec targets the memory *format*, with
the delta-rule write `S_t = (I − β k kᵀ)S_{t−1} + β k vᵀ` as the cheapest
suspect.

Artifacts: `var/example21-owpilot-{frozen,learned_keys,learned_write}-t100/`,
smoke at `var/example21-smoke-learnedwrite/` (uncommitted by var policy).
