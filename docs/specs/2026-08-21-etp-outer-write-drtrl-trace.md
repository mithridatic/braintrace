# Exact D-RTRL trace for `etp_outer_write` + pairing-panel re-measurement

Status: approved 2026-08-21 ("Spec and execute the next experiment/implementation work"); in progress
Date: 2026-08-21
Branch: `investigate/ex21-learned-memory-keys` (worktree `.worktrees/ex21-learned-keys`)
Depends on: `2026-08-21-example21-pairing-gradient-disambiguation.md` (verdict),
`2026-08-20-etp-outer-write-primitive.md` (the primitive)

## Question

The disambiguation probe ended on row two of its reading table: the exact
gradient on the write projections responds strongly to demonstration pairing
(0.43–1.47), but pp-prop's online estimate is 2.5–11× too small with an
unreliable pairing-specific direction (−0.15..+0.72), and trace decay is not
the cause. The remaining suspect is pp-prop's rank-1 collapse
:math:`\epsilon \approx \epsilon_f \otimes \epsilon_x`.

This spec asks the follow-up question the verdict demands: **does an exact
eligibility trace restore the pairing gradient on the write projections?**

- Yes → the finding is confirmed: this primitive needs a per-parameter
  (D-RTRL-class) trace rather than pp-prop, and the ARC question gets re-asked
  on a rule that can see the signal.
- No → the deficit is not the trace factorization either; the next suspect
  (the hidden-group diagonal-Jacobian approximation shared by both
  algorithms, or the model/optimization itself) takes over.

## Why D-RTRL is the right instrument (paper check, 2026-08-21)

Wang et al. 2026 (`docs/s41467-026-68453-w.pdf`, the pp-prop paper) confirms
both halves of the reasoning:

- **D-RTRL's only approximation is the diagonal hidden Jacobian** (Eq. 4:
  :math:`\epsilon^t_D = \mathbf{D}^t \epsilon^{t-1}_D +
  \operatorname{diag}(\mathbf{D}_f^t) \otimes \mathbf{x}^t`). For the
  associative-memory hidden group the recurrence *is* elementwise
  (:math:`S_t = \lambda S_{t-1} + g_t \, y_t \odot \Sigma`), so the diagonal
  is not an approximation there: a D-RTRL trace for `etp_outer_write` is
  **exact on the memory group**, and on the toy memory net (whose only hidden
  state is the memory) D-RTRL must equal BPTT element-wise — a hard gate, not
  a cosine panel.
- **The rank-1 collapse is proven only under sign-consistent inputs** (p. 4:
  "Since inputs x^t maintain consistent signs across time steps … we proved
  that the sum of multiple rank-one matrices can be approximated as a single
  rank-one matrix between summed vectors (Supplementary Note D)"). The
  outer-write factors alternate sign by construction
  (:math:`\varphi'_k = -\sin`, signed :math:`\tanh`), so pp-prop's measured
  failure on this primitive is not an anomaly — it is outside the premise of
  the paper's own proof. That makes "exact trace restores the signal" the
  falsifiable prediction, and D-RTRL the instrument that tests it without
  relying on the violated premise.

## Design — Layer 1 only

`braintrace/_op/outer.py` currently registers `init_drtrl` / `dt_to_t` as loud
`NotSupportedError`. This spec replaces them with real rules. No compiler or
algorithm change: `param_dim_vjp.py` already carries the exact hook pair for
primitives whose **trace structure differs from their parameter structure** —
`ETP_RULES_INSTANT_DRTRL` / `ETP_RULES_SOLVE_DRTRL`, built for LoRA's
effective-weight trace and registered LoRA-style by direct dict assignment.
`_chunk_supported` already refuses chunk factorization for any relation with
these rules, so the per-step scan fallback engages automatically.

### The trace retains the position axes

A dense-style parameter-shaped trace cannot be exact here: weight entry
:math:`W_k[a, i]` influences the whole memory row :math:`S[i, :]`, not one
position. The trace therefore keeps both memory axes:

| entry | shape | meaning |
|---|---|---|
| `key_weight` | `(B, A_k, K, V, S)` | :math:`\partial S_{ij} / \partial W_k[a, i]` slots, per state |
| `key_bias` | `(B, K, V, S)` | same with the presynaptic factor ≡ 1 |
| `value_weight` | `(B, A_v, K, V, S)` | :math:`\partial S_{ij} / \partial W_v[a, j]` slots |

This is the ≈`B·(A_k+A_v+1)·K·V·S` cost the outer-write spec priced at ~55 MB
per state at Example 21 scale and trivial at rung scale. Only rung scale runs
here.

### The four rules

- **`init_drtrl`** — allocate the dict above, zeros, dtype via
  `jnp.result_type` of the participating avals (dense precedent).
- **`dt_to_t(hidden_dim, trace)`** — the recurrent :math:`y \to` position
  link. Because the trace keeps full `(K, V)` position axes, the link is an
  elementwise broadcast: weight entries multiply by
  `hidden_dim[..., None, :, :]` (insert the input-feature axis at −3),
  `key_bias` multiplies directly. Valid in both executor contexts (batched
  trace update; batch-stripped solve), same as dense's `axis=-2` trick.
- **`instant_drtrl(x, df, weights)`** — the exact instantaneous term, on
  **batch-free, state-free slices** (the algorithm vmaps both axes):
  `x : (A_k+A_v,)` raw current-step packed input, `df : (K, V)`. Reuses the
  primitive's `_codes` at the raw `x` — no filtered-argument evaluation
  anywhere:

  ```
  key_factor   = key_slope ⊗ value_code          # (K, V)
  value_factor = key_code  ⊗ value_slope         # (K, V)
  key_weight   : x_key  ⊗ (df ∘ key_factor)      # (A_k, K, V)
  key_bias     :          df ∘ key_factor        # (K, V)
  value_weight : x_value ⊗ (df ∘ value_factor)   # (A_v, K, V)
  ```
- **`solve_drtrl(dg_hidden, trace, weights)`** — contract the learning signal
  with each trace, reducing the axis the parameter does not have
  (batch-free, state-free slices):

  ```
  key_weight   : einsum('ij,aij->ai', dg, ε_kw)
  key_bias     : einsum('ij,ij->i',   dg, ε_kb)
  value_weight : einsum('ij,aij->aj', dg, ε_vw)
  ```

Composition claim: `instant` + `dt_to_t` + `solve` implement Eq. 4 with the
diagonal restricted to positions the memory recurrence actually couples
(none), so on a model whose only hidden state is the memory the total is
BPTT, element-wise, at any window length. That is the test.

pp-prop is untouched: `io_dim_vjp.py` never reads `dt_to_t`, and the raising
rules were provably never called on the pp path (the pilots ran).

### Tests (TDD, co-located)

`braintrace/_op/outer_test.py`:
1. registration — `init_drtrl` returns the dict above (currently raises →
   RED), instant/solve rules present in their registries;
2. one-step exactness — instant term contracted by solve equals the direct
   `jax.vjp` of the impl (oracle-style, all three parameters);
3. `dt_to_t` broadcast shapes in both executor contexts.

`braintrace/_algorithm/param_dim_vjp_test.py`:
4. **the gate** — on `_OuterWriteMemoryNet`, finite-window D-RTRL
   (`chunked_online_param_gradients`, chunk sizes 2 and 3, multiple seeds)
   equals BPTT **element-wise** (allclose, not a cosine floor). The
   finite-window path is required by the AGENTS.md oracle rule; for an exact
   algorithm it must pass at every chunk size.
5. pairing discrimination — D-RTRL separates `_pairing_permuted` sequences
   exactly as BPTT does (delta alignment ≈ 1, response ratio ≈ 1).

## Measurement

Extend `var/pairing_gradient_probe.py` with a D-RTRL online arm
(`braintrace.D_RTRL(candidate, vjp_method='multi-step')`) next to the pp arm,
same five rungs, same seed, same sequences, same scopes and derived numbers.
Reported side by side with the recorded pp numbers. Same budget discipline:
wall clock measured per rung, stop at ~2 minutes per rung or when the answer
is unambiguous; no projection past a measured rung.

### Reading the result (write scope)

| D-RTRL baseline ratio | D-RTRL delta alignment | conclusion |
|---|---|---|
| ≈ 1 | ≈ 1 | Exact trace restores the pairing gradient. pp-prop's rank-1 collapse is the confirmed mechanism; next spec puts `learned_write` under a per-parameter trace (or hybrid) and re-asks ARC. |
| ≈ 1 | < 1 noticeably | The magnitude comes back but the pairing direction does not: the loss half is exact, so this would implicate the *learning-signal* path (multi-step VJP window), not the trace — investigate before touching the model. |
| ≪ 1 | any | Something other than the factorization also attenuates the write gradient (diagonal-Jacobian coupling across groups on the real model). The rank-1 story is incomplete; the diagonal approximation gets its own probe. |

On the toy net the gate (test 4) already pins the top row; the ladder asks
whether it survives contact with `LatentWorkspaceModel`, whose *other* groups
keep the diagonal approximation.

## Scope

- No delta-rule write, no memory-format change (refuted as next step).
- No ARC training run in this spec — measurement first; a training arm under
  D-RTRL is its own decision after the panel is read.
- No fast-path/chunk kernels for the new rules — diagnostic-scale primitive,
  legacy vmap path is the point.
- Probe driver stays in `var/` (gitignored); this spec plus recorded numbers
  are the substance.

## Results (2026-08-21, all measured)

### Toy-scale gate

All 8 new tests RED for the intended reasons (raising rules / absent registry
entries), then GREEN after implementation; the 23 pre-existing tests in the
same selection stayed green throughout. The gate itself: on
`OuterWriteMemoryNet`, finite-window D-RTRL reproduces BPTT **element-wise**
(`assert_param_gradients_close`, atol 1e-4) at chunk sizes 2 and 3 across
seeds 21/31/41 — including the pairing-specific gradient component, asserted
element-wise rather than directionally. pp-prop on the identical model and
seeds holds only a 0.80 cosine floor.

### Ladder — same five rungs, same seed and sequences as the disambiguation

Both online arms per rung, ~23–25 s per rung (48.7 s for rung 0 with
compilation), whole ladder under 2.5 minutes on CPU. `noise_floor` exactly
`0.0` everywhere. The pp arm **reproduced yesterday's recorded numbers to the
digit** (e.g. write baseline 0.6382/0.0929 at rung 0, 0.7317/0.2353 at rung
4) — the D-RTRL addition did not perturb the pp path, byte-identically as
designed.

Write scope (the informative one), D-RTRL vs pp-prop:

| rung | exact resp | online resp (D / pp) | delta align (D / pp) | baseline align (D / pp) | \|O\|/\|E\| (D / pp) |
|---|---:|---:|---:|---:|---:|
| n64 w2 d2 | 1.4684 | **1.4607** / 0.7592 | **0.9989** / −0.1309 | **0.9992** / 0.6382 | **0.571** / 0.093 |
| n64 w4 d4 | 1.0358 | **1.0343** / 0.9549 | **0.9992** / −0.1531 | **0.9985** / 0.7878 | **0.573** / 0.103 |
| n128 w8 d4 | 0.7572 | **0.7740** / 0.4503 | **0.9959** / 0.6013 | **0.9967** / 0.8075 | **0.555** / 0.396 |
| n256 w8 d6 | 0.4330 | **0.4291** / 0.5462 | **0.9945** / 0.7208 | **0.9992** / 0.8613 | **0.542** / 0.168 |
| n256 w16 d8 | 0.6400 | **0.6380** / 0.1828 | **0.9977** / 0.1338 | **0.9990** / 0.7317 | **0.540** / 0.235 |

The whole-parameter scope stays a plumbing check and now reads ≥ 0.9999996
alignment / ratio 1.000000 under D-RTRL — even the non-ETP-dominated norm
tightens.

## Verdict

Top row of the reading table, with one qualification. The exact trace
restores everything pp-prop was losing:

1. **Direction, fully.** Baseline alignment 0.9985–0.9992 and pairing-delta
   alignment 0.9945–0.9992 at every rung — against pp's −0.15..+0.72
   pairing direction. The trace factorization was the mechanism destroying
   the pairing signal. **pp-prop's rank-1 collapse is confirmed as the
   binding constraint**, consistent with the paper's own premise: the
   collapse is proven only for sign-consistent inputs, which these factors
   are not.
2. **Pairing response, fully.** `online_response` tracks `exact_response` to
   within 2% at every rung (e.g. 1.4607 vs 1.4684) — the online rule now
   *feels* a pairing change exactly as hard as BPTT does.
3. **The qualification: a uniform ×0.54–0.57 magnitude scale remains**,
   flat across a 4× model-size range and direction-preserving (alignment
   0.999 at ratio 0.55). This is not the rank-1 pathology — it is consistent
   with credit paths the *shared* per-group diagonal machinery cannot carry
   on the full model (memory → substrate → later loss, cross-group), which
   pp-prop also lacks. Being uniform and direction-preserving, it acts like
   a smaller effective learning rate on the write group — compensable —
   where pp's error was directional — not compensable at any learning rate.

Practical consequence for Example 21: under pp-prop the write projections
received a gradient that was 2.5–11× too small *and* pointed unreliably;
under a D-RTRL trace for this one primitive they would receive the true
pairing-directed gradient at ~0.55 gain. The next decision (not taken here)
is an ARC arm with `etp_outer_write` under an exact trace — either D-RTRL
for the whole model at Example 21 scale (memory cost of the dense traces at
4096 neurons needs pricing first) or a hybrid that keeps pp-prop elsewhere.

Artifacts: `var/pairing_gradient_probe.py` (extended, two-arm),
`var/pairing_gradient_probe_drtrl.json` (gitignored by var policy).
