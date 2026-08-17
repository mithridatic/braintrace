# Latent readout consistency (Example 21)

Status: implementing. Supersedes the readout behaviour described in
`2026-08-16-pp-prop-latent-reasoning.md`; that document's measured results are
retained as the pre-fix baseline.

## Problem

Example 21's accuracy-versus-depth curve is uninterpretable, not merely
negative. `latent_workspace_model.py` decoded a *different object* at `R = 0`
than at `R >= 1`:

```python
representation = query * pure_query_read + (1.0 - query) * state_next[:, -1]
```

At `R = 0` the terminal tick is a query tick, so logits decode the **analog**
memory read `A(Bᵀq)`. At `R >= 1` the terminal tick is a latent tick, so logits
decode the **binary** spike workspace. `Wo` is initialised to `value_codes.T`,
which is the analytic decoder for the analog value code, not for a thresholded
version of it.

Every depth comparison therefore confounds latent depth with a readout
representation switch. The measured drop from `0.994` at `R = 0` to `0.203` at
one latent tick cannot be attributed to iteration.

A second, smaller defect: the workspace state entering the first latent tick was
never seeded from the query. `H_0` as reported was the accumulated spike state,
not the encoding of the query against memory.

## Published interface being instantiated

BDH-CQ (arXiv:2608.09888) publishes the system-level interface as equations
(2)–(4). Dimensions, exact update rules, and the training recipe are explicitly
withheld; these three equations are not.

```
H_0     = E_θ(x*, S_K)                       (2)
H_{r+1} = F_θ(H_r, S_K),   r = 0..R-1        (3)
ŷ       = G_θ(H_R)                           (4)
```

Two properties are load-bearing and both were violated:

1. `H_0` is the query *encoded against contextual memory* — not zero, and not an
   unrelated accumulated state.
2. `G_θ` is one decoder applied to `H_R` for every `R`. There is no separate
   `R = 0` path.

## Decision

Keep the spiking substrate. Carry and decode the **analog membrane voltage**.

```
H_0.voltage := A(Bᵀq)                       # eq. 2, on the query tick
spikes       = surrogate(V - threshold)     # substrate unchanged
V_{r+1}      = decay*V + Wf@spikes + read(S_K, spikes)
ŷ            = Wo @ V_r                     # eq. 4, one path at every r
```

Rationale for the voltage over the alternatives considered:

- `latent_voltage` already exists as a `HiddenState`, so no new state leaf is
  introduced and the coupled hidden Jacobian does not grow. That budget is
  already at 42.6M of 67.1M elements at the release configuration.
- The spiking recurrence is preserved, so the example still exercises pp_prop
  over an SNN. An analog workspace would be closer to BDH's ReLU-low-rank
  formulation but would stop testing what this repository is for.

Accepted cost: subtractive reset makes the voltage a sawtooth, so a reset
artefact rides on the decoded signal.

### Sub-decision: seed both carrier and substrate

Setting only the voltage on the query tick would leave the first latent tick's
`Wf @ workspace_previous` term driven by a spike state unrelated to `H_0` — the
recurrence's first step would ignore its own initial condition. The query tick
therefore also seeds the spike row with `surrogate(H_0 - threshold)`, so both
the analog carrier and the substrate start from `H_0`.

## Invariants

- **`R = 0` output is bit-identical to the pre-fix implementation.**
  `pure_query_read` depends only on the memory factors and the query encoding,
  and neither depends on the workspace, so seeding the workspace cannot perturb
  it. This is what makes the fix safe to land against the recorded baseline.
- Demonstration-phase writes are untouched: `key_rows` and `value_rows` are
  gated to zero on query ticks, so seeding the workspace row cannot disturb
  stored bindings.
- The reported `H_0..H_R` trajectory stays binary. `surrogate` evaluates
  numerically to `{0, 1}`, so the substrate remains a spike train.
- `Wk` and `Wv` remain fixed under `write_mode="fixed_random"`.

## Test changes

- `test_reported_h0_through_hr_and_internal_workspace_are_binary` asserted
  `terminal_logits == workspace[-1] @ Wo` at `R = 2`. That assertion encodes the
  defect and is replaced by one against the latent voltage view. The binary
  trajectory assertions are retained.
- `test_memory_read_includes_final_query_tick_and_controls_r0_logits` must
  continue to pass unchanged. It is the regression guard for the `R = 0`
  invariant above.
- New: assert that terminal logits are decoded from the same carrier at `R = 0`
  and `R >= 1`, and that `H_0` equals the analog memory read.

## Explicitly out of scope

- The Perron–Frobenius collapse of the recurrent workspace. `Wf` is a dense
  non-negative primitive matrix applied to a non-negative state, so repeated
  application is power iteration onto the dominant eigenvector; the measured
  participation ratio falls from 3.82 to ~1.0 out of 32. This fix does not
  address it, and the depth curve is expected to stay flat afterwards. The
  candidate remedy is `braintrace.nn.SignedWLinear` under Dale's law, tracked
  separately.
- Porting the paper's depth-bearing task families (§6.2). Example 21 currently
  implements the paper's §6.3 dense-mapping *control*, which has no depth axis.
- Training that moves. Eight updates at `lr=1e-5` left `W_f` at exactly zero
  delta.

## Test environment

The focused tests must run with `JAX_PLATFORMS=cpu`. On GPU, JAX evaluates
float32 matmuls in TF32 (~1e-3 relative), and the model's zero-padded
`(batch * state_rows, width) @ (width, symbol_count)` readout rounds differently
from the narrow `(batch, width) @ (width, symbol_count)` the assertions
recompute. The gap is ~1.2e-4, which exceeds the 1e-5 readout tolerances. This
predates the change and is a property of the tolerances, not of the seeding:
`max|voltage - memory_read|` is exactly `0.0`. Measured outcomes below are GPU
runs; the test evidence is CPU.

## Measured outcome

Default configuration, RTX 3080 Ti, `seed=2108`, `codebook_seed=313320`,
`projection_seed=210848`, `batch=4`, `width=32`, 8 terminal pp_prop updates per
depth. `acc` is overall supported/intact accuracy across `K = 2..8`; blind
chance is `0.10`. `PR_last` is the participation ratio at the final latent
iteration out of `latent_width=32`. Probe scores use 14 held-out episodes, so
probe resolution is `1/14 ≈ 0.071`.

| R | acc (pre) | acc (post) | intact−shuffled (pre) | intact−shuffled (post) | PR_last (post) |
|---|---|---|---|---|---|
| 0 | 0.429 | 0.429 | +0.357 | +0.357 | 3.82 |
| 1 | 0.107 | **0.429** | +0.036 | **+0.357** | 3.35 |
| 2 | 0.143 | 0.393 | −0.071 | +0.250 | 1.42 |
| 4 | 0.107 | 0.357 | +0.000 | +0.250 | 0.72 |
| 8 | 0.143 | 0.357 | +0.036 | +0.250 | 0.69 |

The `R = 0` row is unchanged, as the invariant requires. The `R = 0 → R = 1`
discontinuity is gone: accuracy and both interventions are now identical across
that boundary, which is the readout-continuity property this change exists to
establish. The memory-dependence interventions now survive iteration
(`intact − shuffled` holds at `+0.250` through `R = 8` instead of decaying to
noise), so the workspace carries the contextual read forward.

Workspace answer decodability at `R = 8`:
`[0.143, 0.214, 0.357, 0.286, 0.214, 0.143, 0.143, 0.143, 0.357]` against a
`memory_read` baseline of `0.571`. It peaks at `r = 2` and never reaches the
baseline at any iteration or any depth. `rule_decodability` remains exactly
`0.0` everywhere.

Participation ratio still collapses from `3.82` to `0.69`, as predicted by the
Perron–Frobenius argument in "Explicitly out of scope". The depth curve is
therefore flat-to-declining — but it is now a *valid* flat curve. Before this
change it measured a representation switch; after it, it measures depth.

## Claim boundary

This change makes the depth axis *measurable*. It does not make it *non-trivial*
and produces no evidence of latent reasoning. A flat post-fix curve is a valid
negative result where the pre-fix curve was not a result at all.
