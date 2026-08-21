# Example 21 — exact-path pairing-gradient disambiguation

Status: approved 2026-08-21 ("sounds quick enough. Execute."); in progress
Date: 2026-08-21
Branch: `investigate/ex21-learned-memory-keys` (worktree `.worktrees/ex21-learned-keys`)
Depends on: `2026-08-20-etp-outer-write-primitive.md` (decision rule, null branch)

## Question

`learned_write` made the memory write differentiable and the write projections
train hard (l2_delta 1.80 key / 1.93 value), yet the shuffled-demonstrations
deviation stayed at +0.0096 against frozen's +0.0192 — smaller, and
positive-signed in every arm. The model still does not use demonstration
pairing.

The `etp_outer_write` decision rule forbids reading that as "the binder is not
the coding" until an exact path has been consulted, because pp-prop's rank-1
collapse is outside its sign-consistency envelope for this primitive. Two
candidate explanations are already refuted:

- *the learning rule cannot carry pairing* — refuted on a 2×2 toy memory:
  finite-window pp-prop tracks BPTT at cosine 0.87–0.98 and separates
  pairing-permuted sequences;
- *the read drifted out of the written code* — refuted directly: write and
  retrieval key encoders sit at cosine 0.99992 after 260 updates.

What remains untested is the **model itself**: does `LatentWorkspaceModel`, at
a scale where the exact gradient is computable, have a pairing gradient at all?

## Design

One measurement, repeated up a size ladder.

Two input sequences are built that are identical in every respect except the
association between demonstration keys and values. `straight` pairs key `i`
with value `i`; `permuted` reverses the value sequence across demonstration
events, leaving every key in place. Both therefore present the same multiset of
keys, the same multiset of values, the same phase flags and the same
side-validity flags — only *what is stored with what* changes.

For each rung, four gradient trees:

| symbol | path | meaning |
|---|---|---|
| `E_s`, `E_p` | `bptt_param_gradients` | exact total gradient, straight / permuted |
| `O_s`, `O_p` | `chunked_online_param_gradients` | finite-window pp-prop, same two |

and three numbers derived from them:

- `exact_response = ||E_p − E_s|| / ||E_s||` — how much the exact learner's
  gradient moves when only the pairing changes;
- `online_response = ||O_p − O_s|| / ||O_s||` — the same for the online rule;
- `alignment = cos(E_p − E_s, O_p − O_s)` — whether the online rule moves in
  the same direction the exact one does.

Reported for the whole parameter set and, separately, restricted to the three
write projections (`write_key_weight`, `write_key_bias`, `write_value_weight`),
which is where a pairing gradient must appear if it appears anywhere.

`memory_coding="learned_write"`, since that is the arm under test.

### Reading the result

| `exact_response` | `online_response` | conclusion |
|---|---|---|
| ≈ 0 | ≈ 0 | **The binder is not the coding.** The model's own loss is blind to pairing at this scale; no learning rule could recover it. Next spec targets the memory *format* — delta-rule write first. |
| > 0 | ≈ 0 | The trace factorization loses pairing on the real model. The rule needs work, not the format. |
| > 0 | > 0, `alignment` > 0 | The rule sees pairing. The failure is downstream — optimization, capacity, or readout — and is a different investigation. |
| > 0 | > 0, `alignment` ≤ 0 | The online rule moves *against* the exact one; a sign or factor error to hunt before anything else. |

"≈ 0" means at or below the same-sequence floating-point floor, which the probe
measures directly rather than assuming: `E_s` is computed twice and their
relative deviation is reported as `noise_floor`. A response must exceed that
floor to count.

### Size ladder

Rung 0 is the configuration the existing trainability test already uses (64
neurons, memory width 2, 2 demonstrations), whose cost is known to be seconds.
Each rung raises one dimension:

| rung | neurons | memory width | demonstrations |
|---|---:|---:|---:|
| 0 | 64 | 2 | 2 |
| 1 | 64 | 4 | 4 |
| 2 | 128 | 8 | 4 |
| 3 | 256 | 8 | 6 |
| 4 | 256 | 16 | 8 |

Wall clock is measured and reported per rung. The ladder stops when the answer
is unambiguous, when a rung crosses ~2 minutes, or at rung 4 — whichever comes
first. No run is projected past a measured rung.

## Scope

Diagnostic only. Nothing in `braintrace/` or the model changes; the probe is a
driver script under `var/` (gitignored, as with the pilot drivers), and its
substance is this spec plus the recorded numbers. The D-RTRL half of the
decision rule's exact path stays out of scope: `etp_outer_write` registers
`init_drtrl`/`dt_to_t` as loud `NotSupportedError` by design, so that path is
implementation work, not a run.

## Results (2026-08-21, all measured)

Five rungs, 9.5–14.5 s each; whole ladder under two minutes on CPU. The
`noise_floor` was exactly `0.0` at every rung and scope, so every response
below is real, not floating-point.

### Whole parameter set — a plumbing check, not evidence

| rung | exact response | online response | delta alignment | baseline alignment | \|O\|/\|E\| |
|---|---:|---:|---:|---:|---:|
| n64 w2 d2 | 0.3990 | 0.3990 | 0.9999 | 1.0000 | 0.9998 |
| n64 w4 d4 | 0.7375 | 0.7374 | 1.0000 | 1.0000 | 0.9998 |
| n128 w8 d4 | 0.7894 | 0.7893 | 1.0000 | 1.0000 | 0.9999 |
| n256 w8 d6 | 0.6321 | 0.6321 | 1.0000 | 1.0000 | 0.9999 |
| n256 w16 d8 | 0.2378 | 0.2378 | 1.0000 | 1.0000 | 0.9999 |

The baseline alignment of exactly 1.0000 and magnitude ratio of 0.9999 say what
this scope really measures: it is dominated in norm by parameters pp-prop
computes by **exact reverse-mode inside the window** (the readout and row/shape
heads, which the compiler excludes from ETP under the non-parametric-tail
invariant). It confirms the harness is wired correctly and nothing more. This
was not anticipated in the design above and is recorded so the table is not
mistaken for a trace result.

The one thing it does establish, and it matters: **`exact_response` is 0.24–0.79
everywhere.** The model's own loss is strongly pairing-sensitive at every rung.
The top row of the reading table — "the binder is not the coding" — is refuted.

### Write projections — the informative scope

| rung | exact response | online response | delta alignment | baseline alignment | \|O\|/\|E\| |
|---|---:|---:|---:|---:|---:|
| n64 w2 d2 | 1.4684 | 0.7592 | −0.1309 | 0.6382 | 0.093 |
| n64 w4 d4 | 1.0358 | 0.9549 | −0.1531 | 0.7878 | 0.103 |
| n128 w8 d4 | 0.7572 | 0.4503 | 0.6013 | 0.8075 | 0.396 |
| n256 w8 d6 | 0.4330 | 0.5462 | 0.7208 | 0.8613 | 0.168 |
| n256 w16 d8 | 0.6400 | 0.1828 | 0.1338 | 0.7317 | 0.235 |

Three separate facts, and they must not be conflated:

1. **The exact learner has a strong pairing gradient on the write
   projections** (0.43–1.47). The signal exists and it is large.
2. **The online write gradient is usable but heavily attenuated.** Its baseline
   alignment with BPTT is 0.64–0.86 — a genuine descent direction, not noise —
   but its magnitude is **2.5× to 11× too small** at every rung, with no trend
   toward improvement as the model grows.
3. **Its pairing-specific component is not recovered.** The direction the
   gradient moves when only the pairing changes aligns with BPTT at −0.15 to
   +0.72 with no trend: sometimes anti-correlated, never reliable.

### Mechanism: not trace truncation

The obvious suspect was the trace decay discounting the demonstration-to-query
gap. Swept at rung `n128 w8 d4` (4.3–7.1 s per point):

| `decay_or_rank` | baseline alignment | \|O\|/\|E\| | delta alignment |
|---|---:|---:|---:|
| 0.9 (default) | 0.8075 | 0.3964 | 0.6013 |
| 0.95 | 0.7992 | 0.4306 | 0.5552 |
| 0.99 | 0.7922 | 0.4588 | 0.5157 |
| 0.999 | 0.7906 | 0.4653 | 0.5072 |

Effectively unbounded trace memory recovers almost nothing: magnitude moves
0.40 → 0.47, and both alignments get slightly *worse*. **Trace truncation is
refuted as the mechanism.**

## Verdict

Row two of the reading table: `exact_response > 0`, `online_response > 0`,
alignment unreliable — **the trace loses the pairing-specific part of the write
gradient on the real model.** The failure is in the learning rule, not in the
memory format, and the delta-rule write is therefore *not* the right next spec.

What remains, having eliminated the decay, is pp-prop's rank-1 collapse
`ε ≈ ε_f ⊗ ε_x` itself — precisely the caveat the `etp_outer_write` spec
flagged and declined to assume away: BrainScale justifies that collapse by
sign-consistent pre/post quantities, and this primitive's factors alternate
sign by construction (`φ'_k = −sin`, signed `tanh`). The measurement is
consistent with that being the binding constraint, though the hidden-group
diagonal-Jacobian approximation is not separately excluded and would need its
own probe.

The practical reading of the pilot is now different from what it looked like.
The write projections' large movement (l2_delta 1.80/1.93) was real but
substantially *not* pairing-directed, and the write gradient carries roughly a
third to a tenth of its true weight — so at a learning rate shared with every
other parameter group, the write is both under-driven and pointed partly the
wrong way. That is enough to explain a flat pairing readout without any appeal
to the memory format.

### Recommended next step (not executed)

Give `etp_outer_write` an exact trace and re-measure the same panel. The
deferred D-RTRL weight-shaped trace is the direct instrument — at rung scale
its `B × A_k × K × V × S` cost is trivial, and only at Example 21 scale does it
reach ≈55 MB/state. If the exact trace restores the pairing alignment, the
conclusion is that this primitive needs D-RTRL (or SnAp) rather than pp-prop,
and the ARC question can be re-asked on a rule that can actually see the
signal. That is implementation work, not a run: `init_drtrl` / `dt_to_t`
currently raise `NotSupportedError` by design.

Artifacts: `var/pairing_gradient_probe.py`, `var/pairing_gradient_probe.json`
(gitignored by var policy).
