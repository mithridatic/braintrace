# Example 21 — why the demonstration channel carries no usable signal

Status: root cause identified; padding-advance fix applied, remaining
directions not implemented
Date: 2026-08-17
Branch: `feat/example21-latent-reasoning`
Follows: `docs/specs/2026-08-17-example21-chunked-training-stream.md`

## The symptom

At 4096 updates the `shuffled_demonstrations` control scores within ±0.01 of
the intact arm at every effort. Destroying the demonstration input/output
pairing costs the model nothing.

## Root cause

**Demonstration *presence* is strongly represented in the state the readout
heads see. The input→output *pairing* is at the null.** Whatever the model has
learned to read from the demonstration block is invariant to which output
accompanies which input, which is why the control is free.

### Why the control is invariant, exactly

`_derange_task` (`21-latent-reasoning-in-context.py:988`) rotates the outputs
cyclically across the demonstration pairs. The multiset of demonstration inputs
and the multiset of demonstration outputs are both preserved to the byte; only
which output accompanies which input changes. **Any pooled, order-free readout
of the demonstration block is mathematically unchanged by this control.** A
model reading demo-output shape and color statistics scores identically on both
arms by construction, not by coincidence.

### Evidence

A ridge linear probe on the final-checkpoint spike vector — the state the
readout heads are actually given — of a model trained 512 updates at lr 1e-3.
Folds are grouped by query so both arms of a query stay on the same side of the
split, and the null permutes labels at the group level:

| discrimination | grouped-CV accuracy | permuted-label null |
| -------------- | ------------------- | ------------------- |
| intact vs `no_context` (demonstrations blanked) | 0.9296 | 0.4893 |
| intact vs `shuffled_demonstrations` (pairing destroyed) | 0.5084 | 0.5072 |

The three arms share a byte-identical advance schedule — `_arm_sequences:1048`
takes `_packed_advances` from the intact encoding for every arm — so neither
number can be explained by a difference in how many steps the network ran.
`no_context` blanks only rows `[:query_start]`, the demonstration region, and
leaves the query rows untouched. The two rows above therefore isolate the same
variable at two strengths: remove the demonstrations entirely and the final
state changes enormously; keep every demonstration grid but re-pair them and the
final state is indistinguishable from intact.

Presence and content of the demonstration block: strongly represented.
Pairing: at the null.

### Where the pairing is lost

Nothing in the architecture spans the distance over which a pairing would have
to be held:

1. `dt` is 1 ms, membrane tau 20 ms, feed-forward synapse tau 40 ms — a leak
   horizon of a few tens of steps.
2. Demonstration rows sit 30–330 advancing steps before the readout.
3. The only structure that could bridge that is the recurrence: 16,384 edges
   over 2,048 neurons, 8 per neuron, gain 0.8 — too sparse and too weak to hold
   the attractor that carrying a relation 300 steps would need.

**Hypothesis, not yet measured.** A leaky integrator over a long block computes
something close to a decaying *sum* of its inputs, and a sum is order-free — so
the model may be reading pooled demonstration statistics (output sizes and
colors, pooled across pairs). That would be a correct solution to the objective
as posed and invariant to the control by construction. But nothing measured here
separates that from the weaker reading: that the 0.93 is generic drive from ~90
non-blank rows and the demonstration *content* is not read at all.

The distinction decides which fix is worth building — under pooled statistics,
the span directions below are sensible; under generic drive, only the
objective direction is.
Settling it is one regression on machinery that already exists: predict a
demo-output statistic from the intact final spikes with folds grouped by task,
and compare to the null. That belongs in the next spec.

The compiler agrees about what the output parameters can see:

```
readout_projection.weight ... excluded from ETP; learn it by BPTT
height_head.weight / width_head.weight / color_factor_head.weight
    ... has no connected hidden states, treated as a non-temporal parameter
```

All four output-side parameters are fed the final spike vector and nothing else.

### A concrete amplifier: padding rows advance the state (since fixed, below)

`encode_query_episode` gives every demonstration a **fixed 30-row block**
regardless of grid height (`latent_workspace_task.py:1554`), and
`_packed_advances` marks the *whole* block as advancing
(`21-latent-reasoning-in-context.py:463`). The query block, by contrast,
advances only over its true height (`query_stop = query_start +
query.input.height`).

Each demonstration is therefore followed by `30 - height` all-zero advancing
steps of pure leak. For a typical 8-row ARC grid that is 22 wasted steps per
demonstration, attenuating the most recent demonstration by `exp(-22/20) ≈ 0.33`
before the query block even begins. The asymmetry is unintentional: demo blocks
pay for padding, the query block does not.

This is verified from the code and stands independently of the probe. It makes
the span roughly 3× longer than it needs to be; it does not by itself explain a
300-step gap.

## What this rules out

- **Credit horizon.** `ETraceConfig(decay=0.9)` is the `scalar_leak`
  temporal-recursion coefficient (`braintrace/_algorithm/axes.py:483`), a
  ~10-step trace horizon, and it was the first suspect. It cannot be the binding
  constraint: the readout heads are non-temporal and read the final spike vector
  directly, so no eligibility trace is involved in what they learn. Lengthening
  the trace would not change what they are fed.
- **Training budget.** Already bought and spent; loss descends, pixel accuracy
  does not follow.
- **Capacity.** Untestable and irrelevant until a pairing survives to the
  readout.

## Limits of this evidence

The probe is linear; `readout_projection` followed by a GELU is mildly
non-linear, so "not linearly decodable" is very close to, but not identical
with, "not head-accessible". The measurement was taken at 512 updates on the
assumption that a distinction absent at 512 is not created by more training on
the same objective — the 4096 control arms are consistent with that but do not
prove it.

## Fix applied: demonstration padding no longer advances the state

`_packed_advances` now advances each demonstration block over
`_demonstration_advance_width` rows instead of the full `max_grid_size`. The
width is the per-episode maximum occupied height, shared by every block, which
has two properties the control depends on:

- **It never drops an encoded row.** Every block advances at least as far as the
  tallest one.
- **It is invariant under `_derange_task`.** Rotation is a bijection on the
  outputs, so `max_i max(in_i, out_i)` and `max_i max(in_i, out_{i+1})` are both
  `max(max in, max out)`. Intact and deranged encodings therefore produce a
  byte-identical schedule, and `shuffled_demonstrations` stays a content-only
  control with matched timing.

Measured at 512 updates, lr 1e-3, same seed:

| | before | after |
| --- | --- | --- |
| final-eighth mean training loss | 6.2636 | 5.7727 |
| intact vs `no_context` decodability | 0.9296 | 0.9857 |
| intact vs `shuffled` decodability | 0.5084 | 0.5107 |
| its null | 0.5072 | 0.4785 |

### Full qualification, 4096 updates, chunk 512, lr 1e-3, 400-task evaluation

All 8 structural and all 12 scientific gates pass; 749 s on GPU.

Training loss improves at **every** window against the pre-fix run of the same
configuration:

| updates | before | after |
| ------- | ------ | ----- |
| 0–256 | 7.887 | 7.619 |
| 256–512 | 6.484 | 6.192 |
| 512–1024 | 5.724 | 5.415 |
| 1024–2048 | 5.032 | 4.833 |
| 2048–4096 | 4.341 | 4.163 |

Evaluation is mixed rather than uniformly better — intact best-effort pixel
0.3915 → 0.4144, intact best-effort shape 0.4105 → 0.3842.

The `no_context` control separates further than before (shape 0.057–0.086,
pixel 0.098–0.114, against 0.103–0.146 / 0.156–0.171 pre-fix), consistent with
the demonstrations landing harder.

`shuffled_demonstrations` is unchanged in the way that matters: shape
0.3723–0.3771, pixel 0.3951–0.4126, every effort within ±0.012 of intact. The
finding this spec records survives the fix intact.

### Summary

The demonstrations land harder and the model trains better for free — the loss
gain is roughly three quarters of what a doubling of the update budget buys.
**The pairing is still at the null.** This was the predicted outcome: removing
~22 leak steps per demonstration shortens the span roughly 3×, and the span was
never the whole story. The remaining directions below are unaffected.

## Candidate directions (none implemented, none costed)
1. **Give the substrate a state that outlives the episode** — a slow adapting
   variable or a long-tau synapse sized to the 300-step span, rather than asking
   a 20 ms membrane to hold it.
2. **Shorten the span** — re-present or summarise the demonstrations adjacent to
   the query, so tens of steps must be bridged rather than hundreds.
3. **Make the objective require the pairing** — an auxiliary term the deranged
   arm provably fails. Without this, the other two enlarge the capacity to
   represent a pairing without giving the model any reason to.

These are architecture changes, not tuning. Each wants its own spec, a stated
prediction for the `shuffled_demonstrations` control, and one change at a time.

## Retractions from this investigation

Two, both from reading a measurement that could not support its conclusion:

- "The demonstrations do reach the readout with large magnitude" — based on raw
  state divergence on an *untrained* model, which measures amplification of a
  perturbation, not information.
- "`no_context` collapses, so the model uses the query grid" — carried forward
  from the 4096 analysis. `no_context` blanks the demonstration rows, not the
  query rows (`_arm_sequences:1021`); its collapse says the model uses the
  demonstrations, which is the opposite reading.

A first version of this spec also reported an intact-vs-shuffled probe accuracy
of 0.4487 against a 0.5048 null. That was ungrouped k-fold on a paired design,
which trains on one arm of a query and tests on the other; the anti-correlated
leak is what pushed it below chance. The grouped numbers above supersede it.
