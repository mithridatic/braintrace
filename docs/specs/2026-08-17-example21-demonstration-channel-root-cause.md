# Example 21 — why the demonstration channel carries no usable signal

Status: root cause identified; no fix implemented (awaiting approval)
Date: 2026-08-17
Branch: `feat/example21-latent-reasoning`
Follows: `docs/specs/2026-08-17-example21-chunked-training-stream.md`

## The symptom

At 4096 updates the `shuffled_demonstrations` control scores within ±0.01 of
the intact arm at every effort. Destroying the demonstration input/output
pairing costs the model nothing.

## Root cause

**The demonstrations reach the readout, but only as pooled statistics. The
input→output *pairing* is absent from the state the readout heads see.**

The model has learned "what outputs for this task tend to look like" — their
sizes and colors, pooled over the demonstrations — and not "what operation maps
an input to its output". That is a correct solution to the objective as posed
and it is invariant to the control, which is why the control is free.

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

A leaky integrator over a long block computes something close to a decaying
*sum* of its inputs. A sum is order-free. That is precisely the class of
readout the control cannot touch, and precisely what the measurements show.

The compiler agrees about what the output parameters can see:

```
readout_projection.weight ... excluded from ETP; learn it by BPTT
height_head.weight / width_head.weight / color_factor_head.weight
    ... has no connected hidden states, treated as a non-temporal parameter
```

All four output-side parameters are fed the final spike vector and nothing else.

### A concrete amplifier: padding rows advance the state

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

## Candidate directions (none implemented, none costed)

1. **Stop advancing padding rows** — smallest change, partial gain, and it makes
   the demo and query blocks symmetric. Changes numerics for every existing
   artifact.
2. **Give the substrate a state that outlives the episode** — a slow adapting
   variable or a long-tau synapse sized to the 300-step span, rather than asking
   a 20 ms membrane to hold it.
3. **Shorten the span** — re-present or summarise the demonstrations adjacent to
   the query, so tens of steps must be bridged rather than hundreds.
4. **Make the objective require the pairing** — an auxiliary term the deranged
   arm provably fails. Without this, options 1–3 enlarge the capacity to
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
