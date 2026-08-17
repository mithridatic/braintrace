# Example 21 — why the demonstration channel carries no usable signal

Status: root cause identified; no fix implemented (awaiting approval)
Date: 2026-08-17
Branch: `feat/example21-latent-reasoning`
Follows: `docs/specs/2026-08-17-example21-chunked-training-stream.md`

## The symptom

At 4096 updates the `shuffled_demonstrations` control scores within ±0.01 of
the intact arm at every effort. Destroying the demonstration input/output
pairing costs the model nothing. `no_context` does collapse, so the model uses
the query grid. The model is learning a query-conditioned output prior, not the
demonstrated transformation.

## Root cause

**The demonstration content is not present, in any form a readout head can
use, in the state the readout heads are given.** The failure is
representational. It is not a credit-assignment failure and not a capacity
failure.

### Evidence

A linear probe was fitted, 5-fold cross-validated, on the final-checkpoint
spike vector of a model trained for 512 updates at lr 1e-3, against a
permuted-label null (2048 features on a few hundred samples separates noise, so
the null is the only meaningful reference):

| discrimination | CV accuracy | permuted-label null |
| -------------- | ----------- | ------------------- |
| intact vs `no_context` | 0.9391 | 0.5024 |
| intact vs `shuffled_demonstrations` | 0.4487 | 0.5048 |

Query content is strongly decodable. Demonstration content is **at the null**.
The heads cannot use what is not there, so they learn the query prior — exactly
what the control arms report.

### The mechanism

Raw state divergence is *not* the discriminator, and reading it as one is what
sent the first pass down the wrong path. On an untrained model the shuffled arm
diverges 0.71 in spikes but only 0.11 in voltage, against 0.83 / 0.84 for
`no_context`. A 6× amplification of a small membrane footprint into a large
spike difference is unstructured jitter, not signal — and the decodability
measurement above confirms it carries nothing.

The membrane footprint is small because nothing in the architecture spans the
required distance:

1. `dt` is 1 ms, membrane tau is 20 ms, feed-forward synapse tau is 40 ms. The
   leak horizon is a few tens of steps.
2. Demonstration rows sit 30–330 advancing steps before the readout.
3. The only structure that could bridge that gap is the recurrence — 16,384
   edges over 2,048 neurons, 8 per neuron, gain 0.8. Too sparse and too weak to
   sustain the attractor that carrying content 300 steps would require.

The compiler agrees about which parameters see what. Running the model emits:

```
readout_projection.weight ... excluded from ETP; learn it by BPTT
height_head.weight / width_head.weight / color_factor_head.weight
    ... has no connected hidden states, treated as a non-temporal parameter
```

All four output-side parameters are fed the final spike vector and nothing
else. Whatever is not in that vector is unreachable by any amount of training.

### A concrete amplifier: padding rows advance the state

`encode_query_episode` gives every demonstration a **fixed 30-row block**
regardless of grid height (`latent_workspace_task.py:1554`), and
`_packed_advances` marks the *whole* block as advancing
(`21-latent-reasoning-in-context.py:463`). The query block, by contrast, is
advanced only over its true height (`query_stop = query_start +
query.input.height`).

So each demonstration is followed by `30 - height` all-zero advancing steps —
pure leak. For a typical 8-row ARC grid that is 22 wasted steps per
demonstration, attenuating the most recent demonstration by `exp(-22/20) ≈ 0.33`
before the query block even begins. The asymmetry is unintentional: demo blocks
pay for padding, the query block does not.

This is an amplifier of the root cause, not the root cause itself. Removing it
shortens the span by roughly 3× and would not by itself bridge 300 steps.

## What this rules out

- **Credit horizon.** `ETraceConfig(decay=0.9)` is the `scalar_leak`
  temporal-recursion coefficient (`braintrace/_algorithm/axes.py:483`), giving a
  ~10-step trace horizon, and it was the first suspect. It cannot be the binding
  constraint: at effort 32 the entire query block sits ≥32 advancing steps
  before the readout, where `0.9^32 ≈ 0.034`, yet the query is plainly used.
  Query use is explained without the trace at all — the heads are non-temporal
  and read the final spike vector directly.
- **Training budget.** Already bought and spent; loss descends, pixel accuracy
  does not follow.
- **Capacity.** Untestable and irrelevant until the demonstrations reach the
  readout.

## Limits of this evidence

The probe is linear; `readout_projection` followed by a GELU is mildly
non-linear, so "not linearly decodable" is very close to, but not identical
with, "not head-accessible". A non-linear probe would tighten this. The
measurement was taken at 512 updates, not 4096, on the assumption that a channel
absent at 512 is not created by further training on the same objective — the
4096 control arms are consistent with that but do not prove it.

## Candidate directions (none implemented, none costed)

1. **Stop advancing padding rows** — smallest change, real but partial gain,
   and it makes the demo and query blocks symmetric. Changes numerics for every
   existing artifact.
2. **Give the substrate a state that outlives the episode** — a slow adapting
   variable or a long-tau synapse sized to the 300-step span, rather than
   asking a 20 ms membrane to hold it.
3. **Shorten the required span** — re-present or summarise the demonstrations
   adjacent to the query, so the distance to be bridged is tens of steps rather
   than hundreds.

These are architecture changes, not tuning. They want their own spec, a stated
prediction for the `shuffled_demonstrations` control, and one change at a time.

## Retraction

An earlier reading of this session claimed the demonstrations "do reach the
readout with large magnitude", based on raw state divergence on an untrained
model. That was wrong on both counts — untrained divergence measures
amplification, not information, and the decodability measurement shows the
channel is empty.
