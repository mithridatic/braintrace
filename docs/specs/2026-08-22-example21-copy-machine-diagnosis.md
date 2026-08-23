# Example 21: the model-only channel is a copy machine end to end

Date: 2026-08-22

## Summary

Example 21's model-only channel does not perform in-context learning. Every
component that contributes to its score is a copy of the query input. The
cumulative-4 result is produced by the shape head cropping the query input to a
predicted extent; the colour head contributes nothing at all.

This document records the measurements, the two hypotheses they refuted, and the
one intervention they point to.

## Measurements

All figures at effort 60, replayed offline from checkpoint logit dumps
(`EXAMPLE21_LOGITS_DUMP`) through the fixed decoder on `main`.

### The colour head is an exact identity map

On `logits-fixed-s31337.npz` (the cumulative-4 base, `row_head_carrier_scale=0.0`,
`copy_residual_gain=2.0`):

| quantity | value |
|---|---|
| colour argmax == query input, over the input footprint | **113,982 / 113,982 = 1.000000** |
| cells where the head deviates from the input (same-shape queries) | **0** |
| mean top1-top2 colour logit margin | 1.87 |
| of the 89,117 copied cells, correct | 72,692 (**0.8157**) |

The head reproduces the input exactly and confidently. Two consequences follow
directly and were both confirmed:

- Adding a copy prior to the colour logits is a **no-op**. Sweeping the gain
  0 -> 20 leaves cumulative, shape and pixel identical to four decimals, because
  the argmax already *is* the input.
- Pixel accuracy is bimodal by target size, which is the copy signature:
  **0.757**/cell on targets over 100 cells, but **0.00-0.25** on 3x3 and smaller.

### A perfect copy scores zero

Submitting the query input verbatim scores cumulative **0**: shape 0.6611,
pixel 0.6032, and **no evaluation query has target == input**. Copying can never
produce an exact answer. Notably a verbatim copy predicts the *shape* better
(0.6611) than the trained shape head does (0.5776).

### The head has no information about the cells that change

Restricted to the 16,425 cells where `target != query_input` on same-shape
queries, the rank of the true colour in the head's own logits:

| base | mean rank | P(true == rank 2) | rank-1 accuracy |
|---|---|---|---|
| `fixed-s31337` (carrier scale 0.0) | 5.843 | 0.1310 | 0.0000 |
| `default-u390` (carrier scale 1.0) | 5.263 | 0.1105 | 0.1046 |
| `default-u700` (carrier scale 1.0) | 5.091 | 0.1195 | 0.1195 |

Uniform over the nine non-input alternatives is 5.5, and the marginal colour
prior alone would score 0.1919. The carrier-blind head is therefore **at chance,
and worse than the marginal prior**. Connecting the carrier buys a real but
small signal (10-12% of changed cells at rank 1) at the cost of corrupting the
copy: those runs score cumulative 0 against the copy machine's 4.

This distinction is load-bearing. A head whose true-colour rank is uniform is a
**confident conditional mode of a distribution it has no conditioning
information for**, not a diluted learner. Reweighting the loss toward changed
cells would destroy the 81.6% copy and gain nothing.

### The shape head is also a copy

| base | all queries | the 19 exact-crop queries |
|---|---|---|
| `fixed-s31337` | 242/419 = 0.578 | **2/19 = 0.105** |
| `default-u700` | 248/419 = 0.592 | **2/19 = 0.105** |

Overall shape accuracy is carried by the 277/419 queries whose target shape
equals the input shape. Where the answer's shape genuinely differs, the head is
at 10.5%, and it is beaten outright by simply echoing the input shape (0.661).

## Refuted hypotheses

### 1. The row head lacks a carrier x row product term

`answer_row_head` is a bias-free linear map over `concat(carrier, event blocks)`,
so it has no term in which the carrier and the query row multiply: it can add a
task embedding to a row but cannot apply a task-selected map to it. Added
`--row-head-modulation bilinear` (rank 512), zero-initialised, with the two
factors as fixed random sketches so only one trainable matrix feeds a hidden
state. (Making the factors trainable trips the ETP non-parametric-tail invariant,
excludes them from eligibility tracing, and flips `row_routes_all_direct` to
False, failing structural qualification.)

Full-scale run matched to `default-u390` with the bilinear head as the only
difference:

| run | shape | pixel | input echo | cumulative |
|---|---|---|---|---|
| `default-u390` | 0.5442 | 0.4425 | 0.652 | 0 |
| `bilinear-u390` | 0.5394 | 0.4455 | 0.660 | 0 |

Same two near-misses, same zero exact answers, near-identical oracle-shape
Hamming histogram. **Refuted.**

This is a refutation rather than a budget-limited null because the weight
actually trained: `row_modulation_output.weight` (stored as `arr_6`, shape
(512, 300)) has L2 **1.24**, absmax 0.027, and all 153,600 entries non-zero —
about 7% of the row head's own norm. Always check the checkpoint norm before
recording a zero-initialised term as refuted.

### 2. Model confidence localizes the crop window

19 evaluation queries have a target that is an exact sub-grid of the query input;
**18 are on single-query tasks**, where one exact grid increments all four
metrics at once (+4 cumulative each). Only **2** sit at the top-left origin the
decoder is hardcoded to read, which is exactly why the crop machine caps near 4.

If the model's own per-cell confidence were higher over the true crop window, the
offset would be recoverable with a decoder change and no retraining. Scoring
every candidate window by mean top1-top2 margin, by mean max logit, and by
negative entropy:

| statistic | correct window |
|---|---|
| margin | **0/19** |
| max logit | **0/19** |
| entropy | **0/19** |

**Refuted.** The offset is not present in the logits; it would have to be learned.

## Ceilings

| policy | cumulative |
|---|---|
| perfect copy (submit the input verbatim) | 0 |
| best measured model-only run (`fixed-s31337`) | **4** |
| oracle shape + the current colour head | 6 |
| decode policy, over fixed logits | 6 |
| post-hoc copy prior on the carrier-connected head | 2 |

Every decode-side manipulation saturates between 2 and 6. Reaching 10 requires
the model to emit correct non-copy content, which no current component does.

## What this points to

The consistent picture across all five null results on record — width sweep,
training budget, learned keys, `etp_outer_write`, and now the bilinear term — is
that they all enriched *what the carrier contains* or *how the head reads it*,
while the head had already collapsed onto copying. The binding constraint is
upstream of the heads: the demonstrations do not measurably influence the query
output (shuffled-demonstration deviation is approximately zero).

The narrowest testable target was the crop family: 19 queries, 18 of them worth
+4 each. The decode origin is hardcoded to (0, 0) and was never a learned
quantity. That proposal was investigated and is refuted below.


### 3. Learned crop-window heads

The proposal was to widen `answer_shape_head` from 60 to 120 outputs so it emits
an origin `(r, c)` alongside the extent `(h, w)`, supervised by a masked loss on
episodes where the target is a unique sub-grid of the input, with the decoder
reading `canvas[r:r+h, c:c+w]`. Reading the canvas rather than the input keeps
the channel model-only, and is well-posed because the canvas is a full copy of
the input.

Three measurements refuted it before implementation.

**Origin alone converts nothing.** The shape head is 2/19 on the crop family, and
the intersection of "extent already correct" with "needs a non-zero origin" is a
single query, `bf699163`. Extent and origin would have to be learned jointly.

**Label support is thin and concentrated.** Across the 400-task training split:

| | pairs | share |
|---|---|---|
| demonstration pairs | 1302 | |
| target is a sub-grid of the input | 114 | 8.76% |
| ... with a unique matching window | 76 | 5.84% |
| ... unique and area >= 4 (usable) | 74 | 5.68% |

Those 74 labels come from only **27 distinct tasks**. Dihedral and colour
permutation augmentation re-present the same 27 concepts rather than creating new
ones. 16 of the 27 (43 pairs) follow a marker-bbox rule, and 39 pairs teach
"bounding box of some colour" — which colour permutation collapses into one
colour-invariant concept, and which is exactly the rule the winnable evaluation
queries need. That was the strongest form of the case for building.

**The function is not expressible on the family, let alone learnable.** An
offline probe built the strongest version of this head that hand engineering
allows: candidate windows generated as every colour's bounding box and each
bounding box's interior, ten hand-designed features per candidate (colour
rarity, hollowness, area fraction, background flag, aspect), and a softmax
ranker trained over candidates on the training split.

| | result |
|---|---|
| candidate generator covers the eval test-query crop family | **3 / 19** |
| learned ranker top-1, on groups it can express | 8/20 = **0.400** |

Expected conversion is therefore about **one** query. The ceiling is set by
coverage, not by the ranker: 16 of the 19 crop windows are not a colour bounding
box or its interior at all. A dedicated feature-engineered model with oracle
candidate generation cannot reach the target, so a shape head computing from
row-serial input cannot either. **Refuted.**

## Conclusion

Cumulative 10 on the model-only channel is not reachable by any lever measured
here. The measured best remains **4**, and it is two top-left crop queries:
`bbb1b8b6` q1 at rank 1 and `e872b94a` q0 at rank 2.

Ten distinct paths were closed, each by measurement rather than by argument:
decode policy (ceiling 6), training budget (u700 null), width and edge count
(null), the bilinear carrier x row product (refuted, with the trained weight norm
as evidence), copy priors on the colour logits (no-op), changed-cell loss
reweighting (refuted -- the head's true-colour rank on changed cells is uniform,
so there is no diluted signal to recover), test-time dihedral augmentation
(averaging a copy machine returns the copy), confidence-based crop localization
(0/19), learned crop-window heads (refuted above), and `memory_coding="frozen"`
severing demonstration writes (refuted by code read -- the `else` branch still
calls `update_context_memory` with the write gate, so writes do happen).

The binding constraint is upstream of every one of them, but it is *not* that
the demonstrations fail to reach the network. The controls arm of
`ex21-fixed-controls-s31337` reports
`causally_null_at_measured_precision = False` for `no_context`,
`shuffled_demonstrations`, and `slot_ablation` alike, and
`shuffled_pairing_sensitive_for_every_applicable_query = True`. The
demonstrations measurably change the trajectory, and the memory is written and
read (the `frozen` branch still calls `update_context_memory` with the write
gate).

The constraint is that **the heads cannot convert that signal into correct
output**. Two regimes, both measured:

- At `row_head_carrier_scale = 0.0` the colour head is architecturally excluded
  from the carrier, so it is an exact copy of the query input and the
  demonstration-dependent state cannot reach the answer at all. This is the
  cumulative-4 configuration.
- At `row_head_carrier_scale = 1.0` the head does see the carrier and extracts a
  real but weak signal -- 10-12% rank-1 accuracy on changed cells, with the
  rank-2 guess still below the marginal colour prior. That is not enough to pay
  for the copy it destroys, and those runs score cumulative 0.

So the task information is present in the state and absent from the answer. That
is the gap every null result on record sits in: the width sweep, the training
budget, learned keys, `etp_outer_write`, and the bilinear term all changed what
the carrier contains or how the head reads it, when the deficit is the head's
ability to extract a usable rule from a carrier that already varies with the
demonstrations. A productive next experiment changes the *write* mechanism or
the head's readout objective, not its capacity -- `delta_write` and
`situ_glu_update` are the two write mechanisms never yet run.


## Write mechanisms: the two never-run codings

Run inventory by `memory_coding` before this session: frozen 33, learned_update 15,
learned_write 3, learned_keys 0, delta_write 0, situ_glu_update 0. Both untried
mechanisms were run.

**`situ_glu_update`** ran cleanly -- 14 eligibility-trace temporal routes including
`memory_situ_glu.gate_weight`, `.up_weight` and `.output_weight`, 0 warnings, 0
errors -- and scored **cumulative 0** at shape 0.5227 and pixel 0.5157. Its
changed-cell rank-1 accuracy is **0.035**, *below* the carrier-connected baseline
of 0.105-0.120, because the copy residual suppresses precisely the edits the new
write mechanism was meant to enable.

**`delta_write`** trains but diverges in the evaluation forward pass, producing
non-finite height logits. This reproduced at **lr 1e-3 and lr 2e-4** -- five times
apart -- so it is the mechanism, not the learning rate. It replaces the whole
memory state per write (`candidate_memory = encode_delta_memory_candidate(event)`)
with no decay term, which is the likely cause. Recorded as unstable in this
configuration rather than as evaluated.

## The score is a step function of copy fidelity

| configuration | input echo | changed-cell rank-1 | cumulative |
|---|---|---|---|
| copy residual 2, carrier 0, frozen | **1.000** | 0.000 | **4** |
| copy residual 2, carrier 1, situ_glu | 0.913 | 0.035 | 0 |
| copy residual 0, carrier 1, frozen u700 | 0.652 | 0.120 | 0 |

Cumulative 4 occurs at exactly one point: where the colour head is a *perfect* copy.
Every step away trades a guaranteed win for a small chance of a correct edit, and the
trade never pays -- the two winning queries are exact crops, and a 9% deviation
breaks both.

The 4 is narrower than even that suggests. Forcing `situ_glu`'s logits to a perfect
copy offline recovers only **1**, not 4: its shape head (0.5561) misses the two crop
shapes the winning run's (0.5776) happens to hit. Cumulative 4 requires a perfect copy
*and* two fortunate shape predictions -- a coincidence rather than a capability, which
is the honest characterisation of the baseline this work started from.
