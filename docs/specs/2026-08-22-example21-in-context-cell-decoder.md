# Example 21: in-context cell decoder

**Status:** draft, gated on the offline proxy measurement in §4.
**Goal:** raise the Example 21 *model-only* ARC-AGI-1 evaluation score
(`q@1 + q@2 + strict@1 + strict@2`) from its measured ceiling of 6 to 20.

## 1. Why the present decoder cannot reach 20

Three measurements, already on `main`, bound the current design:

- The colour head's argmax equals the query input on 113,982 / 113,982
  overlapping cells. It is an exact identity map, not a weak learner.
- A copying head can only emit a sub-grid of its input. Exactly two evaluation
  queries have a target that is a top-left crop of the query input, so the
  attainable maximum is cumulative **6**, and the best run attains it.
- The crop origin is unlearnable three independent ways (model confidence
  0/19, hand-built bbox ranker 3/19, demonstration memorisation 0/19), so
  widening the crop family is closed.

The mechanism is structural, not a training deficiency. `_refinement_head_input`
concatenates the LIF carrier with three event-derived blocks: a row-position
one-hot, the query colours *of the row being transcribed*, and the query's input
dimensions. The head is a bias-free linear map. Therefore:

- the head sees **no vertical context** — nothing above or below the current
  row — so no transformation with two-dimensional structure is representable;
- the demonstrations reach the head **only through the carrier**, a 4096-unit
  LIF membrane whose row-to-row cosine over one sweep is 0.99. It identifies
  the task and carries essentially no cell-level demonstration content;
- 81.6% of same-shape target cells equal the input, so "copy" is a free 81.6%
  and the only reachable optimum.

## 2. What the score actually requires

Cumulative 20 is five single-query tasks answered exactly at rank 1 (each
contributes q@1 + q@2 + s@1 + s@2 = 4), or an equivalent mixture.

Measured on the 419-query evaluation split:

| family | count | note |
|---|---|---|
| output shape == input shape | 277 | the reachable family for a cell decoder |
| same shape, <= 12 cells differ | 39 | 33 of them single-query |
| target is a sub-grid of the input | 19 | only 2 at the top-left origin |

Reading the near-copy tasks shows what a decoder must compute. `6ea4a07e` is a
per-cell recolour under a demonstration-inferred colour map (mask inversion,
8->2 / 3->1 / 5->4). `4cd1b7b2` completes a Latin square: each blank takes the
colour missing from its row and column. `27a77e38` places the most frequent
colour of the region above a separator at the bottom-centre cell. None is
expressible from one row plus a task vector; all are expressible from a cell's
local patch, its row/column/global colour statistics, and a lookup against the
demonstration cells.

## 3. Design

Two changes, both inside the model, both trained by pp-prop through ETP ops.

### 3.1 Demonstration cell bank

The episode encoding already places each demonstration's input row and output
row in the *same* row event (`input_color_slice` and `output_color_slice`), and
the model already accumulates the query's full grid in the `query_grid`
short-term state. Add the symmetric stores for demonstrations:

- `demo_input_grid`  — `(batch, D, 30, 300)`
- `demo_output_grid` — `(batch, D, 30, 300)`
- `demo_valid`       — `(batch, D, 30)`

written during the demonstration phase exactly as `capture_query_rows` writes
the query. At `D = 4` this is 4.6 MB at batch 32.

### 3.2 Cell features and the in-context row head

Define a per-cell feature map `phi(grid, r, c)` computed inside the model from a
stored grid: the 5x5 colour patch with an out-of-grid class, the row / column /
global colour presence and normalised counts, and position and edge flags.

At refinement tick `r` the row head emits the 30 cells of row `r` as

    logits[c] = direct(phi(query, r, c)) + alpha * log( attention_pool + eps )

where the attention pools demonstration *output* colours over demonstration
cells, keyed by `phi` of the demonstration *input* cell:

    w[c, n] = softmax_n( <Wq phi(query, r, c), Wk phi(demo_in, n)> * s )
    attention_pool[c] = sum_n w[c, n] * onehot(demo_out_colour[n])

`Wq`, `Wk` and `direct` are `braintrace.nn.Linear` (ETP-marked). The pool is a
mask-weighted sum with no reshape between an `etp_mm` and a hidden group, so the
pp-prop position-preserving requirement is met. Cost per tick at batch 32,
30 query cells and 3,600 demonstration cells is 3.5M dot products — the same
order as the existing row head.

Copying stays reachable: the centre cell of the patch is in `phi`, so `direct`
can learn the identity. It is no longer the *only* reachable map.

### 3.3 Shape head

`out_shape == in_shape` for every demonstration, or a constant demonstration
output shape, predicts the evaluation shape correctly on **355 / 419** queries
(100% precision when it fires) against the current head's 0.578. Feed the shape
head the per-demonstration input and output dimension one-hots and make it
two-layer so the selection between the two relations is representable. The head
still *learns* the relation; it is not handed a rule's answer.

## 4. Gate

An offline PyTorch proxy of §3.2 — same features, same attention, no LIF —
trained on the ARC-AGI-1 training split and evaluated on the evaluation split
decides whether to build this. A first 78-second proxy already answers
`6ea4a07e` exactly on both queries, which is cumulative 6 on a task the present
architecture cannot express. The gate for the port is **proxy cumulative >= 20**.

If the proxy cannot reach 20, this document records the bound and the port is
not built.

---

## 5. Measurements that redirected the design (2026-08-22)

### 5.1 The training-data ceiling was never real

`APPROVED_TRAINING_SOURCES` already contains `re-arc`, `conceptarc`, `arc-heavy`
and `arc-gen100k`. Every Example 21 run to date used only `arc-agi-1 training` —
400 tasks. RE-ARC's generators produce verified examples for all 400 training
concepts: **56,000 examples in 335 s**, 36,280 of them same-shape across 261
tasks.

### 5.2 The shape head is representational, not capacity-bound

| predictor | eval shape accuracy |
|---|---|
| current model shape head | 0.578 |
| MLP over raw demonstration dimension one-hots | 0.530 (train loss 7e-4) |
| MLP over signed differences + equality flags, offset-mixture output | **0.730** |
| rule: demos all same-shape -> query shape, else constant demo out-shape | 0.847 |

### 5.3 Cross-task in-context attention is not the mechanism that scores

Three offline variants, all evaluated on the real 419-query split:

| variant | cumulative |
|---|---|
| cross-attention over demonstration cells, ARC-train only | 6 (`6ea4a07e`) |
| 4-block conv + cross-attention stack, ARC-train only | 2 |
| cross-attention, RE-ARC + ARC, rich features | measured separately |

### 5.4 What does score: fitting the head on the task's own demonstrations

A per-cell head **fitted on each evaluation task's own demonstration cells**,
over features that include the cell's mirror images, nearest non-background
colour along each ray, the grid's dominant period and its connected component:

| feature set | learner | query exact | strict tasks |
|---|---|---|---|
| patch + row/col/global statistics | per-task trees | 2 | 1 |
| + mirrors, rays, periods, components | per-task trees | **10** | **8** |

Ten exact evaluation queries over eight tasks. This clears the cumulative-20
target with margin, and the mechanism is **online adaptation on the
demonstrations** — braintrace's own thesis — not cross-task pretraining.

## 6. Revised design

The decoder is a per-cell head over the §5.4 feature map, **adapted online from
the demonstration stream** and then applied to the query:

1. Per-cell features are computed inside the model from `query_grid` and the new
   demonstration grid stores. Every block is index arithmetic, an equality
   reduction or a `cummax`, so it lowers to JAX inside the refinement sweep.
2. The head is a shared per-cell map — `braintrace.nn.Linear` folds leading
   axes, so `(batch, 30, features) -> (batch, 30, 10)` is native and the weights
   are shared across cells.
3. During the demonstration phase each demonstration cell is a supervised online
   example: input-cell features in, output colour as target, ETP update applied.
   The query phase then reads the adapted head.
4. The shape head uses the §5.2 relational features.

The open risk is the **learner gap**: §5.4's 10/8 was measured with trees, which
do not port. A gradient-fitted head over the same features is the portable
learner and is measured separately; the port is sized against that number, not
the tree number.

## 7. Gate results (2026-08-22)

Per-cell head **fitted on each evaluation task's own demonstration cells**,
scored on the real 419-query split with the harness four-tuple:

| feature set | learner | q@1 | q@2 | s@1 | s@2 | cumulative |
|---|---|---|---|---|---|---|
| patch + row/col/global statistics | trees | 2 | – | 1 | – | 6 |
| + mirrors, rays, periods, components | trees | 11 | 11 | 9 | 9 | 40 |
| + mirrors, rays, periods, **no components** | trees | **12** | 12 | **10** | 10 | **44** |

Dropping connected components *raises* the score, so the block that does not
lower cleanly to JAX is not needed; `latent_workspace_cell_features.py` omits it
and reproduces the better feature set.

Three facts about that 44 that bound the claim:

- **The shape branch is the easy one.** All eleven winning tasks have
  `output shape == input shape` on every demonstration *and* on the query, so the
  shape sub-problem for the winners is exactly the relation the §5.2 head
  predicts as a zero offset. The 44 does not depend on the 0.847 rule's hard
  branch.
- **The second candidate contributes nothing.** `q@2 == q@1` and `s@2 == s@1`
  across all 400 tasks: flipping the least-confident cell to its runner-up never
  converts a miss. A real second candidate (a second fitting seed, or a
  different feature subset) is unclaimed upside.
- **Trees do not port.** The portable learner is a head adapted by gradient
  updates over the demonstration cells. That arm is measured separately; the
  port is sized against it, not against 44.

## 8. Correction: §7's numbers were label-gated (2026-08-23)

Every four-tuple in §7 came through this gate:

```python
if predicted_shape(t, gi) != go.shape or gi.shape != go.shape:
    a1 = a2 = False; continue
...
probs = probs[: gi.shape[0], : gi.shape[1]]
```

`go` is the **target**. That is not "the shape rule applied" — it checks the
rule against the label, drops every query where they disagree, and hands the
survivors their extent for free. 44, 40 and 18 are upper bounds on a
label-verified subset, reportable in neither the `model_only` nor the
`rule_then_model` channel. §7's three bullets are corrected below, and its
table should be read as a feature-set comparison only.

### 8.1 Extent has to come from the head

Re-scored with the eleventh class carrying "not part of the output grid", so
extent is decoded from the fitted head and no target is read anywhere. Full
419-query evaluation split, harness two-candidate decode:

| learner | features | q@1 | q@2 | s@1 | s@2 | cumulative | shape |
|---|---|---|---|---|---|---|---|
| shipped carrier-row head | carrier + one query row | 2 | 2 | 1 | 1 | 6 | 242/419 |
| gradient MLP, 600 updates | 482-dim, one-hot | 0 | 0 | 0 | 0 | 0 | — |
| 1-nearest demonstration cell | 482-dim, one-hot | 0 | 0 | 0 | 0 | 0 | 304/419 |
| softmax attention over demo cells | 482-dim, one-hot | 0 | 0 | 0 | 0 | 0 | 160/419 |
| exact 5x5→3x3→1x1 backoff lookup | raw colours | 2 | 2 | 2 | 2 | 8 | 295/419 |
| sklearn tree | 482-dim, continuous | 14 | 14 | 10 | 10 | **48** | 353/419 |
| sklearn tree | 482-dim, binarised at 0.5 | 11 | 11 | 9 | 9 | 40 | 320/419 |
| **shipped JAX forest, depth 12** | 482-dim, binarised at 0.5 | **9** | 9 | **7** | 7 | **32** | 320/419 |

The learned extent is not the weak link: 353/419 = 0.843 against 355/419 =
0.847 for the hand-written shape rule. A head that learns where the grid ends
from the task's own demonstrations is as accurate as the rule and owes it
nothing.

### 8.2 The three corrected bullets

- **Trees do port.** `latent_workspace_demonstration_forest.py` is an
  axis-aligned tree in JAX: one segment sum plus one argmax per level, node
  arrays sized `2 ** depth`, class counts read back along the whole
  root-to-leaf path so an unreached leaf backs off to its ancestors. Against
  sklearn on identical binary features it fits demonstrations to 0.992 (sklearn
  0.995) and reaches 0.946 query-cell accuracy (sklearn 0.944).
- **The learner was the whole gap, not the features.** Over one feature map the
  same-extent spread runs 0 (gradient MLP) to 48 (tree). The gradient head was
  not underfitting a hyperparameter; it was carrying the wrong inductive bias
  for exact conjunctive rules over one-hot cells.
- **The second candidate still contributes nothing**, and now the reason is
  visible: a purity-grown tree returns near-degenerate probabilities, so the
  harness's "second-ranked shape or one colour substitution" rule has nothing
  to trade. The one arm where `q@2 > q@1` (4→6, 3→5) was an eight-tree
  ensemble, whose smoother probabilities cost more on q@1 than they bought on
  q@2. A second candidate that pays is still unclaimed upside.

### 8.3 Known cost, not yet paid

Binarising at 0.5 costs 48 → 40 (shape 353 → 320): the position, extent and
colour-count columns are ordered quantities a 0.5 threshold cannot split
usefully. Threshold codes (`q >= k` for each level) recover the shape accuracy
(110/126 against 109/126 on a 120-task probe) but only part of the colour
accuracy, because the per-row and per-column colour counts need the same
treatment. The fix is a wholly binary feature map, not a change to the learner.
