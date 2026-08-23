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
