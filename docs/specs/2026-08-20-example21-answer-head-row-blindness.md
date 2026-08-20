# Example 21 — answer-head row blindness and carrier scale collapse

Status: in progress
Date: 2026-08-20
Branch: `worktree-agent-aeedc790fc9afce6a`

Explains the universal all-black ARC prediction reported across the seven
`var/example21-shared-*` evaluation runs. Supersedes nothing; complements
`2026-08-18-example21-arc-score-recovery.md`, whose defect **D1 — the output is
generated from scratch** turns out to survive the row-refinement redesign in a
new form.

## 1. Problem

Seven ARC-AGI-1 evaluation runs (400 tasks / 419 queries) spanning 1024–2048
neurons, 1024–1,047,552 recurrent edges, batch 32/64, 13–1040 updates and
150–390 latent steps all score exactly `0/419` exact queries. In every run the
model emits an **all-colour-0 grid for every query**, and the predicted
`(height, width)` collapses to two or three distinct values across all 419
queries.

## 2. Measured evidence

### 2.1 The collapse is universal and shape-degenerate

From `var/example21-shared-1024n-1024e-b32-u260-l150/result.json`,
`evaluation.checkpoint_queries`:

| checkpoint | n | distinct shapes | nonzero grids | distinct log-probs |
|---|---|---|---|---|
| 0 | 419 | (1,1)×178, (10,10)×174, (10,3)×67 | 0 | 236 |
| 30 | 419 | (10,10)×342, (10,3)×77 | 0 | 419 |
| 150 | 419 | (10,10)×341, (10,3)×78 | 0 | 419 |

All 419 log-probabilities are distinct from checkpoint 30 onward, so the model
*does* respond to the query. It simply never argmaxes off colour 0.

### 2.2 The training asymptote is the unconditional ARC colour marginal

Colour marginal over all ARC-AGI-1 output grids (computed from
`/datasets/arc/raw/data`):

| split | p(colour 0) | marginal entropy |
|---|---|---|
| training | 0.5428 | 1.6289 nats |
| evaluation | 0.5310 | **1.6617 nats** |

`(10,10)` is also the modal output shape in both splits (162/1000 eval grids).

Mean top-two logit margin and mean predictive entropy at the final checkpoint,
across all seven runs, ordered by optimizer updates:

| updates | margin | entropy (nats) |
|---|---|---|
| 13 | 0.048 | 2.353 |
| 130 | 0.600 | 2.343 |
| 260 | 1.262 | 2.184 |
| 390 (2048n) | 2.104 | 1.824 |
| 1040 | 2.323 | **1.689** |

`ln(10) = 2.3026`. The 13-update runs sit at uniform. The margin then grows
**linearly** in the update count and the entropy descends to 1.689 — which, after
removing the two 30-way shape decisions that `_decision_uncertainty`
(`latent_workspace_analysis.py:953-975`) averages in alongside the 100 colour
decisions, brackets the 1.6617-nat evaluation colour marginal.

A model capturing *any* input-conditional information would sit strictly below
`H(Y)`. It does not. **Training converges to the unconditional colour marginal,
whose argmax is colour 0 in every cell.** Perfectly linear margin growth is the
signature of a gradient whose sign is constant across every example and every
step — i.e. the per-example component averages away and only the dataset-constant
component survives.

Corroboration: mean pixel accuracy restricted to the shape-correct queries is
0.5085 (1040 updates) and 0.5224 (260 updates) — the background fraction. (This
measurement is near-tautological given a constant-zero prediction and is reported
only for consistency, not as independent evidence.)

### 2.3 Root cause — the answer heads are row-blind

`latent_workspace_model.py:2246-2255` feeds both row-refinement answer heads the
LIF **membrane voltage**:

```python
carrier = (
    self.workspace_carrier.value      # := self.voltage, set at :2244
    if self.config.memory_enabled
    else self.voltage
)
carrier = _unit_l2_cap(carrier)
next_row = self.answer_row_head(carrier)
next_shape = self.answer_shape_head(carrier)
```

The membrane potential is a slow integrator. Forward-pass probe, untrained
1024-neuron model, four real ARC evaluation tasks, one 30-row refinement sweep:

| signal | across-ROW std | across-QUERY std | row/query | row-to-row cosine |
|---|---|---|---|---|
| **voltage (the carrier)** | 0.866 | 4.040 | 0.214 | **0.9914** |
| spikes | 0.141 | 0.189 | 0.746 | 0.6905 |
| feedforward current | 1.535 | 7.735 | 0.198 | 0.9906 |
| recurrent current | 0.138 | 0.754 | 0.183 | 0.9922 |

The 30 carrier vectors of one sweep are cosine-0.991 identical, and the resulting
`answer_row` logit vectors are **cosine-0.995 identical** (min 0.975). The head is
functionally a per-query constant replicated to all 30 rows.

A linear probe (ridge, held-out *queries*, 32 queries × 30 rows) confirms this is
not a scale artefact:

| carrier | 30-way ROW accuracy | query-identity accuracy |
|---|---|---|
| chance | 0.0333 | 0.0312 |
| voltage | 0.0583 | **1.0000** |
| spikes | 0.0333 | 0.9420 |
| concat(voltage, spikes) | 0.0625 | 1.0000 |

**The carrier encodes perfectly which task is being solved and essentially
nothing about which row is being written.** The row index and the row-local query
content *are* injected into the event by
`build_refinement_feedback_event` (`latent_workspace_refinement.py:400-427`,
`row_index_slice` and `input_color_slice`), but they enter only through
`ff_syn`, and the membrane integrates them away before the head reads them.

Consequence: during training the head sees ~30 mutually inconsistent row targets
through an essentially constant input. The unique consistent fixed point is the
row-independent, query-independent colour marginal — exactly the asymptote
measured in §2.2.

### 2.4 Compounding defect — the head input is attenuated by √n

`_unit_l2_cap` (`latent_workspace_model.py:1371-1382`) normalises to unit **total
L2** across all `n` neurons, so the per-coordinate magnitude is `1/√n`. The heads
(`:1571-1581`) are bias-free (`b_init=None`) and initialised at `1/√n`:

```python
row_weights = random.randn(config.neuron_count, MAX_GRID_SIZE * COLOR_COUNT)
row_weights = row_weights / math.sqrt(config.neuron_count)
```

For `W_ij ~ N(0, 1/n)` and `‖c‖ = 1`, `Var(logit) = 1/n`, so `std(logit) = 1/32`
at n=1024. Measured:

| quantity | predicted | measured |
|---|---|---|
| carrier per-coordinate RMS | 0.03125 | **0.03125** |
| answer_row logit std | 0.03125 | **0.03020** |
| colour softmax entropy | 2.3026 | **2.3022** |

The softmax is indistinguishable from uniform at initialisation, and the
optimizer must spend its entire budget merely reaching O(1) logits. This explains
the pre-asymptote regime (13–260 updates), where the argmax is decided by a
noise-level tilt. It is **not** the root cause of the asymptote — Adam is
invariant to gradient scale, so this defect changes learning *speed*, not the
prior-versus-conditional ratio.

### 2.5 What is explicitly not a defect

`color_factor_head`, `height_head`, `width_head` and `readout_projection` report
L2 delta 0.0 and emit "state not found in the compiled model" warnings. This is
by design: `21-latent-reasoning-in-context.py:4696-4703` whitelists them as
`legacy_plain_paths` and the active `decoder_mode="row_refinement"` never calls
them. Verified: the ETP compiler classifies both live heads as
`all_direct` and includes them in `param_states`.

```
=== compiled param_states ===
   answer_row_head.weight
   answer_shape_head.weight
   ...
   relation_included | info | answer_row_head.weight
      etp_mm(('answer_row_head', 'weight')) -> [0]
         answer_row -> all_direct
```

## 3. Defect statement

**D-RB (row blindness, root cause).** The row-refinement answer heads read a
slow-integrator carrier from which the refinement row index is not recoverable
(row-to-row cosine 0.991; linear row decodability 0.058 against 0.033 chance,
versus 1.000 for query identity). A bias-free linear head over such a carrier can
only emit one row vector per query, so the supervised fixed point is the
row-independent colour marginal and every predicted grid is all-black.

**D-SC (scale collapse, compounding).** `_unit_l2_cap` scales the head input to
`1/√n` per coordinate against `1/√n`-initialised bias-free heads, giving
initialisation logits of std 0.030 and a softmax within 0.0004 nats of uniform.

## 4. Fix

### 4.1 D-RB — condition the answer heads on the refinement row

Widen the answer-head input from `carrier` alone to

```
[ scaled_carrier , row_one_hot , query_row_colours ]
```

where `row_one_hot` (30) and `query_row_colours` (300) are sliced directly from
the refinement feedback event already built at
`latent_workspace_model.py:2144-2153`. Both are query-independent-by-construction
row codes; `query_row_colours` additionally supplies the row × query interaction
a purely additive row bias cannot express. Head input width becomes
`neuron_count + 330`.

The heads are gated to refinement-latent ticks only
(`refinement_gate`, `:2256-2262`), so reading the feedback event is well-defined
wherever the head output is retained.

### 4.2 D-SC — scale the head carrier to unit RMS

Introduce a head-path-only normalisation that scales the capped carrier to unit
**RMS per coordinate** rather than unit total L2, giving O(1) initialisation
logits. `_unit_l2_cap` itself is left untouched at its other two call sites
(`:2060` legacy compact readout, `:2198` `workspace_query_projection`), which are
not part of this defect.

## 5. Tests (co-located, suffix style)

`examples/pp_prop/latent_workspace_model_test.py`:

1. **`test_row_refinement_answer_rows_are_conditioned_on_the_refinement_row`** —
   run one 30-tick refinement sweep on an untrained model and assert the mean
   pairwise cosine between the 30 `answer_row` logit vectors is below a
   threshold. Reproduces D-RB: currently 0.995.
2. **`test_row_refinement_head_carrier_has_unit_rms_scale`** — assert the head
   input's per-coordinate RMS is O(1), not `1/√n`. Reproduces D-SC: currently
   0.03125 at n=1024.
3. **`test_row_refinement_heads_remain_all_direct_after_row_conditioning`** —
   assert `answer_row_head.weight` and `answer_shape_head.weight` are still in
   `compile_pp_prop(model).param_states` and still classify `all_direct`.
   Guards the trap that a widened head input silently drops out of the compiled
   model and trains with a zero gradient.

## 6. Expected outcome — stated in advance

Removing row blindness lets the decoder emit row-varying, query-conditioned
grids, which should end the all-black collapse (`nonzero_grids > 0`) and the
two-value shape distribution. **It is not expected to produce a nonzero exact ARC
score at the 260-update operating point**, and this spec does not claim it will.
The reported result will be the collapse metrics, plus the exact score whatever
it is.
