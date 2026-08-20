# Example 21 — why the row head does not learn a rule it can express

Status: implemented; three defects fixed, exact score reported honestly in §8

Originally filed as "parameter-travel starvation and answer-shape blindness".
That title names the two hypotheses this investigation started from; measurement
demoted both. Travel starvation (D-PT) is confirmed for the shape path and
unresolved for the colour path; shape blindness (D-SB) is a real expressivity
gap whose repair changed nothing measurable. The load-bearing results are §8.4 --
a 110-parameter lookup table the head can express exactly beats it by 0.13 pixel
accuracy -- and §8.6, the colour-loss height normalisation (D-GH) that produced
the first exactly-correct colour content in this investigation.

Date: 2026-08-20
Branch: `worktree-agent-ace1b26e7a1ca359f`

Explains why exact pass@1 stayed at `0/419` after the row-conditioning fix in
`2026-08-20-example21-answer-head-row-blindness.md`, which broke the all-black
collapse (0 → 170 non-black grids, 2 → 90 distinct shapes) without moving the
score. Complements that spec; supersedes nothing.

## 1. Problem

`var/rowcond-1024n-1024e-b32-u260-l150` (1024n / 1024e / b32 / u260 / l150,
seed 2108, full ARC-AGI-1 evaluation, 400 tasks / 419 queries):

| metric | value | trivial floor |
|---|---|---|
| exact pass@1 | **0 / 419** | — |
| shape diagnostic | 0.3604 | `copy_or_rule_shape` 0.8687 |
| pixel diagnostic | 0.3806 | `copy_input` 0.6032 |

Both diagnostics are below predictors that use no neurons at all.

## 2. Measured evidence

All figures in this section are computed from
`var/rowcond-1024n-1024e-b32-u260-l150/result.json` joined against
`/datasets/arc/raw/data/evaluation`, on the post-fix code at commit `0b31dfb`.

### 2.1 Exact is gated by colour exactness, not by shape

Decomposing the conjunction `exact = (shape correct) AND (every cell correct)`:

| condition | count |
|---|---|
| shape correct (either candidate) | 151 / 419 |
| **exact under an oracle shape** (crop model colours to the true shape) | **1 / 419** |
| exact under oracle colours (i.e. shape-correct) | 151 / 419 |

The single query that an oracle shape would make exact is `e872b94a` q0, whose
true output is the degenerate `[[0],[0],[0]]` — a 3×1 all-black grid. Every
other query fails on colour content, not on shape.

Distance to exact among the 151 shape-correct queries, in wrong cells:

| wrong cells | queries |
|---|---|
| ≤ 3 | 1 (`e345f17b` q0, 3 of 16) |
| ≤ 8 | 12 |
| 0 | **0** |

**Conclusion: the shape gate is not what pins the score at zero.** Colour
content is nowhere near exact on any query.

### 2.2 The model has not learned even the identity rules

| rule | model | dataset frequency |
|---|---|---|
| output shape == input shape | predicted for 224/419 queries (0.535) | true for 277/419 (0.661) |
| shape accuracy on same-shape queries | 146/277 = **0.527** | 1.000 for a copy rule |
| shape accuracy on different-shape queries | 5/142 = 0.035 | — |
| pixel accuracy | 0.3806 | `copy_input` 0.6032 |

The row head's input already contains the query colours of the row being
written (`latent_workspace_model.py:1429-1431`, `input_color_slice`), so a
per-cell copy **is** expressible as a bias-free linear map. The model does not
express it. This is an optimisation failure, not an expressivity failure.

### 2.3 Root cause — the answer heads cannot travel

Adam's per-coordinate displacement per step is bounded by the learning rate:
`Δθ = lr · m̂/(√v̂ + ε)` and `|m̂/√v̂| ≤ 1` in the limit of a consistent gradient
sign. So over `U` updates,

```
max |θ_U − θ_0|  ≤  lr · U
```

For the shipped operating point `lr = 1e-4`, `U = 260`:

```
travel bound      = 1e-4 × 260            = 0.0260
head init σ       = 1/√head_width
                  = 1/√(1024 + 330)       = 0.0272     (latent_workspace_model.py:1640-1644)
travel bound / σ  = 0.956
```

**No single weight in either answer head can move even one initialisation
sigma.** The bound is per coordinate and it is loose, not tight — the whole
matrix does not travel at the bound — so the honest reading is that the trained
head remains in the regime where its random initialisation still dominates its
output, and the decoder emits the random prior plus whatever dataset-constant
tilt survives averaging. That is exactly the colour marginal reported in §2.2 of
the row-blindness spec.

Measured Frobenius deltas corroborate the bound independently
(`training.parameter_changes` in the same artifact). Both heads are bias-free
with `w ~ N(0, 1/head_width)`, so the initialisation Frobenius norm of a
`head_width × out` matrix is exactly `√out`:

| head | shape | ‖W₀‖_F | ‖ΔW‖_F | relative travel |
|---|---|---|---|---|
| `answer_row_head.weight` | 1354 × 300 | √300 = 17.32 | 2.0588 | **0.119** |
| `answer_shape_head.weight` | 1354 × 60 | √60 = 7.746 | 1.8790 | **0.243** |

These are aggregate, and they are well below the per-coordinate bound: if every
coordinate travelled at 0.956 σ the relative figure would be near 1.0, not 0.12
and 0.24. The bound rules out large travel; the measurement shows actual travel
is an order of magnitude smaller still. Both point the same way, and neither
says the head did not move at all.

The training loss confirms the run is not converged: mean loss falls 0.890
(first 26 updates) → 0.614 (last 26) and is still descending noisily at the last
update.

### 2.4 Compounding defect — the answer heads are blind to the query's shape

`build_refinement_feedback_event` writes the query's input height and width as
30-way one-hots into the refinement event
(`latent_workspace_refinement.py:404-405`, `input_height_slice` /
`input_width_slice`). `_refinement_head_input`
(`latent_workspace_model.py:1427-1431`) reads only `row_index_slice` and
`input_color_slice` from that event:

```python
row_position = event[:, layout.row_index_slice] * math.sqrt(MAX_GRID_SIZE)
row_colors = event[:, layout.input_color_slice] * math.sqrt(COLOR_COUNT)
return jnp.concatenate((carrier, row_position, row_colors), axis=-1)
```

So the input height and width reach the answer heads only after the LIF membrane
has integrated them — the same path the row-blindness spec measured as washing
out per-tick structure (row/query std ratio 0.214, row-to-row cosine 0.991).

This matters most at the one tick where the shape head is supervised. The shape
loss fires only on the completed sweep, `row_indices == 29`
(`latent_workspace_refinement.py:694`). The colour block is written gated by
`input_row_valid` (`latent_workspace_refinement.py:419-423`), which is
`_soft_column_mask(input_height)` evaluated at the current row — exactly 1 for
`row < input_height` and exactly **0** beyond it. So for every query whose input
is shorter than 30 rows, the entire 300-wide colour block of the head input is
identically zero at row 29.

At its only supervised tick the shape head therefore sees the carrier, a row-29
one-hot that is the same for every query, and 300 zeros. **The carrier is the
only query-varying signal reaching the shape head where it is supervised**, and
the head must recover the output height and width from it alone.

With the input-shape one-hots present, `output_shape = input_shape` — true for
66.1% of evaluation queries — is a 30 × 30 identity block in the head weight, and
is exactly representable. Without them it is not directly representable at all.

**Stated in advance: this fix is necessary but not sufficient.** §2.1 shows that
an oracle shape yields 1/419 exact. A shape fix alone cannot move the score off
zero and will not be run as a solo arm.

### 2.5 What is explicitly not the cause

- **`clip_norm`.** Global-norm clipping rescales the whole gradient tree
  uniformly, and Adam's per-coordinate step is invariant to gradient scale. It
  cannot change the travel bound. No sweep arm is spent on it.
- **`balanced_color_loss`.** It is rejected for `row_refinement`
  (`21-latent-reasoning-in-context.py:412-414`) and
  `row_refinement_loss_per_example` takes no class-weighting argument
  (`latent_workspace_refinement.py:603-608`). The prior spec named it as the
  next lever. It is **not** pursued here: §2.1 shows the failure is not a
  background-class bias in a nearly-correct predictor but a head that has barely
  left its initialisation, and reweighting a loss changes gradient *direction*,
  not the travel bound that §2.3 identifies.
- **Task-local adaptation being a no-op.** It is not. The default
  `adaptation_update_schedule="per_tick"` drives `learner.etrace_online` with a
  mask that is positive on every latent tick
  (`latent_workspace_arc_adaptation.py:883-888, 999-1008`), so it performs one
  optimizer step per latent tick per fold: at `latent_steps=150`,
  `adaptation_epochs=2` and ~9 folds that is ~2700 steps, a travel bound of
  `5e-5 × 2700 = 0.135` ≈ 5 σ — more travel than the entire base training. Task
  adaptation therefore has adequate travel already, consistent with the earlier
  finding that sweeping adaptation steps and rates left exact flat because the
  pretrained model was the binding constraint.
- **The four zero-delta heads** (`color_factor_head`, `height_head`,
  `width_head`, `readout_projection`) are whitelisted `legacy_plain_paths`
  (`21-latent-reasoning-in-context.py:4696-4703`) and unused by the active
  decoder.

## 3. Defect statements

**D-PT (parameter-travel starvation, root cause).** At the shipped operating
point the Adam displacement bound `lr × updates = 0.026` is 0.96 of the answer
heads' initialisation sigma `1/√1354 = 0.0272`, so no head weight can move one
sigma. Measured relative travel is 0.119 (row head) and 0.243 (shape head). The
decoder therefore remains a perturbed random initialisation and scores below
predictors that use no neurons, even though the copy rule its inputs make
expressible would beat both trivial floors.

**D-SB (answer-shape blindness, compounding).** The query's input height and
width one-hots exist in the refinement event but are absent from the answer-head
input, and at the shape head's only supervised tick (row 29) the colour block it
does receive is padding. The identity shape rule, true for 66.1% of evaluation
queries, is not representable by the head.

## 4. Fix

### 4.1 D-PT — raise the travel budget, and make it auditable

The travel budget is already CLI-configurable (`--learning-rate`,
`--training-updates`); the defect is that nothing surfaces when it is set below
the scale at which learning is possible, so seven runs were executed at an
operating point that arithmetic rules out.

Add `adam_parameter_travel_budget(learning_rate, updates, head_width)` to
`latent_workspace_analysis.py`, returning the displacement bound, the head
initialisation sigma and their ratio, and record it in the run artifact under
`training.parameter_travel_budget` together with the `head_width` it was
computed against. A ratio below 1 is reported as `starved`. Recording the width
is what lets a later reader compute the arithmetic for an artifact without
knowing which revision of the head produced it.

The operating point itself is chosen by sweep on a held-out fold of the ARC
**training** split, never on evaluation output (§6).

### 4.2 D-SB — condition the answer heads on the query's grid shape

Widen `_refinement_head_input` from

```
[ unit-RMS carrier , row_one_hot(30) , query_row_colours(300) ]
```

to

```
[ unit-RMS carrier , row_one_hot(30) , query_row_colours(300) ,
  input_height(30) , input_width(30) ]
```

Head input width becomes `neuron_count + 390`. Both new blocks are already
one-hot-shaped in the event and are scaled to unit root-mean-square on their
occupied coordinates by `√30`, matching the treatment the row one-hot already
receives.

This changes the head initialisation sigma, and therefore the travel arithmetic
of §2.3, from `1/√1354 = 0.02718` to `1/√1414 = 0.02659` at 1024 neurons. Every
pre-fix figure in this spec is stated against 1354 and every post-fix figure
against 1414; the two must not be mixed. The change is 2% and does not affect
any conclusion.

## 5. Tests (co-located, suffix style)

`examples/pp_prop/latent_workspace_model_test.py`:

1. **`test_refinement_head_input_separates_the_query_grid_shape`** — assert
   `_refinement_head_input` responds to the event's `input_height_slice` and
   `input_width_slice`. Exact, deterministic, no threshold. Reproduces D-SB:
   before the fix the head input is bit-identical for two events differing only
   in the query's input shape.
2. **`test_refinement_head_width_counts_the_query_shape_blocks`** — pin
   `refinement_head_width(n) == n + 390`.
3. **`test_refinement_heads_stay_compiled_all_direct_with_shape_conditioning`** —
   assert `answer_row_head.weight` and `answer_shape_head.weight` remain in
   `compile_pp_prop(model).param_states` and still classify `all_direct` after
   the widening. Regression guard against the trap that a widened head input
   silently drops out of the compiled model and trains with a zero gradient.

`examples/pp_prop/latent_workspace_analysis_test.py`:

4. **`test_adam_parameter_travel_budget_flags_a_starved_operating_point`** —
   assert the shipped operating point `(1e-4, 260, 1024)` reports
   `starved=True` with a ratio below 1, and that a raised budget does not.
   Reproduces D-PT with the exact arithmetic of §2.3.
5. **`test_adam_parameter_travel_budget_rejects_nonpositive_inputs`** — argument
   validation.

## 6. Selection protocol — pre-registered

Arm selection runs on a held-out fold of the ARC **training** split, built as a
probe manifest (`var/probe/manifest.json`): the first 315 accepted training
tasks are the training source and the last 84 tasks (91 queries) are declared
with `role="evaluation"`. The ARC evaluation split is never used for selection;
it is run once, with the selected arm, at the end.

The probe surface tracks the evaluation surface: the shipped operating point
scores shape 0.4945 / pixel 0.4370 there against 0.3604 / 0.3806 on evaluation.
It is somewhat easier, so probe figures are not a prediction of evaluation
figures — only an ordering of arms.

**Selection metric, fixed before any arm beyond the baseline was read.** The
probe surface has `exact under oracle shape = 0/91` and its closest query is
`dc433765` q1 at 2 wrong cells of 9, so exact will be 0/91 on every arm and the
wrong-cell counts over its 18 tiny-output queries move in single-query steps —
too coarse to rank arms at n=91. Arms are therefore ranked by:

1. **valid-cell pixel accuracy** (dense: ~91 grids × hundreds of cells);
2. **shape accuracy** as the tiebreak, when pixel accuracy is within 0.01;
3. exact and the wrong-cell distribution are **reported as observation, never
   used to rank**.

`--learning-rate` and `--training-updates` are swept; `clip_norm` is not, for
the reason in §2.5.

### 6.1 Amendment, recorded before the `lr = 1e-2` arm was read

Rules 1–3 above were fixed after the baseline arm and applied unchanged to
`lr = 1e-3` and `lr = 3e-3`. Reading those two exposed a flaw: **exact is
conjunctive in shape and colour**, so ranking on a single marginal metric can
select an arm that is strictly worse for the objective — `3e-3` scores the best
pixel accuracy of the three (0.4657) while its shape accuracy collapses to
0.4505, below even the baseline's 0.4945. The 0.01 tiebreak band happened to
catch that case, but only by luck of the margin.

The rule is therefore amended, **before the `lr = 1e-2` arm was inspected**, by
adding a disqualification that cannot be tuned after the fact:

> 0. An arm whose shape accuracy falls **below the baseline arm's 0.4945** is
>    disqualified regardless of its pixel accuracy, because a shape regression
>    reduces the number of queries that can be exact at all.

Rules 1–3 then rank whatever survives. This is disclosed rather than applied
silently: it is a post-hoc amendment, and the reader should weigh it as one.

## 7. Expected outcome — stated in advance

Raising the travel budget should move both diagnostics toward and possibly past
their trivial floors, because the copy rules that beat those floors are
expressible by the current heads and are simply unreachable within 0.96 σ of
travel. **Whether that produces a nonzero exact score is not predicted here.**
§2.1 measures that no query is within fewer than 3 wrong cells of exact, so a
nonzero exact score requires a qualitative improvement in colour content, not a
marginal one. The reported result will be the diagnostics, the exact score
whatever it is, and the per-cell wrong-count distribution on the 26 evaluation
queries whose outputs have ≤ 9 cells — the only combinatorially reachable exact
targets, and a leading indicator that aggregate pixel accuracy (dominated by
large grids) does not provide.

## 8. Verification

### 8.1 Tests

`299 passed` across `latent_workspace_analysis_test.py`,
`latent_workspace_model_test.py` and `latent_workspace_refinement_test.py` in
the container on CPU. Before the fix the four new tests failed as designed:

- `test_refinement_head_input_separates_the_query_grid_shape` — `assert not True`
  (the head input was bit-identical for events differing only in the query's
  input height, and again for the width);
- `test_refinement_head_width_counts_the_query_shape_blocks` —
  `assert 1354 == 1414`;
- both `adam_parameter_travel_budget` tests — `AttributeError`.

`test_refinement_heads_stay_compiled_all_direct_with_shape_conditioning` passed
before and after; it is a regression guard, not a reproduction.

D-GH added three more tests to `latent_workspace_refinement_test.py`. Their red
phase was verified against the pre-fix module restored from `HEAD` into a
mounted overlay, because the fix and the tests were written in the same pass:

- `test_sweep_colour_weight_does_not_scale_with_the_output_height` — 3·ln10
  against 30·ln10 before, equal after;
- `test_row_colour_loss_is_scaled_by_the_reciprocal_output_height` — 2.303
  against 13.816 before;
- `test_complete_sweep_balances_shape_once_and_each_valid_color_row`, an
  existing test whose golden value encoded the defect, updated with its reason.

`row_refinement_loss_per_example` has three other callers — the entry point,
`latent_workspace_arc_adaptation.py:983` and
`latent_workspace_binding_qualification.py:363,403` — and D-GH rescales its
colour term by up to 30×, so those suites were run too:
`147 passed` for the entry point and `237 passed` for
`latent_workspace_binding_qualification_test.py`,
`latent_workspace_arc_adaptation_test.py` and
`latent_workspace_binding_gate_test.py`. No caller had a golden value that the
rescale broke.

Doctests on the three changed modules pass, including the
`refinement_head_width(1024) -> 1414` example and the new
`adam_parameter_travel_budget` example block: `5 passed` for
`latent_workspace_analysis.py` and `latent_workspace_refinement.py` under
`--doctest-modules`, and `TestResults(failed=0, attempted=6)` for
`latent_workspace_model.py`.

**Not verified: the whole `examples/pp_prop` directory in one run.** It was
started and killed at 23 minutes without reporting. Every caller of the three
changed symbols is covered by the four suites above, so the marginal coverage
lost is small — but "all tests pass" is not a claim this spec makes. The four
suites named above are what was actually run and seen green.

### 8.2 Learning-rate sweep, pre-fix code, probe surface

1024n / 1024e / b32 / u260 / l150, seed 2108, 84 held-out ARC training tasks /
91 queries. `sigmas` is the §2.3 travel bound divided by the head
initialisation sigma at `head_width = 1354`.

| arm | lr | sigmas | shape | pixel | non-black | `answer_row_head` ΔL2 | mean loss, last 26 | exact |
|---|---|---|---|---|---|---|---|---|
| baseline | 1e-4 | 0.96 | 0.4945 | 0.4370 | 26/91 | 2.21 | 0.5985 | 0/91 |
| **selected** | **1e-3** | **9.57** | **0.5824** | **0.4628** | **62/91** | **8.88** | **0.5033** | **0/91** |
| — | 3e-3 | 28.7 | 0.4505 | 0.4657 | 71/91 | 22.23 | 0.5662 | 0/91 |
| disqualified | 1e-2 | 95.7 | 0.3407 | 0.4734 | 80/91 | 67.38 | 0.9934 | 0/91 |

`lr = 1e-3` wins outright under rule 1 among the arms that clear the §6.1 shape
disqualification (`1e-4` and `1e-3`); the two larger rates are disqualified for
shape collapse, and `1e-2` additionally fails to descend — its mean loss over
the last 26 updates, 0.9934, is worse than every other arm's *first* 26.

**Travel starvation was real, and it is not the whole story.** Shape, non-black
count and head displacement all respond strongly and monotonically to the travel
budget, which is what D-PT predicts. Pixel accuracy barely moves: across a 100×
range of learning rate it goes 0.4370 → 0.4734, a swing of 0.036, while a
trivial table beats it by 0.13 (§8.4).

The honest arithmetic on the colour path is that travel is **not yet
comfortable** there, contrary to a first reading of the sigma column. At
`lr = 1e-3 × 260` the displacement bound is 0.26; against the `√10` scaling of
the colour one-hot that buys a logit contribution of about `0.26 × √10 = 0.82`,
while the carrier alone contributes a logit standard deviation of about
`√(1024/1414) = 0.85` at initialisation and grows during training. So the copy
path has roughly *one* unit of the travel it needs to compete with the carrier,
not ten. D-PT is therefore neither confirmed nor falsified for colour content by
this sweep, and §8.5 tests it directly by tripling the update count at the same
step size — the one thing `lr = 3e-3` could not separate, since it reached a
comparable bound only by taking steps large enough to collapse shape.

### 8.3 D-SB (shape conditioning) — negative result

Same operating point, same seed, pre-fix versus post-fix code, probe surface:

| metric | pre-fix (n+330) | post-fix (n+390) |
|---|---|---|
| shape | 0.5824 | 0.5604 |
| pixel | 0.4628 | 0.4652 |
| non-black | 62/91 | 61/91 |
| `answer_row_head` ΔL2 | 8.883 | 9.102 |
| mean loss, last 26 | 0.5033 | 0.5014 |
| exact | 0/91 | 0/91 |

**Conditioning the answer heads on the query's grid shape changed nothing
measurable.** The expressivity gap in §2.4 is real — the identity shape rule
genuinely was not representable before, and is now — but making it
representable did not make the optimizer find it. The two runs are **not
seed-matched**: `random.randn(head_width, ...)` draws `head_width * out`
values, so widening the head from 1354 to 1414 shifts every parameter drawn
afterwards. Differences of this size (0.02 in shape, 0.002 in pixel) are within
what that reseeding can produce and **none of them should be attributed to the
fix in either direction**. What can be said is that the fix did not produce the
step change in shape accuracy §7 anticipated.

The change is kept: it is correct, it costs 60 head inputs, and it removes a
representability gap that would otherwise confound any later attempt to fix the
shape path. It is reported as a negative result, not a win.

### 8.4 What the row head could do and does not — the binding finding

The row head's colour block is indexed `column * 10 + colour`
(`latent_workspace_task.py:1817-1818`) and its output logits reshape as
`(30, 10)` on the same index (`latent_workspace_refinement.py:683`). So a
per-cell colour map `out[r][c] = f(in[r][c])` is a block-diagonal weight in the
colour block — **exactly representable by the head as it stands.**

Fitting that map as an 11 × 10 conditional table (ten input colours plus an
out-of-range class) on the demonstration pairs of the 315 probe-training tasks
and evaluating it on the 84 held-out tasks' test queries:

| predictor | pixel accuracy | mean NLL (nats) | exact under oracle shape |
|---|---|---|---|
| colour marginal | — | 1.4748 | — |
| `copy_input` (overlap only) | 0.6026 | — | 0/91 |
| **11 × 10 conditional table** | **0.6243** | **1.0029** | **1/91** |
| **trained model, best arm** | **0.4652** | — | **0/91** |

The fitted table is the identity on all ten colours and maps out-of-range to
colour 0. It is a lookup table with 110 free parameters and it uses no neurons,
no recurrence and no demonstrations.

**Weighting.** Pixel accuracy here is **query-weighted** — the mean over queries
of each query's own cell accuracy — because that is how the harness aggregates
`valid_cell_pixel_accuracy_diagnostic` (`latent_workspace_analysis.py:654`,
`sum(...) / query_count`). An earlier revision of this spec quoted the table at
**0.7734**, which is the *cell-weighted* figure (total matched cells over total
cells) and is not comparable to any model number in this document. The
cell-weighted figure is larger because the table does relatively better on the
large grids that dominate a cell-weighted mean. Every table figure here is
query-weighted; the correction roughly halves the reported gap and does not
change its direction.

**The exact column is under an oracle shape for every row, including the
model's**, because the table predicts no shape at all.
Compared like for like, the table scores one exact answer the model does not:
`d10ecb37` q0, whose 2 × 2 output is the top-left corner of a 4 × 8 input, so
the identity map reproduces it cell for cell. That is not a shape result — it is
a colour-content result, and the model's colour content does not reach it on any
query.

The model is 0.13 below a solution its own architecture can express. That is the
defect that pins the exact score at zero, and neither travel starvation (D-PT)
nor shape blindness (D-SB) explains it — the head has the input, has the
representational form, and is not visibly starved for the shape path where the
same travel produced large movement.

### 8.5 D-GH — colour gradient is reweighted by output height

Found while asking why the head does not find the table.

`row_refinement_loss_per_example` zeroes the colour term for any row beyond the
target height (`latent_workspace_refinement.py:689-693`):

```python
valid_row = row_indices <= target_height
valid_colors = valid_row[:, None] & valid_columns
color_loss = jnp.sum(jnp.where(valid_colors, color_nll, 0.0), axis=-1)
color_loss /= jnp.maximum(jnp.sum(valid_colors, axis=-1), 1)
```

and the per-tick batch reduction divides by the number of **supervised ticks**,
which does not depend on row validity
(`21-latent-reasoning-in-context.py:2160-2165`):

```python
return jnp.sum(jnp.where(supervised, losses, 0.0)) / jnp.maximum(
    jnp.sum(supervised), 1
)
```

Over one 30-row sweep an example with a 30-row target contributes a colour term
on 30 ticks and an example with a 3-row target on 3. The colour term is already
averaged over valid *columns*, so width is normalised and height is not:
**every example's colour gradient is weighted by its output height, up to 10×
between the extremes, inside the batch mean.** This is a relative reweighting
between examples, not a global scale, so Adam's scale invariance does not remove
it — it changes the direction of the summed gradient.

It bites precisely where exactness is reachable. The only combinatorially
reachable exact targets are the small outputs, and those are the examples
weighted down. On the probe surface the four tiny-output queries the best arm
gets shape-correct sit at 2, 3, 6 and 7 wrong cells of 9 — pixel 0.78, 0.67,
0.33, 0.22 against an aggregate of 0.4652, so small grids are not outperforming
despite having fewer cells to get right.

**Fix.** Scale the colour term by `MAX_GRID_SIZE / (target_height + 1)` so that
each example contributes the same total colour weight over a completed sweep,
independent of its output height. The shape term is untouched.

**The alternative that was considered and not taken** is scaling the *whole*
per-example loss by `30 / (target_height + 1)`. The difference matters: the
shipped fix takes an example's per-sweep total from
`2·ln30 + (h+1)·c` to `2·ln30 + 30·c`, so it also shifts the shape-to-colour
balance — for a 3-row target the colour term grows 10× against an unchanged
shape term, weakening shape supervision exactly on the short outputs where the
reachable exact targets live. Scaling the whole loss would preserve that balance
while still equalising examples. The colour-only form is shipped because the
measured defect is specifically that *colour* supervision is height-weighted
while *shape* supervision already fires exactly once per sweep for every example
regardless of height, so the shape term was never the asymmetric one. §8.6
reports shape accuracy on the ≤9-cell subset specifically, which is where this
choice would show up if it is wrong.

Loss values are **not comparable across this change**: the colour term is
rescaled by up to 30×, so a `probe-gh-*` arm's reported loss cannot be put in
the same column as a `probe-fix-*` arm's.

### 8.6 D-GH (colour-loss height normalisation) — the first movement

`lr = 1e-3`, u260, l150, probe surface. Both arms carry D-SB; they differ only
in the colour-loss scaling. Loss values are omitted because the rescale makes
them incomparable (§8.5).

| metric | without D-GH | with D-GH |
|---|---|---|
| shape (harness, candidate one) | 0.5604 | **0.6044** |
| pixel | 0.4652 | 0.4669 |
| non-black | 61/91 | 66/91 |
| predicted shape == input shape | 0.6264 | 0.6484 |
| shape correct on ≤9-cell subset | 5/18 | 5/18 |
| **exact under oracle shape** | **0/91** | **1/91** |
| exact pass@1 | 0/91 | 0/91 |

Shape accuracy rises 0.044 and, for the first time in this investigation, **the
model produces colour content that is exactly right on a query**: `d9fac9be` q0,
whose true output is the 1 × 1 grid `[[4]]`, is written as colour 4 in the
model's top-left cell. It is not counted as exact because the shape head emits
`(5, 4)`. Under an oracle shape the model now matches the 11 × 10 lookup table's
1/91 (§8.4) instead of trailing it.

The advisor-flagged risk that D-GH weakens shape supervision on short outputs
did **not** materialise: shape accuracy on the ≤9-cell subset is 5/18 both with
and without it, and aggregate shape improved. The whole-loss-scaling alternative
in §8.5 is therefore not needed on this evidence.

Pixel accuracy is unmoved (0.4652 → 0.4669) and still 0.16 below the lookup
table. **D-GH is a real but small effect. It does not close the §8.4 gap.**

These two arms share an identical draw sequence: D-GH changes no `random.randn`
call, only a loss scale. So unlike the §8.3 comparison, these differences are
attributable to the change. They did not run under identical machine conditions
— 2.9 versus 4.7 minutes at the same configuration — which does not affect the
0.044 shape delta but is worth knowing before reading anything smaller.

### 8.7 Task-local adaptation

`probe-gh-l60-adapt20`: `lr = 1e-3`, u260, **`latent_steps = 60`**, D-SB + D-GH,
20 probe-holdout tasks / 21 queries. Adaptation is leave-one-demonstration-out
on each task's own demonstrations — it never sees a test output
(`latent_workspace_arc_adaptation.py:930-1010`).

| arm | shape | pixel | exact pass@1 | exact pass@2 |
|---|---|---|---|---|
| frozen, same run | 0.7619 | 0.5333 | 0/21 | 0/21 |
| **adapted** | **0.8095** | **0.5608** | **0/21** | **0/21** |

132 folds applied, 71.7 wall seconds, base parameters verifiably restored
afterwards (`base_parameter_sha256 == restored_parameter_sha256`).

**Adaptation helps consistently and does not produce an exact answer.** It lifts
shape by 0.048 and pixel by 0.028 over the frozen baseline *of the same run*,
which is the only valid comparison here.

Two caveats that must not be dropped when quoting these numbers:

1. **This is 21 queries, not 91.** The frozen figures here (0.7619 / 0.5333) are
   far above the 91-query probe figures (0.6044 / 0.4669) purely because these
   20 tasks are an easier subset. Nothing in this table may be compared against
   any other table in this spec.
2. **This is a different model.** `compile_arc_task_local_adaptation_runner`
   rejects any `latent_steps != 60`
   (`latent_workspace_arc_adaptation.py:763`), so an adaptation run gets two
   refinement sweeps where the selected configuration gets five. Adaptation and
   the selected `l150` operating point cannot be combined.

At 71.7 s for 20 tasks, a full 400-task adaptation pass extrapolates to roughly
24 minutes on top of training and evaluation.

### 8.7.1 Update-count arm — is the colour path still travel-bound?

`probe-gh-lr1e-3-u780` — 3× the updates at the same step size, the clean test of
whether travel still binds the colour path (§8.2). It was started once and killed
about a minute in because it would have shared the 12 GB GPU with the full
ARC-AGI-1 evaluation; it was re-run afterwards. Results in §8.10.

### 8.8 Full ARC-AGI-1 evaluation — the headline result

`var/eval-gh-lr1e-3-u260-l150`: the configuration §6 selected — 1024n / 1024e /
b32 / u260 / l150, `lr = 1e-3`, seed 2108, D-SB + D-GH, no adaptation — against
`/datasets/arc/example21-sources.json`, 400 tasks / 419 queries. Compared with
`var/rowcond-1024n-1024e-b32-u260-l150`, the run this investigation started
from.

| metric | rowcond (start) | this run | trivial floor |
|---|---|---|---|
| **exact pass@1** | **0 / 419** | **0 / 419** | — |
| **exact pass@2** | **0 / 419** | **0 / 419** | — |
| shape diagnostic | 0.3604 | **0.5704** | `copy_input` 0.6611, `copy_or_rule_shape` 0.8687 |
| pixel diagnostic | 0.3806 | **0.4138** | `copy_input` 0.6032 |
| non-black grids | 170 / 419 | **348 / 419** | — |
| distinct predicted shapes | 90 | **133** | — |
| predicted colour-0 share | 0.834 | **0.6412** | true marginal 0.531 |
| predicted shape == input shape | 0.535 | 0.6348 | true 0.661 |
| shape correct on ≤9-cell subset | 3 / 26 | **7 / 26** | — |
| exact under oracle shape | 1 / 419 | 1 / 419 | — |
| closest shape-correct query | 3 wrong of 16 | **2 wrong of 4** (`be03b35f` q0) | — |
| `answer_row_head` ΔL2 | 2.059 | 9.550 | — |
| travel budget | 0.96 σ (`starved: true`) | 9.78 σ (`starved: false`) | — |
| runtime | — | 5.3 min | ~4.5 min reference |

**The exact score did not move. It is 0/419, exactly as it was.** That is the
answer to the question this task asked, and no other number in this table
changes it.

What did move is substantial and consistent with §8.2 and §8.6: shape accuracy
rose 0.21 to within 0.09 of the `copy_input` floor it was 0.30 below; the
colour distribution moved two thirds of the way from the collapsed 0.834 to the
true 0.531 marginal; non-black grids went from 41% to 83% of queries; and the
closest query is now 2 wrong cells of 4 rather than 3 of 16. Pixel accuracy rose
only 0.033 and remains 0.19 below `copy_input` and, by the §8.4 measurement,
about 0.16 below a 110-parameter lookup table.

The single oracle-shape-exact query changed identity, from `e872b94a`
(true output `[[0],[0],[0]]`, satisfiable by the all-black collapse) to
`7039b2d7` q0, whose true output is a 3 × 5 grid of colour 1 that the model
writes correctly in its top-left corner but sizes as 16 × 16. The earlier one
was a degenerate artefact of predicting black everywhere; this one is not.

**Comparison caveats.** The two runs are **not seed-matched**: D-SB widens the
answer heads from 1354 to 1414 inputs, changing every `random.randn` draw after
them, and the learning rate differs by 10×. The gross movements above are far
too large to be reseeding noise; nothing small in this table should be
attributed to any single change. Separately, this run overlapped
`probe-gh-lr1e-3-u780` on the same 12 GB GPU for roughly twelve seconds before
that arm was killed (§8.7.1); its 5.3-minute runtime is in line with the
~4.5-minute reference for this operating point, so the overlap does not appear
to have distorted it.

### 8.9 Arms run and arms dropped

Every arm executed against the probe surface, so nothing is silently narrowed.

| arm | outcome |
|---|---|
| `probe-lr{1e-4,1e-3,3e-3,1e-2}-u260` | complete, §8.2 |
| `probe-fix-lr{3e-4,1e-3,3e-3}-u260` | complete, §8.3 |
| `probe-fix-adapt20` | **dropped, 3.8 min wasted.** `compile_arc_task_local_adaptation_runner` rejects any `latent_steps != 60` (`latent_workspace_arc_adaptation.py:763`), and the arm inherited `--latent-steps 150`. Re-run as `probe-gh-l60-adapt20`. |
| `probe-gh-lr1e-3-u260` | tests D-GH at the selected rate |
| `probe-gh-l60-adapt20` | task-local adaptation, 20 tasks, timing + signal |
| `probe-gh-lr1e-3-u780` | 3× the updates at the same step size — separates travel from step size on the colour path, which `lr = 3e-3` conflated |

**Adaptation cannot be combined with the selected configuration.** Task-local
adaptation requires exactly 60 latent steps, so an adaptation run is a different
model from the `l150` arms, with its own frozen baseline. The only valid
comparison for it is adapted versus the `frozen_no_adaptation` block **inside
the same run**, never against an `l150` arm.

Not swept, with reasons: `clip_norm` (§2.5, Adam is invariant to the global
gradient scale that norm clipping changes); `balanced_color_loss` (§2.5, and
D-GH is the better-evidenced loss defect); `neuron_count` and `recurrent_edges`
(§8.4 shows the failure is that an expressible 110-parameter solution is not
found, which more capacity does not address).

## 9. Conclusion, and what to do next

**Superseded in part by §10.** The follow-up runs showed the colour path is
not travel-bound but generalisation-bound: tripling the update budget lowers
the training loss and lowers held-out accuracy. Read §9 as the state of the
investigation at the end of §8, and §10.6 for the corrected conclusion.


**Exact pass@1 on ARC-AGI-1 is 0/419. It did not move.** Three defects were
found and fixed and the model improved substantially on every diagnostic short
of exactness, but it did not produce a single exactly correct answer.

What the three fixes are worth, stated separately:

- **D-PT (travel starvation)** — real and the largest single contributor, on the
  shape path. Isolated on the probe surface at fixed code and seed, raising the
  budget from 0.96 σ to 9.8 σ moved shape accuracy **0.4945 → 0.5824, +0.088**
  (§8.2). Its effect on colour content is unresolved; the arm that would settle
  it was killed once for GPU contention and re-run in §10.4, which found the
  colour path is not travel-bound.
- **D-SB (answer-shape blindness)** — a real expressivity gap, repaired, with
  **no measurable effect** (§8.3). Kept because it is correct and cheap, not
  because it helped.
- **D-GH (height-weighted colour gradient)** — real, small, and the only change
  that produced exactly correct colour content on a query. Isolated with an
  identical draw sequence, shape **0.5604 → 0.6044, +0.044** (§8.6).

**The +0.21 shape gain on evaluation is not attributable to any one of these.**
The isolated probe deltas sum to roughly +0.13; the rest is the probe-versus-
evaluation surface difference and the reseeding that D-SB's head widening
forces. Quote 0.3604 → 0.5704 only as the combined, non-seed-matched
end-to-end figure.

**The finding that should drive the next attempt is §8.4.** An 11 × 10 lookup
table with 110 parameters, which the row head can represent exactly through its
colour block, beats the trained model by 0.13 pixel accuracy on held-out tasks
and scores an oracle-shape exact answer. The model is not short of capacity,
supervision, or (for the shape path) travel. It is failing to find a solution
sitting inside its own hypothesis class.

Ranked next steps at the time this section was written — **all of these were
subsequently run, and both leading hypotheses were wrong. See §10, which
supersedes this list.**

1. Run `probe-gh-lr1e-3-u780` to test whether the colour path is still
   travel-bound. *Done: it is not — more travel overfits (§10.4).*
2. Decompose the row head's weight by input block, expecting the carrier to
   have absorbed the travel. *Done: it had not — the colour block gains share
   and absolute norm faster than the carrier at every budget (§10.3).*
3. If (2) confirmed carrier dominance, rebalance the block scales or
   identity-initialise the colour block. *Superseded: the carrier is not
   starving the copy path, though it does supply the capacity the model
   overfits with, so a carrier-shrinking arm survives for a different reason
   (§10.7).*
4. **Task-local adaptation** helps consistently (§8.7) and is worth a full
   evaluation run once the base model clears the lookup table, but it is a
   multiplier on the base model and the base model is the binding constraint —
   which is what the earlier investigation also concluded. *Unchanged.*

## 10. Follow-up: is the colour path travel-bound, or carrier-bound?

Two questions were left open by §8: whether the colour path is still limited by
parameter travel (§8.7.1), and whether the 1024-dimensional carrier is absorbing
travel that should reach the copy path (§9 step 2). This section answers both.

### 10.1 Pre-registered reading of the block decomposition

Fixed before the trained weights were inspected.

`answer_row_head.weight` is `(1414, 300)`; its input rows partition as carrier
`0:1024`, row one-hot `1024:1054`, query colours `1054:1354`, input height
`1354:1384`, input width `1384:1414`. Every head-input coordinate has unit
expected square — the carrier is scaled to unit RMS, and each one-hot block puts
one coordinate at `sqrt(arity)`, so its expected square is also one. A block's
**squared Frobenius norm is therefore proportional to the logit variance it
contributes**, and shares are comparable across blocks.

At initialisation `W ~ N(0, 1/1414)`, so nominal variance shares are fixed by
block width alone: carrier **72.4%**, query colours **21.2%**, shape one-hots
4.2%, row one-hot 2.1%.

Those nominal shares **understate** the imbalance. The colour block only carries
a one-hot on columns within the input width (`latent_workspace_task.py:1817`
enumerates `grid.cells[row_index]`), and `input_row_valid` zeroes the block
entirely for rows past the input height. For a typical 10-wide ARC query the
colour block supplies about 100 units of input energy rather than 300, putting
the effective split nearer carrier 84% / colour 8%. The tables below report
nominal shares; scale the colour column by roughly `mean input width / 30` for
the effective one.

How this is to be read, decided in advance:

- **Colour-block variance share rises materially above 21.2%** ⇒ training is
  moving toward the copy path, travel is the limiting factor, and `u780` should
  extend the trend.
- **Colour share flat or falling while the carrier block absorbs the travel** ⇒
  the head is solving through a random projection rather than the direct copy
  path, no amount of travel redirects it, and the next fix is architectural.
  The imbalance grows as `n / (n + 390)`, so it would be ~91% carrier at the
  4096-neuron default scale.

The comparison of `u260` against `u780` on this one number is the experiment.

### 10.2 The carrier drives the answer, and drives it the wrong way

This is measurable without any weights, from `eval-gh-lr1e-3-u260-l150`.

The answer heads read `[carrier, row one-hot, query colours of that row]`. At a
fixed row index the last two blocks are **identical in every refinement sweep**:
`build_refinement_feedback_event` sources them from `query_grid` and
`query_shape`, which are only updated by `capture_query_rows` under a gate
requiring `event[:, event_valid_index] > 0.5`
(`latent_workspace_refinement.py:295-301`), and refinement feedback events hold
that channel at zero by construction
(`latent_workspace_refinement.py:339-342`). So **the carrier is the only head
input that changes between sweeps**, and every change in the decoded answer
across the effort axis is attributable to it.

Measured against the sweep-1 answer (effort 30), on all 419 evaluation queries:

| effort | sweeps | queries whose predicted shape changed | cells changed |
|---|---|---|---|
| 30 | 1 | 0 / 419 | 0.0000 |
| 60 | 2 | 66 / 419 | 0.0921 |
| 90 | 3 | 99 / 419 | 0.1402 |
| 120 | 4 | 109 / 419 | 0.1699 |
| 150 | 5 | 121 / 419 | **0.1873** |

**The carrier rewrites 18.7% of the answer's cells and changes 29% of predicted
shapes** between the first sweep and the last. It is not a weak side input.

And it makes the answer worse. Diagnostics along the same axis:

| effort | shape | pixel |
|---|---|---|
| 30 | 0.5656 | **0.4520** |
| 60 | **0.5847** | 0.4457 |
| 90 | 0.5752 | 0.4304 |
| 120 | 0.5752 | 0.4197 |
| 150 (submitted) | 0.5704 | 0.4138 |

Pixel accuracy peaks at the **first** sweep and decays monotonically; shape peaks
at the second. The submission checkpoint is 150. This reproduces, and
strengthens, the interior-peak effect reported independently for the
row-blindness fix (shape and pixel peaking at 60) — post-D-GH the pixel peak has
moved earlier still, to 30.

Two consequences:

1. **The submitted checkpoint is not the model's best checkpoint.** Submitting
   effort 60 instead of 150 would report shape 0.5847 and pixel 0.4457 rather
   than 0.5704 and 0.4138 — a pixel gain of 0.032, comparable to what D-PT and
   D-GH bought together. **It changes exact pass@1 by nothing: exact is 0.0 at
   every effort in every arm measured.** This is a diagnostic-reporting defect,
   not a path to a score.
2. **Extra latent refinement is actively harmful here**, and the only thing that
   varies across it is the carrier. That is independent evidence for carrier
   dominance, arrived at without touching a weight.

### 10.3 The block decomposition — training does load the copy path

`answer_row_head.weight` is `(1414, 300)`; `answer_shape_head.weight` is
`(1414, 60)`. Init shares are the analytic `width / 1414`; init block norms are
`sqrt(width · outputs / 1414)`, the expectation for i.i.d. `N(0, 1/1414)`. For
the colour block that baseline concentrates to about ±0.1% of its share, so the
movements below are far outside draw noise.

**`answer_row_head`**

| block | width | init var% | u260 var% | u780 var% | ‖W₀‖ | ‖W‖ u260 | ‖W‖ u780 |
|---|---|---|---|---|---|---|---|
| carrier | 1024 | 72.42% | 67.34% | **61.47%** | 14.740 | 15.532 (+5.4%) | 16.133 (+9.5%) |
| row one-hot | 30 | 2.12% | 1.83% | 1.82% | 2.523 | 2.561 (+1.5%) | 2.774 (+10.0%) |
| **query colours** | 300 | 21.22% | 26.17% | **30.33%** | 7.978 | 9.682 (**+21.4%**) | 11.332 (**+42.0%**) |
| input height | 30 | 2.12% | 2.39% | 3.23% | 2.523 | 2.923 (+15.9%) | 3.699 (+46.6%) |
| input width | 30 | 2.12% | 2.27% | 3.15% | 2.523 | 2.853 (+13.1%) | 3.652 (+44.8%) |

**`answer_shape_head`**

| block | width | init var% | u260 var% | u780 var% | ‖W₀‖ | ‖W‖ u260 | ‖W‖ u780 |
|---|---|---|---|---|---|---|---|
| carrier | 1024 | 72.42% | 58.83% | **44.12%** | 6.592 | 8.344 (+26.6%) | 9.769 (+48.2%) |
| row one-hot | 30 | 2.12% | 1.35% | 1.95% | 1.128 | 1.263 (+12.0%) | 2.051 (+81.8%) |
| **query colours** | 300 | 21.22% | 29.48% | **33.58%** | 3.568 | 5.906 (**+65.5%**) | 8.523 (**+138.9%**) |
| input height | 30 | 2.12% | 5.39% | **10.25%** | 1.128 | 2.526 (+123.9%) | 4.710 (+317.4%) |
| input width | 30 | 2.12% | 4.95% | **10.10%** | 1.128 | 2.420 (+114.5%) | 4.674 (+314.3%) |

**Training moves toward the copy path, monotonically in the update budget.** The
colour block's share of the row head rises 21.22% → 26.17% → 30.33% while the
carrier's falls 72.42% → 67.34% → 61.47%. This is not an artefact of the carrier
shrinking: the colour block grows *more in absolute norm* (+21.4%, +42.0%) than
the carrier (+5.4%, +9.5%) despite being 3.4× smaller.

The shape head is more striking still: its input height and width blocks — the
ones D-SB added — are the largest movers in the entire table, +318% and +314% at
u780, taking their combined share from 4.2% to 20.4%. **Training reaches for the
query's input shape as hard as it reaches for anything.** That makes the D-SB
negative in §8.3 more interesting, not less: the head was given an input it
demonstrably wants, uses it heavily, and gains no accuracy from it. That is an
open question this spec does not resolve.

Per-block **ΔW** was not measured. The plan was to reconstruct W₀ from the
recorded seed, but the guard comparing the rebuilt tree against the run's
`parameter_sha256_before` refused the match, and neither routing construction
through `_make_model`'s `brainstate.random.seed_context` nor fixing a module
identity clash closed it. **The cause of that divergence is unresolved.** The
analytic init baseline above is used instead; it needs no reconstruction and is
exact in expectation.

### 10.4 The colour path is not travel-bound — it is generalisation-bound

`probe-gh-u260-ckpt` against `probe-gh-u780-ckpt`: identical code, seed,
learning rate and surface, differing only in the update budget (travel 9.78 σ
against 29.33 σ). Loss values are comparable — same D-GH scale, same arms.

| | u260 | u780 |
|---|---|---|
| travel budget | 9.78 σ | 29.33 σ |
| **training loss, last 26 updates** | 1.9436 | **1.8258** (better) |
| colour-block share, row head | 26.17% | **30.33%** (more copy path) |
| **held-out shape** | **0.6044** | **0.5165** (worse) |
| **held-out pixel, effort 150** | **0.4659** | **0.4438** (worse) |
| held-out pixel, best effort (30) | 0.4996 | 0.4833 (worse) |
| exact pass@1 | 0/91 | 0/91 |
| runtime | 4.2 min | 13.0 min |

**Tripling the travel lowers the training loss, loads the copy path further, and
makes held-out accuracy worse on every diagnostic.** That is overfitting, and it
answers §8.7.1: the colour path is **not** travel-bound. More travel is not the
missing ingredient; it is actively harmful past the u260 operating point.

**The pre-registered dichotomy in §10.1 was too simple and its first branch must
not be claimed.** It said a rising colour share implies travel is limiting and
u780 should extend the trend. The colour share did rise, u780 did extend it, and
performance fell. The correct reading is the one neither branch anticipated:

> Training preferentially loads the copy path when given more travel, and the
> resulting predictor still does not approach the lookup table. Travel is a lever
> on the **mechanism** without being the cap on **performance**.

This reframes §8.4. The model is not failing to find the lookup table because it
cannot reach it — given more budget it moves toward it and gets worse anyway. It
has roughly 424,000 parameters in the row head alone plus a 1024-neuron recurrent
substrate, fits 315 training tasks better with every update, and transfers worse.

**What extra budget costs is generalisation; what it does not do is explain the
baseline deficit.** §10.5 measures the u260 model 0.136 below the table on the
very tasks it trained on, so overfitting is what the *u780 arm* demonstrates, not
why the table beats the model in the first place. Two distinct statements, both
supported:

- Beyond u260, more budget buys memorisation and costs transfer (§10.7).
- At u260, the model is already well below a 110-parameter table on seen and
  unseen tasks alike (§10.5), and this spec does not explain why.

### 10.5 How much of the gap does the carrier explain?

Both §10.2 and §10.4 implicate the carrier, so it is worth stating plainly how
much of the §8.4 gap it accounts for. On the probe surface:

| quantity | pixel accuracy |
|---|---|
| 11 × 10 lookup table | 0.6243 |
| best model checkpoint (u260, effort 30) | 0.4996 |
| submitted model checkpoint (u260, effort 150) | 0.4659 |

**Carrier drift across sweeps costs 0.034 of a 0.125 gap — about 27%.** It is a
real defect with a real diagnostic cost, and it is not the whole explanation.

**Nor is generalisation the rest of it.** §10.7 evaluates the u260 model on the
324 test queries of the tasks it trained on and gets pixel 0.4905 at its best
checkpoint, against 0.4996 on held-out tasks — no pixel advantage at all on tasks
it has seen. The lookup table fitted and evaluated on that same training surface
scores 0.6266 query-weighted. So the model is **0.136 below the table even on
tasks whose demonstrations were in its training stream**, essentially the same
deficit it shows on held-out tasks.

That sharpens §8.4 rather than dissolving it. What extra training budget buys is
**memorisation** — the same-task-minus-held-out shape gap widens from +0.072 at
u260 to +0.172 at u780 (§10.7) — but the baseline deficit against the table at
u260 is not a generalisation gap. The predictor is worse than a 110-parameter
table everywhere, on seen and unseen tasks alike.

### 10.6 Answers to the two open questions

1. **Is the colour path still travel-bound after D-GH?** **No.** Tripling the
   update budget at the same step size lowers training loss and lowers held-out
   accuracy (§10.4). Travel was the binding constraint at the shipped 0.96 σ
   operating point and it is no longer binding at 9.78 σ.
2. **Is the carrier absorbing travel that should go to the copy path?** **No —
   the opposite.** The colour block gains share and absolute norm faster than the
   carrier at every budget (§10.3). But the carrier still holds 61–67% of the row
   head's logit variance, still rewrites 18.7% of the answer's cells across
   sweeps, and still drives accuracy down as it does so (§10.2). It is not
   stealing the copy path's travel; it is supplying capacity that the model uses
   to overfit.

Both answers are confirmed in the same metric by §10.7: tripling the budget
makes the model **better on tasks it trained on** (shape 0.6759 → 0.6883) and
**worse on tasks it did not** (shape 0.6044 → 0.5165), which is task-level
memorisation.

A third thing follows that neither question asked about: at the u260 operating
point the model is **0.136 below the lookup table on the tasks it trained on**
(§10.5), so its baseline deficit is not a transfer failure either. That deficit
is the open question this investigation hands forward.

**Neither answer moves exact pass@1 on ARC-AGI-1. It is 0/419 on the full
evaluation and 0/91 on every held-out probe arm.** §10.8 records the one place
an exact answer did appear — 1/324 on ARC *training-split* tasks whose
demonstrations were in the training stream — which is a demonstration that the
machinery can emit an exact answer, not a score.

### 10.7 Confirming the diagnosis — same-task versus held-out

§10.4 inferred overfitting by comparing a training *loss* to a held-out
*accuracy*, which are different quantities. This measures both in the same
metric.

Each trained checkpoint was restored with `--parameter-checkpoint` (the entry
point takes the restore branch when the file exists, so no training re-runs) and
evaluated against a manifest whose `evaluation` role points at the **315 tasks
the model trained on**, with the 84 holdout tasks demoted to the unused `train`
role. 0.7–0.8 minutes per arm.

**Naming.** This is *same-task, unseen-query*, not "train-seen". Training
episodes are leave-one-demonstration-out over those tasks' **demonstrations**;
the tasks' test queries were never in the loss. The genuinely train-seen signal
is the training loss already reported in §10.4.

| model | surface | queries | shape | pixel | exact pass@1 |
|---|---|---|---|---|---|
| u260 | held-out (84 tasks) | 91 | 0.6044 | 0.4659 | 0/91 |
| u260 | same-task (315 tasks) | 324 | 0.6759 | 0.4587 | 0/324 |
| u780 | held-out (84 tasks) | 91 | **0.5165** | **0.4438** | 0/91 |
| u780 | same-task (315 tasks) | 324 | **0.6883** | **0.4778** | **1/324** |

Read down each surface — that is the like-for-like comparison, since the two
surfaces differ in size and difficulty and cannot be subtracted from each other
at a fixed budget:

- **On tasks the model trained on, tripling the budget makes it better**: shape
  0.6759 → 0.6883, pixel 0.4587 → 0.4778.
- **On tasks it did not, the same change makes it worse**: shape 0.6044 →
  0.5165, pixel 0.4659 → 0.4438.

This is the first branch of the reading pre-registered before the run: better on
same-task while worse on held-out means **task-level memorisation**, and
"overfitting" is the right word for §10.4. The competing explanation — that
D-GH's colour reweighting skews the shape:colour balance and more updates push
harder on a skewed objective — predicts short-output shape collapsing fastest,
since those are the examples whose colour term D-GH multiplies by up to 30. The
opposite happened: shape correct on the ≤9-cell subset went **up** at u780, 5/18
→ 8/18, while aggregate shape fell. D-GH imbalance is not what degraded u780.

### 10.8 The machinery can produce an exact answer

`trainseen-u780` scores **1/324 exact pass@1**, the first nonzero exact score
recorded anywhere in this investigation. The exactly-correct queries:

| effort | task | true output | model output |
|---|---|---|---|
| 0 | `b9b7f026` q0 | `[[7]]` | `[[7]]`, shape (1,1) |
| 30 | `27a28665` q1 | `[[1]]` | `[[1]]`, shape (1,1) |
| 60–150 | `27a28665` q2 | `[[2]]` | `[[2]]`, shape (1,1) |

`27a28665` q2 takes a 3 × 3 input `[[2,0,2],[0,2,0],[2,0,2]]` and must emit the
single cell `[[2]]`. The model predicts shape (1,1) and colour 2, and holds that
answer across every sweep from effort 60 to 150. These are real end-to-end
answers: shape head and colour head both correct, decoded through the normal
path, scored by the harness's own exact criterion.

**This is not a score and must not be quoted as one.**

1. It is on **ARC training-split tasks whose demonstrations were in the training
   stream**. The model had seen those tasks, just not these queries.
2. It is precisely the memorisation §10.8 measures: u260 scores 0/324 on the same
   surface. The extra budget bought an exact answer *only where the model has
   seen the task*, while costing accuracy everywhere else.
3. **Exact pass@1 on the ARC-AGI-1 evaluation split is 0/419 and did not move.**

What it does establish is narrower and still worth having: the decoder, the shape
head, the colour head and the scoring path can jointly produce an exactly correct
ARC answer. Before this, no configuration had ever emitted one, and it was open
whether some structural defect made exactness unreachable end to end. It is not.
The barrier is generalisation.

### 10.9 Follow-up arms run

No code changed in this phase; every arm ran on the commit that produced §8's
full evaluation, so all four are directly comparable to it and to each other.
**Nothing was dropped or truncated.**

| arm | purpose | runtime | outcome |
|---|---|---|---|
| `probe-gh-u260-ckpt` | reference budget, weights dumped | 4.2 min | complete; also a determinism check |
| `probe-gh-u780-ckpt` | 3× travel at the same step size | 13.0 min | complete (§10.4) |
| `trainseen-u260` | same-task surface, checkpoint restored | 0.7 min | complete (§10.7) |
| `trainseen-u780` | same-task surface, checkpoint restored | 0.8 min | complete (§10.7, §10.8) |

**Determinism.** `probe-gh-u260-ckpt` repeats `probe-gh-lr1e-3-u260` with only a
checkpoint write added. Shape reproduces exactly (0.6044); pixel differs by
0.0010 (0.4659 against 0.4669) and `answer_row_head` ΔL2 by 0.001. **Treat ~0.001
in pixel as the arm-to-arm noise floor** — every difference this spec attributes
to a change is at least an order of magnitude larger.

**The `trainseen-*` arms performed no training**: the entry point takes its
restore branch when `--parameter-checkpoint` names an existing file, so their
0.7-minute runtimes are evaluation only. Any `learning_rate × updates` figure
printed for them describes the checkpoint they restored, not travel they spent.

### 10.10 Revised next steps

Replaces §9's ranked list, which assumed travel and carrier-starvation.

1. **Regularise or shrink the carrier path**, since §10.4 and §10.7 identify
   generalisation as the cap and §10.2 and §10.3 identify the carrier as the
   capacity supplying it. The cheapest decisive test is an arm that scales
   the carrier block of the head input down by a constant, or drops it entirely,
   leaving `[row one-hot, query colours, input shape]` — a ~390-input head that
   cannot memorise a task. If a carrier-free head beats the full one on held-out
   pixel, the carrier is a net liability at this data scale and the architecture
   should change. This is one short probe arm.
2. **Report the best checkpoint, not effort 150** (§10.2). Worth 0.03 pixel and
   0.014 shape, costs nothing, and changes exact by nothing. Selecting the
   checkpoint on the probe surface is legitimate; selecting it on evaluation is
   not.
3. **More training tasks or augmentation** addresses what §10.7 measures — the
   model getting better on tasks it trained on and worse on tasks it did not, at
   the same time. 315 tasks is very few for a 424,000-parameter head plus a
   1024-neuron recurrent substrate. Note this treats the *u780 degradation*, not
   the baseline deficit: §10.5 shows the model 0.136 below the table on trained
   tasks, which more tasks would not obviously fix.
5. **Explain the baseline deficit**, which is now the central open question and
   which this spec does not answer. The model trails a 110-parameter lookup table
   on seen and unseen tasks alike at an operating point where it is neither
   travel-starved nor overfitting. Worth isolating before spending on anything
   above it.
4. **Do not spend more on update count or learning rate.** §8.2 and §10.4
   bracket the useful range: below `lr·U ≈ 0.26` the head cannot travel, above it
   the model overfits, and the window between them is narrow.
