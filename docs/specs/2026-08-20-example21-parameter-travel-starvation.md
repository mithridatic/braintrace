# Example 21 — parameter-travel starvation and answer-shape blindness

Status: draft (diagnosis complete, implementation in progress)
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

**No weight in either answer head can move even one initialisation sigma.** The
trained head is a small perturbation of its random initialisation, so the
decoder's output is dominated by the random prior plus whatever dataset-constant
tilt survives averaging — which is exactly the colour marginal reported in
§2.2 of the row-blindness spec.

Measured Frobenius deltas corroborate the bound independently
(`training.parameter_changes` in the same artifact). Both heads are bias-free
with `w ~ N(0, 1/head_width)`, so the initialisation Frobenius norm of a
`head_width × out` matrix is exactly `√out`:

| head | shape | ‖W₀‖_F | ‖ΔW‖_F | relative travel |
|---|---|---|---|---|
| `answer_row_head.weight` | 1354 × 300 | √300 = 17.32 | 2.0588 | **0.119** |
| `answer_shape_head.weight` | 1354 × 60 | √60 = 7.746 | 1.8790 | **0.243** |

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
trivial table beats it by 0.31 (§8.4).

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
| **11 × 10 conditional table** | **0.7734** | **1.0029** | **1/91** |
| **trained model, best arm** | **0.4652** | — | **0/91** |

The fitted table is the identity on all ten colours and maps out-of-range to
colour 0. It is a lookup table with 110 free parameters and it uses no neurons,
no recurrence and no demonstrations.

**The exact column is under an oracle shape for every row, including the
model's**, because the table predicts no shape at all; the pixel column follows
the harness's own `valid_cell` convention (matches over the true grid's cells).
Compared like for like, the table scores one exact answer the model does not:
`d10ecb37` q0, whose 2 × 2 output is the top-left corner of a 4 × 8 input, so
the identity map reproduces it cell for cell. That is not a shape result — it is
a colour-content result, and the model's colour content does not reach it on any
query.

The model is 0.31 below a solution its own architecture can express. That is the
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

### 8.6 Post-fix runs

Filled in below as they complete.
