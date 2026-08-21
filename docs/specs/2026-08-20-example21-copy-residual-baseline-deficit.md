# Example 21 — the baseline deficit is an unlearned identity map; fix by copy residual

Status: draft; hypothesis test pending GPU run
Date: 2026-08-20
Branch: `worktree-ex21-copy-residual`

Answers the central open question handed forward by
`2026-08-20-example21-parameter-travel-starvation.md` §10.10: why the trained
model trails a 110-parameter lookup table by 0.125–0.136 query-weighted pixel
accuracy on seen and unseen tasks alike, at an operating point that is neither
travel-starved nor overfitting. The 39-run six-factor sweep (13 configs × 3
seeds, 2026-08-20) found no hyperparameter that closes any of it: every effect
clearing 2σ was negative, so the deficit is structural, not an operating-point
problem.

References that motivated the fix shape:

- Engdahl et al., *BDH-CQ: In-Context Learning with Recurrent Latent
  Reasoning*, arXiv:2608.09888 — the contract Example 21 instantiates; the fix
  must stay inside "no weight update at inference".
- Kimi Team, *Attention Residuals*, arXiv:2603.15031 — the identity-mapping
  argument: a residual path preserves a direct route for information so
  training only has to learn *modulations* of it, not the route itself; and
  the observation that uniform accumulation across depth dilutes early
  contributions, which is the sweep-axis pathology §10.2 of the travel spec
  measured (pixel peaks at the first refinement sweep and decays monotonically).

## 1. Phase-1 evidence — where the 0.13 lives

All figures computed from sweep artifacts already on disk
(`var/example21-mv-*/result.json`, per-query candidate grids) joined against
the raw ARC-AGI-1 evaluation split, using the harness's own overlap metric
(matching cells in the candidate-1/truth overlap ÷ truth size,
query-weighted). Replication of the harness was verified exact on every run
inspected (e.g. b64/s31337 reported 0.4613, replicated 0.4613).

### 1.1 Cell-class decomposition under a shared oracle shape

Cells of the true output partition into three classes against the query input:
`copy` (in-range, `truth == input`), `rule` (in-range, `truth != input`),
`oor` (outside the input's bounds; the table emits black there). Model grids
are cropped/zero-padded to the true shape so model and table wear the same
shape policy — this removes the confound that the §8.4 table was scored with a
shape the model has to predict.

b64/s31337 (the sweep's best run), effort 60, 419 queries:

| class | share of cells | model (oracle shape) | table (oracle shape) |
|---|---|---|---|
| copy | 0.757 | **0.6879** | 1.0000 (by construction) |
| rule | 0.193 | 0.1250 | 0.0000 (by construction) |
| oor | 0.050 | 0.5435 | 0.5658 |

The three baseline seeds put model copy-cell accuracy at 0.57–0.65. **The
entire deficit is on copy cells.** The model wins on rule cells (+0.024
weighted) and ties on out-of-range cells; it loses ~0.24 weighted on cells
whose answer is delivered into the answer head as a one-hot.

### 1.2 The copy failure is a prior, not noise

Copy-cell accuracy by colour, b64/s31337 effort 60:

| colour | accuracy | n |
|---|---|---|
| 0 (black) | 0.8798 | 47,452 |
| 1 | 0.5717 | 6,885 |
| 4 | 0.4514 | 2,951 |
| 5 | 0.3675 | 3,116 |
| 3 | 0.2960 | 3,405 |
| 8 | 0.2307 | 4,347 |
| 2 | 0.1959 | 2,756 |
| 6 | 0.1887 | 1,314 |
| 7 | 0.1721 | 1,586 |
| 9 | 0.0859 | 757 |

Accuracy ranks almost exactly by dataset colour frequency. That is the
signature of a head whose output is dominated by learned colour-frequency
priors, with a copy signal too weak to override them — not of decode-time
noise, which would not produce a frequency ordering. Non-black copy accuracy
is 0.30–0.35 aggregate; at effort 0 it is 0.03 (no copy behaviour at all
before the first refinement sweep).

### 1.3 The copy signal grows with optimisation steps and is nowhere near done

Non-black copy accuracy across sweep arms (mean ± stdev over 3 seeds,
4096n/4096e/l60 unless noted):

| arm | non-black copy | black copy |
|---|---|---|
| u260 lr1e-3 | 0.2700 ± 0.0457 | 0.8174 |
| u390 lr1e-3 | 0.2967 ± 0.0196 | 0.7999 |
| u520 lr1e-3 | **0.3825 ± 0.0259** | 0.7984 |
| u260 lr2e-3 | 0.2358 ± 0.0215 | 0.8462 |
| u260 lr3e-3 | 0.2432 ± 0.0518 | 0.8214 |
| l90 lr1e-3 | 0.2898 ± 0.0155 | 0.7995 |
| l120 lr1e-3 | 0.2558 ± 0.0401 | 0.8877 |

Three facts, all from existing artifacts:

1. **More update steps monotonically grow copy competence** (+0.11 from u260
   to u520, ~4σ), even though aggregate pixel moved only +0.016 — the copy
   path is still climbing at every budget the sweep bought.
2. **Bigger steps do not substitute**: lr 2e-3/3e-3 reach more raw travel and
   make copy *worse*. What the copy path needs is optimisation steps at a
   stable step size, not displacement.
3. Latent steps do not help it.

## 2. Root-cause statement

**D-IC (identity-map cost).** The answer row head must *learn* the identity
colour map — the 110-parameter table of §8.4 — from a random initialisation,
through cross-entropy whose targets are ~53–63% black, against a carrier
carrying easier descent directions (task-marginal priors). The identity map is
expressible but expensive: at every stable operating point the sweep explored,
the optimiser buys roughly +0.06 non-black copy accuracy per +130 updates
starting from ~0.27, so reaching the table's 1.0 costs an order of magnitude
more optimisation than the stability window allows (larger steps collapse
shape; more steps at 1024n-scale demonstrably overfit via the carrier). The
model spends its entire training budget part-way through learning what the
lookup-table baseline has by construction, and the residue is the baseline
deficit.

This subsumes the observations the travel spec left unexplained: it is why the
deficit is identical on seen and unseen tasks (the unlearned map is
task-independent), why no sweep factor moved it (none change what must be
learned), and why the colour block gains share monotonically without the model
approaching the table (it is climbing a hill it never finishes).

## 3. Fix — a fixed identity residual on the row-head logits (D-IC)

Following the identity-mapping argument of Attention Residuals: preserve a
direct path so training learns deviations from copy instead of copy itself.

In `cell_step`, after the row head fires:

```python
next_row = self.answer_row_head(head_input)
next_row = next_row + gain * event[:, layout.input_color_slice]
```

The colour block is indexed `column * 10 + colour`
(`latent_workspace_task.py:1818`) and the row logits reshape `(30, 10)` on the
same index (`latent_workspace_refinement.py:696`), so the addition is
index-aligned by construction: it raises the logit of exactly the query's
colour at each occupied column by `gain`. Properties:

- **Out-of-range cells are untouched**: the block is written gated by
  `input_row_valid` and per-column presence, so beyond the input's height or
  width the residual contributes zero and the learned head keeps sole custody
  — the class where the model already matches the table (§1.1).
- **Rule cells stay learnable**: the head can counteract the residual with
  negative weights exactly where demonstrations show `out != in`; the model
  already beats the table on rule cells with no residual to fight.
- **No new trainable parameter, no ETP compiler surface**: the residual is a
  constant-scaled slice of the event the head already receives; the tracked
  `answer_row_head.weight` op is untouched, so its `all_direct` classification
  must not change (regression-guarded).
- **In-contract for BDH-CQ**: pure architecture; no inference-time update.
- `gain = 0.0` is bit-exact backward compatibility.

The gain ships as `--copy-residual-gain` with default 2.0: softmax over 10
colours where the correct logit starts +2.0 above rivals whose spread is the
~0.85-σ carrier contribution yields a strong but overridable copy prior
(initial p(correct) ≈ 0.75–0.9 on occupied columns), i.e. the model starts
near the table instead of an order of magnitude of updates away from it. The
shape head is untouched.

## 4. Tests (co-located, suffix style, red first)

`latent_workspace_model_test.py`:

1. `test_copy_residual_raises_the_query_colour_logit_by_the_gain` — for a
   synthetic event with known colours, the post-residual logit at
   `column*10+colour` exceeds the bare head output by exactly `gain`, and
   non-occupied coordinates are unchanged.
2. `test_copy_residual_gain_zero_reproduces_the_bare_head` — bit-exact no-op
   at 0.0.
3. `test_refinement_heads_stay_compiled_all_direct_with_copy_residual` —
   compile regression: both answer heads remain tracked `all_direct`.

## 5. Pre-registered experiment

Seed-matched against the sweep baseline: 4096n/4096e/b32/u260/l60/lr1e-3,
seeds 2108/31337/7777, gain 2.0 vs the existing gain-0 sweep artifacts,
ARC-AGI-1 evaluation split, Docker GPU, scored by the same scripts that
produced §1 (`gap_decomposition.py`, `copy_error_anatomy.py`).

Expected if D-IC is the root cause, stated in advance:

- non-black copy-cell accuracy ≥ 0.8 (from ~0.30);
- query-weighted pixel above the `copy_input` floor 0.6032 and at or above the
  table's 0.6375 oracle-shape / 0.6243-style figures, less whatever shape
  errors cost outside the oracle;
- rule-cell accuracy not below the baseline's ~0.12 (the head must still be
  able to override the residual);
- shape accuracy unchanged within seed noise (the shape head is untouched).

Exact pass@1 is reported but not predicted: §2.1 of the travel spec shows
exactness additionally requires shape correctness and every rule cell, so the
residual is necessary-not-sufficient for a score. Falsified if non-black copy
stays near 0.3 or pixel fails to clear 0.55 at gain 2.0 — that would mean the
decode path discards the colour block for a reason not yet found, and the next
step would be weight-level inspection of a dumped checkpoint, not a bigger
gain.
