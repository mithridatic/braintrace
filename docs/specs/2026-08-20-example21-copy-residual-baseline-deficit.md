# Example 21 — the baseline deficit is an unlearned identity map; fix by copy residual

Status: measured; §6 results confirm the mechanism in two stages
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

## 6. Results — the deficit has two layers, both now measured

### 6.1 Residual alone: real, consistent, and walked back by training

Seed-matched, full ARC-AGI-1 evaluation, effort 60, gain 2.0 vs the sweep's
gain-0 artifacts:

| seed | pixel base → residual | non-black copy base → residual | rule | shape |
|---|---|---|---|---|
| 2108 | 0.3835 → 0.4199 | 0.2054 → 0.3327 | 0.104 → 0.080 | 0.549 → 0.535 |
| 31337 | 0.4122 → 0.4507 | 0.2987 → 0.4589 | 0.121 → 0.097 | 0.585 → 0.566 |
| 7777 | 0.3852 → 0.4157 | 0.3058 → 0.4488 | 0.119 → 0.084 | 0.570 → 0.563 |

Every seed improves pixel by ~0.035 — but the §5 bar (non-black copy ≥ 0.8)
was missed, which by the pre-registered falsification clause demanded a cause
before a bigger gain. Two controls found it:

- **Plumbing is intact.** An untrained 64-neuron model decodes 86% in-range
  copy at gain 2.0 and 100% at gain 1000 (CPU probe), and an untrained
  4096-neuron `--structural-only` run at gain 2.0 decodes **0.661** of
  non-black copy cells within its predicted-grid overlap on the real
  evaluation split.
- **Training walks copy back.** The same overlap metric reads 0.661
  untrained-with-residual → **0.493** trained-with-residual → 0.303 trained
  baseline. Training *destroys* copy competence the residual provides for
  free, in favour of carrier-conditioned fits that do not transfer — the same
  mechanism the travel spec measured as same-task memorisation at u780.

### 6.2 Carrier starvation: the second layer, confirmed

`--row-head-carrier-scale 0` (row head only; shape head keeps its carrier)
plus gain 2.0, seed 2108, full evaluation — the arm
`2026-08-20-example21-parameter-travel-starvation.md` §10.10 ranked first:

| metric | baseline s2108 | residual s2108 | carrier-free + residual s2108 |
|---|---|---|---|
| pixel (effort 60) | 0.3835 | 0.4199 | **0.5454** |
| pixel under oracle shape | 0.4855 | 0.5253 | **0.6031** |
| copy cells (oracle shape) | 0.6375 | 0.7039 | **0.9178** |
| non-black copy | 0.2054 | 0.3327 | **0.7944** |
| rule cells | 0.1043 | 0.0801 | 0.0248 |
| shape | 0.5489 | 0.5346 | 0.4893 |
| pixel by effort 0/30/60 | falls after 30 | falls after 30 | **rises: 0.232 / 0.529 / 0.545** |
| **exact pass@1** | 0/419 | 0/419 | **1/419** |

The +0.16 pixel step is ten times the ±0.016 seed spread of the sweep;
single-seed suffices for the mechanism claim (a seed-replication is still
worth running before promoting the configuration). Under an oracle shape the
carrier-free model reaches 0.6031 against the `copy_input` floor 0.6032 —
the copy path is saturated — and the remaining oracle-shape gap to the table
(0.6375) is exactly the rule cells the carrier-free head can no longer
express plus a small out-of-range shortfall. Pixel now *rises* with latent
effort instead of decaying: the §10.2 carrier-drift pathology on the row path
is gone by construction.

The exact answer is `bbb1b8b6` q1 (predict the input's left 4×4 panel):
shape head chose (4, 4) and the copy path filled it cell-perfect. It is the
**first exact answer on the ARC-AGI-1 evaluation split in this project** —
every earlier nonzero exact was on training-split tasks the model had seen.

### 6.3 Corrected root-cause statement

The baseline deficit had two stacked causes:

1. **D-IC** — the identity map must be learned from scratch and is an order
   of magnitude of optimisation away at any stable operating point (§1.3).
   The copy residual supplies it architecturally: +0.035 pixel, all seeds.
2. **D-CD (carrier displacement)** — given the carrier, cross-entropy
   training actively *unlearns* the copy prior, because carrier-conditioned
   task fits lower training loss and do not transfer (§6.1). Starving the
   row head of the carrier removes the displacement: +0.13 further pixel,
   copy saturated to its floor, and the first evaluation-split exact answer.

## 7. What this hands forward

1. **The carrier-free row head is a diagnosis, not the destination.** It
   cannot express rules (0.0248 rule cells) and rules are 19.3% of cells and
   a requirement for almost every exact answer. The needed shape is
   *selective* carrier access — the Attention Residuals argument applied
   here: softmax-gated aggregation instead of always-on additive access, so
   the copy path is preserved by construction and the carrier contributes
   only where demonstrations support a deviation. Design candidates: a
   learned scalar gate on the carrier block, a per-column gate driven by the
   colour block, or depth-wise attention over per-sweep carriers.
2. **Shape is now the binding constraint** for exactness (0.4893, and the
   solved query shows the whole pipeline works when shape is right). The
   shape head kept its carrier; whether it suffers the same displacement is
   unmeasured — the same scale experiment applies to it.
3. Replicate the carrier-free arm on seeds 31337/7777 before promoting any
   configuration, and re-run task-local adaptation on top of the carrier-free
   base: adaptation was consistently positive and is a multiplier on a base
   model that is no longer below its own copy floor.

## 8. Follow-up round (2026-08-20, same branch)

Executing §7 items 2 and 3 (J: "Execute the next steps on this branch").

### 8.1 Shape-head carrier scale

`ModelConfig.shape_head_carrier_scale` (default 1.0) mirrors
`row_head_carrier_scale` on the answer shape head's input: the carrier block
of `_refinement_head_input` is multiplied by the scale, the event-derived
blocks (row one-hot, colour block, dimension one-hots) are untouched, and
scale 1.0 returns the shared head input object bit-exactly. Exposed as
`--shape-head-carrier-scale`. Tests mirror the row-head pair: a negative
value is a config error, and at scale 0 two models that differ only in
`recurrent_gain` must produce bit-identical final shape logits.

Interpretation guard: unlike the row head, the shape head *needs* the
carrier wherever `output_shape != input_shape` (33.9% of evaluation
queries) — the dimension one-hots only make the identity shape expressible.
So scale 0 measures the displacement/expressiveness trade on shape exactly
as §6.2 did on rows: if shape accuracy *rises* at scale 0, displacement
dominates; if it falls, the carrier is earning its keep on shape.

### 8.2 Pre-registered runs

All at 4096n/4096e/b32/u260/l60/lr1e-3, gain 2.0:

| run | row scale | shape scale | seed | question |
|---|---|---|---|---|
| cr2cs0 s31337 | 0 | 1 | 31337 | replicate §6.2 |
| cr2cs0 s7777  | 0 | 1 | 7777  | replicate §6.2 |
| cr2cs0ss0 s2108 | 0 | 0 | 2108 | shape-head displacement |

Expectations: replication holds if pixel ≈ 0.54 ± seed noise (baseline seed
spread was ±0.016) on both seeds. Shape arm: shape accuracy vs 0.4893
decides the §8.1 question; pixel may move either way.

### 8.3 Results (measured 2026-08-20, revision fd3b91f)

Harness-replica query-weighted pixel at effort 60, decomposed as §1:

| arm | seed | pixel | oracle-shape pixel | shape | copy@oracle | rule@oracle | exact |
|---|---|---|---|---|---|---|---|
| baseline (§5)         | mean/3 | 0.3936 | — | — | — | — | 0 |
| residual only (cr2)   | 31337 | 0.4507 | — | 0.578 | — | — | 0 |
| residual only (cr2)   | 7777  | 0.4157 | — | 0.563 | — | — | 0 |
| carrier-free (cr2cs0) | 2108  | 0.5454 | 0.6031 | 0.4893 | 0.9178 | 0.0248 | **1** |
| carrier-free (cr2cs0) | 31337 | 0.4644 | 0.5857 | 0.4749 | 0.8603 | 0.0394 | **1** |
| carrier-free (cr2cs0) | 7777  | 0.4758 | 0.5916 | 0.4678 | 0.8301 | 0.0411 | **1** |
| + shape scale 0 (ss0) | 2108  | 0.3781 | 0.5814 | 0.2983 | 0.8508 | 0.0353 | **1** |

**Replication: confirmed.** Carrier-free beats its seed-matched
residual-only arm on all three seeds (+0.10, +0.014, +0.060) and its
baseline by +0.07 to +0.16; mean 0.4952 vs baseline 0.3936. Copy@oracle is
0.83–0.92 everywhere (baseline was 0.57–0.69). Most decisively, **all three
seeds produce the same exact answer** — `bbb1b8b6` q1 at both effort 30 and
60 — so the evaluation-split exact solve is mechanism-driven, not a seed
lottery. Magnitude varies with the shape head (s2108's 0.4893 shape is why
its pixel leads); copy is not fully saturated on the new seeds, so the
carrier still leaks into the row path indirectly through which rows the
sweep visits, or the head initialisation matters — a residual open edge.

**Shape-head displacement: refuted.** Starving the shape head of the
carrier collapses shape accuracy 0.4893 → 0.2983 and pixel 0.5454 → 0.3781
on the matched seed, while the (unchanged) row path holds copy@oracle at
0.85. Unlike the row head, the shape head's carrier is earning its keep:
`output_shape != input_shape` queries genuinely need task information the
event blocks cannot carry. The §8.1 interpretation guard resolves to
"expressiveness dominates" — shape improvements must come from a better
carrier or shape-specific supervision, not from carrier removal.

### 8.4 Standing recommendation

Ship configuration for further work on this branch:
`--copy-residual-gain 2.0 --row-head-carrier-scale 0.0` with the shape head
untouched. Remaining §7 items: the gated-carrier row head (rules are the
next 0.19 of headroom) and task-local adaptation on the carrier-free base.

## 9. Design proposal — gated carrier access for the row head (NOT YET APPROVED)

Rules are the remaining 0.19 of headroom, and the carrier-free row head
forfeits them by construction. The design goal, per Attention Residuals: the
copy path preserved architecturally, carrier access *earned*, not default.

Proposed minimal mechanism (candidate A — split head with a zero-initialised
carrier branch):

1. Split the row head into `answer_row_event_head` (event blocks only) and
   `answer_row_carrier_head` (carrier block only); both bias-free Linear so
   ETP tracking is unchanged.  At the split, event-head weights are
   initialised from the trained carrier-free head's event columns.
2. The carrier branch enters as
   `next_row = event_head(events) + residual + tanh(g) * carrier_head(carrier)`
   with `g` a learned 300-coordinate logit gate initialised at 0 — the model *starts* at the
   §8 carrier-free optimum and training must buy carrier access through `g`.
3. Falsification: if training simply reopens the gate and walks copy back
   down (§6.1 displacement returning through `tanh(g)`), the coordinatewise gate is
   insufficient and a per-column gate driven by the colour block (candidate
   B) or depth-wise softmax over per-sweep carriers (candidate C, the full
   AttnRes form) is next.

Open risks to resolve before implementation: whether a learned logit-vector gate
multiplying a tracked Linear output keeps both heads `all_direct` in the
ETP compiler, and whether the gate needs a slower learning rate than the
heads to prevent early displacement. Awaiting J's approval per the working
agreement before any code.

### 9.1 Approved (J, 2026-08-20 18:26 PDT) — implementation resolution

The flagged ETP risk is real: `hid_param_op.py` registers trainable
parameters only as trainable invars of ETP primitives, so a bare scalar
`ParamState` multiplied into the graph would silently never train.
Resolution, staying inside the approved semantics:

- The gate is a zero-initialised bias-free `braintrace.nn.Linear(1, 300)` fed
  with ones, so `gate = tanh(w)` is an ETP-tracked trainable vector with one
  coordinate per row logit. Its
  path to the `answer_row` hidden state (tanh → mul → add) traverses no
  other ETP primitive, so it classifies `all_direct`; the carrier head's
  output enters the mul as a sibling input, not a chain link.
- The carrier and event heads are literal slices of one same-seed combined
  matrix scaled by the complete input width. At `gate = 0` the carrier
  contribution is exactly zero (the §8 carrier-free start), while the
  gate's own gradient — proportional to the carrier head's output — is
  nonzero, avoiding the frozen-at-zero saddle of a doubly-zero start.
- Config: `row_head_carrier_gate: bool = False`; combining the gate with
  `row_head_carrier_scale != 1.0` is a config error (the gate replaces the
  scale mechanism). Flag: `--row-head-carrier-gate`.
- Both heads train from scratch (no checkpoint grafting — §9 item 1's
  "initialise from the trained head" is dropped; these arms always train
  from a fresh initialisation, so there is no trained head to graft).

Tests: gate-on model at init is bit-identical across recurrent gains
(carrier-free start); compile keeps event head, carrier head, and gate in
`param_states` as `all_direct`; gate + non-default scale rejected; forcing
the gate weight open makes the row answer carrier-dependent.

### 8.5 Task-local adaptation on the carrier-free base (cr2cs0ad, s2108)

Wall clock: 2h19m (01:22–03:41 UTC) — the per-task inner loop does not
amortise like the 3.5-minute frozen runs. **Standing constraint from J: no
more hour-plus runs; the adaptation arm is answered and will not be
repeated.**

Adapted vs frozen (same artifact, effort 60): pixel 0.5412 → **0.5861**,
shape 0.4916 → **0.7112**, exact 1 → **2** pass@1 (`bbb1b8b6` q1 plus
`e872b94a` q0, a 3×1 output — shape- and rule-driven, not copy). Adaptation
attacks exactly the constraint §6.2 left binding: it nearly halves shape
error while leaving the saturated copy path alone (copy@oracle 0.9046).
Cumulative over the day: baseline 0.3835 → carrier-free 0.5454 →
adapted 0.5861, with the reported adapted pixel within 0.017 of the
0.6032 copy-floor reference and shape no longer the dominant deficit.

### 9.2 Implementation amendment — per-logit gate vector

The eligibility-trace VJP (`io_dim_vjp.py`) rejects an ETP head whose
output width differs from the hidden group it feeds (1 vs 300), so the
scalar gate is illegal as designed. Implemented instead as the minimal
legal member of the approved family: `Linear(1, 300)` zero-initialised,
`tanh` applied elementwise — a **per-logit gate vector** (the candidate-B
direction §9 names as first fallback). Same carrier-free start, same
earned-access semantics, 300 gate parameters instead of 1. Verified:
gate-on model at init is bit-identical across recurrent gains; forcing the
gate weights to 3.0 makes the row answer carrier-bound; event head,
carrier head, and gate all compile `all_direct`.

### 9.3 Results — gated arm (cr2g, s2108, 3m20s)

| metric | carrier-free (cr2cs0) | gated (cr2g) | full carrier (b64 ref) |
|---|---|---|---|
| pixel | 0.5454 | 0.5443 | 0.4613 |
| shape | 0.4893 | 0.5489 | — |
| copy@oracle | 0.9178 | 0.8893 | 0.6879 |
| rule@oracle | 0.0248 | 0.0484 | 0.1250 |
| exact pass@1 | 1 (`bbb1b8b6` q1) | 1 (`bbb1b8b6` q1) | 0 |

The gate holds the carrier-free pixel level (−0.001, within noise) while
**doubling rule-cell accuracy** (0.0248 → 0.0484) and improving shape
(+0.06); copy gives back only 0.03 of its 0.23 gain. Displacement did not
return: training opened the gate part-way and bought rules with it, not
copy unlearning. The §9 falsification clause does not trigger. Rule
accuracy remains well below the full-carrier 0.125, so the gate is
conservative at this budget — more updates or a per-column gate driven by
the colour block (candidate B proper) are the natural next dials. The
gated head is the recommended base configuration going forward:
`--copy-residual-gain 2.0 --row-head-carrier-gate`.
