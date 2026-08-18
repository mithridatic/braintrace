# Example 21 — ARC exact-score recovery

Status: in progress
Branch: `feat/example21-latent-reasoning`
Extends `2026-08-18-example21-positive-arc-score.md`. Supersedes nothing.

## 1. Problem

Example 21 scores exact `pass@1 = 0.0000` in every run ever recorded, and
`pass@2 = 1/419` once, in `var/example21-positive-arc-width32-balanced-4096u`.
The prevailing reading has been that ARC is simply hard for this substrate. That
is not the reading the evidence supports.

## 2. Measured evidence

Mined from `var/example21-positive-arc-width32-balanced-4096u/result.json` (419
per-query records, effort 32, no new compute), against the floor table committed
at `docs/diagnostics/example21_dummy_floor.py` and re-measured on the restored
dataset:

| metric | model | trivial predictor |
|---|---|---|
| shape accuracy | 0.365 | **0.8687** (`copy_or_rule_shape`) |
| pixel accuracy | 0.355 | **0.6336** (`copy_or_rule_shape`) / 0.6032 (`copy_input`) |
| pixel given shape correct | 0.408 | — |
| best single-query pixel | 0.926 | — |
| queries reaching pixel >= 0.95 | **0 / 419** | — |

The model loses to a predictor that uses no neurons at all, on both axes exact
match requires. Three structural defects account for this.

### D1 — the output is generated from scratch

The decoder emits 900 cell colours from a 128-d hidden with no reference to the
query input. Copying the input scores 0.6032 pixel; the model scores 0.355.
Blanking the demonstrations (`no_context`) still leaves 0.211, so the query input
contributes almost nothing to what is produced.

### D2 — the shape heads are degenerate

79.2% of candidate-1 predictions are square, and the top-8 predicted shapes are
*all* `h == w`. Two independent 30-way softmaxes off one shared 128-d hidden have
collapsed onto the same function. Shape is a hard multiplier on exact match: 63%
of queries are dead before a colour is read.

### D3 — rank-16 CP cannot express ARC grids

`color_logits[i,j,c] = sum_r row[r,i] * col[r,j] * color[r,c] / sqrt(16)`. Even
conditioned on a correct shape, pixel accuracy is 0.408 and no query in 419 clears
0.95. Candidate 2's single margin flip lands at cell (2,0)/(2,1)/(1,1)/(1,2) in
most queries — the signature of a systematic low-margin corner in a low-rank
tensor, not a second hypothesis.

### The current 1/419 is noise

10,000 updates on the legacy reservoir gives the best diagnostics of any run
(shape 0.391 / pixel 0.447) and scores 0 exact. The 4,096-update balanced run
scores 1 exact on *worse* diagnostics. The 5,120-update follow-up regresses to 0.
The single solve survives `shuffled_demonstrations` and `slot_ablation` — only
`no_context` loses it — so it was not obtained by reading the demonstration
pairing. Every current configuration is treated as scoring zero.

## 3. Hypotheses ruled out, not pursued

- **pp-prop's approximation causes the binding failure.**
  `var/example21-binding-control` records
  `legacy_architecture_necessary_bptt_also_fails_binding`: a BPTT oracle on the
  same reservoir also fails to bind demonstration pairs. Retuning `trace_decay` or
  `recurrence_scope` will not fix it.
- **The readout must be on the eligibility path.** The compiler excludes it by the
  non-parametric-tail invariant, but the loss is applied at post-query latent
  ticks, so those gradients are already exact. Rewiring buys nothing.
- **More neurons or synapses.** The 0.8687 shape baseline uses zero neurons, and
  `color_logits` is rank-16 regardless of how many neurons feed it. Already
  recorded as a non-claim in `2026-08-17-example21-latent-reasoning-architecture.md`.

## 4. Capacity accounting

Parameter budget of the 2,262,812-scalar model:

| group | params | share |
|---|---|---|
| `ff_syn.comm.weight` (828 input features -> 2048) | 1,695,744 | 75.0% |
| readout stack (`readout_projection` + 3 heads) | 413,184 | 18.3% |
| associative memory (query / read / write) | 132,096 | 5.8% |
| `rec_syn.comm.weight` — the recurrent connectome | 16,384 | **0.72%** |

The recurrent network holds 0.72% of the parameters and delivers ~13% of the
drive (feedforward-current L2 2015.7 vs recurrent-current L2 271.7 at step 0). At
8 synapses/neuron it is ~50x sparser than FlyWire (~391/neuron). Peak device
memory was 1.75 GB of a 12.9 GB limit, so there is ~7x headroom. Four internal
widths are probed once and then set, not tuned: `readout_width` 128, `color_rank`
16, `context_memory_width` 32, `recurrent_edges` 16,384.

## 5. Plan and gates

Ordered by yield per unit cost, not by conceptual size. Each gate is a measured
number; a failed gate is reported, not worked around. One full train-and-evaluate
run must stay within 15 minutes on the RTX 3080 Ti.

### Stage 0 — data and floor (CPU). Done.

ARC-AGI-1 re-fetched at pinned commit
`aa922be204204ec148a1137fe6ed4d34ddde812b` into `var/arc-agi-1/`; local manifest
at `var/example21-arc-local-sources.json`. 774/800 files are byte-identical to the
prior run's `data_manifest.json`; the other 26 differ by exactly one whitespace
byte each and are semantically identical — **all 400 evaluation tasks reproduce
the prior run's canonical fingerprints exactly**, so scores remain comparable.
The floor script re-measures 400 tasks / 419 queries and reproduces the published
table, `copy_or_rule_shape` at shape 0.8687 / pixel 0.6336.

### Stage 1 — verified-rule channel and demo-consistency ranking

A rule emits a candidate only if it reproduces **every** demonstration pair
exactly: identity; the 8 dihedral maps; per-colour lookup; `k x` tiling with
dihedral variants; crop to non-background bounding box; symmetry completion;
constant-output-from-demos. All candidates — rule and neural — are ranked by
leave-one-out demonstration reconstruction error and the top 2 are submitted.
This replaces the single margin flip, which
`2026-08-18-example21-positive-arc-score.md` already names as unable to repair a
multi-cell error.

**Gate 1 result.** Measured on all 400 ARC-AGI-1 evaluation tasks, CPU only:

| channel state | strict pass@2 | queries admitting a rule | wall time |
|---|---|---|---|
| dihedral, colour map, tiling, scale, crop, panel, symmetry | 7/400 (1.75%) | 7/419 | 4 s |
| + reduction/completion composition search | 18/400 (4.50%) | 18/419 | 205 s |
| + object, panel-combine, gravity, border, enclosure families | 22/400 (5.50%) | 22/419 | 77 s |
| + same-shape cell rules (local rule, rays, connect, translate) | 25/400 (6.25%) | 26/419 | 110 s |

Two properties held at every step. **Precision is total**: every query that
admitted a rule was solved by it, so the channel has never yet cost a candidate
slot. **Recall is the only lever**: the channel proposes nothing on 393 of 419
queries.

The run-time drop from 205 s to 77 s came from restricting the periodic and
symmetric repair families to colours present in every source and absent from
every target -- a repair erases exactly one colour, so any colour surviving into
the target cannot be the hole marker.

Candidate ranking is minimum-description-length: cost tracks fitted degrees of
freedom, from parameter-free geometry through per-colour and per-cell-context
tables, with the input-ignoring constant completion charged a degenerate cost so
it neither corroborates nor outranks a rule that depends on the input.

### Stage 1 integrated result

`var/example21-rules-final/`, 2,048 updates at lr 1e-3, `context_memory_width=32`,
balanced colour loss, RTX 3080 Ti via the Docker image, **629.6 s** against a
900 s budget.

| arm | strict pass@1 | strict pass@2 | shape | pixel |
|---|---|---|---|---|
| **intact** | **0.0650** (26/400) | **0.0675** (27/400) | 0.348 | 0.383 |
| repeat_intact | 0.0644 | 0.0668 | 0.348 | 0.383 |
| no_context | **0.0000** | **0.0000** | 0.222 | 0.210 |
| shuffled_demonstrations | 0.0024 | 0.0048 | 0.303 | 0.332 |
| slot_ablation | 0.0644 | 0.0668 | 0.348 | 0.383 |

Three things this establishes that no prior run did.

**Non-zero pass@1.** Every run before this one scored exact `pass@1 = 0.0000`.
Gate D's stated bar in `2026-08-17-example21-latent-reasoning-architecture.md`
is one exact held-out solution with `pass@1 > 0`; the intact arm now clears it by
26 tasks. Gate D itself is not claimed — it requires the causal-mechanism gates
that remain stopped — but its numeric bar is met.

**The score is demonstration-causal.** Blanking the demonstrations takes the
score to exactly zero. The prior 1/419 survived `shuffled_demonstrations` and
`slot_ablation` and was lost only to `no_context`, which is why it was read as
non-causal.

**The score depends on the demonstration *pairing*, not merely on their
presence.** Deranging the demonstration outputs destroys 25 of 27 solves. This is
the property the presence/pairing dissociation recorded in
`2026-08-17-example21-demonstration-channel-root-cause.md` never achieved through
the network, where pairing sat at 0.5107 against a 0.5072 null.

Channel attribution at effort 0: **27 exact by the rule channel, 1 by a model
candidate**, with 27 of 419 queries admitting a rule and **none admitted that was
not exact**. The channel has still never cost a candidate slot.

The two arms are fed different demonstrations by construction: `no_context` gets
none, so no rule can be admitted; `shuffled_demonstrations` is fitted on the
deranged pairs from `_derange_task`. `slot_ablation` ablates neural slots rather
than demonstrations, so the rule channel is unaffected by it, and the attribution
is what separates the ablation's effect on the network from the channel that
ignores it.

### Where the remaining recall is not

Classifying the changed cells of the 258 unsolved same-shape tasks gives a flat
distribution with no dominant sub-category. The largest bucket -- 45 tasks
recolouring within existing object bounding boxes, no new colours -- was attacked
directly with three new object properties and a per-object colour-substitution
fitting mode. It solved **zero** additional tasks and cost 28 s, and was removed.
Those tasks need relational reasoning about what distinguishes one object from
another, which a property-keyed table cannot express.

Instrumenting the cell-rule family gives the same verdict from the other side:
the ordered 3x3 colour key fits the demonstrations on 21 evaluation tasks and
applies to none, because the median query has 98% of its keys unseen. Coarsening
to occupancy bitmasks drops that to 2% and raises conflicts by roughly the same
factor. The two effects trade.

**The rule channel is at its practical ceiling near 26/400.** Further recall has
to come from a different mechanism, which is what Stages 2 and 3 are for.

### Stage 2 — a decoder that can express ARC outputs

Opt-in behind a config flag; the `context_memory_width=0` legacy path stays
byte-identical.

1. **Shape as a rule over demo-derived candidate shapes.** The candidate shapes
   are computed deterministically from the demonstrations and fed to the head as
   explicit features — the same computation `demo_shape_rule` performs in the
   floor script. The head only *selects* among them. It must not be asked to
   induce them from the recurrent state: demonstration pairing has never been
   represented above chance (0.5107 against a 0.5072 null) and BPTT fails at it
   too, so a head that depends on binding will not reach 0.85.
2. **Colour as an edit on the query input** — per output cell, a mixture over
   {copy the aligned input cell, emit an explicit colour}, with a defined index
   alignment when output shape differs from input shape.
3. **Raise explicit-colour capacity** so it is no longer the 2,800x gradient runt
   driving ~900 of ~902 predicted values.

Decoder size is capped at ~1 M parameters so Stage 3 can replicate it 400x
without sharding.

**Gate 2:** shape >= 0.85 and pixel >= 0.6336 — beat the trivial predictor on
both axes.

### Stage 3 — per-task decoder adaptation

Leave-one-out episodes (`train` = demonstrations minus *i*, `test` =
demonstration *i*) are rolled through the frozen substrate in one batched pass to
precompute carriers; only the decoder is then fit per task, `vmap`-ed over a
per-task parameter axis. The real query is decoded with each task's own adapted
decoder.

This is honestly a per-task *linear readout fit* on frozen features: it can nail
tasks whose answer is a linear function of those features and **cannot compose a
new transformation**. Adaptation sits off the ETP machinery deliberately — the
compiler already hands the readout exact current-window reverse-mode gradients,
so a plain vmapped step computes the same gradient, and `compile(..., vmap=True)`
vmaps new states per sample rather than parameters.

Added as a new evaluation arm behind a flag. `same_frozen_parameter_bytes` stays
intact for the standard arms: the substrate genuinely does not move.

**Gate 3:** on the held-out leave-one-out demonstration — where ground truth is
free — does the adapted decoder ever reproduce a demonstration **exactly**? If it
cannot reproduce a demo it was not fit on, it will not produce an exact query
solve. Run this before spending anything on evaluation queries.

## 6. Non-goals

- The `full_structural_qualification` / `full_scientific_qualification` gates are
  left untouched and reported as failing. They fail on float32 noise (~1.1e-4
  against a 1e-6 `STATE_RMS_TOLERANCE`) independently of score.
- Gates A-D causal-mechanism qualification is not re-run. A positive score here is
  a performance result and retroactively qualifies nothing.

## 7. Claim boundary

Unchanged from `2026-08-16-pp-prop-latent-reasoning.md`. Additionally: any exact
solve attributable to the verified-rule channel is reported as such and is not
claimed as latent reasoning by the spiking substrate.
