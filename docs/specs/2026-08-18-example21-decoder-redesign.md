# Example 21 — decoder redesign (Stage 2 design, not yet implemented)

Status: designed, not implemented. Prerequisite gate identified in §7.
Branch: `feat/example21-latent-reasoning`
Follows `2026-08-18-example21-arc-score-recovery.md`, whose Stage 1 is complete.

## 1. Why

Stage 1 raised exact `pass@1` from 0.0000 to 0.0650, but 26 of its 27 solves come
from the verified rule channel. The spiking model itself is untouched and still
loses to the repo's own trivial predictors on both axes exact match requires:

| metric | model | trivial predictor |
|---|---|---|
| shape accuracy | 0.348 | 0.869 (`copy_or_rule_shape`) |
| pixel accuracy | 0.383 | 0.634 (`copy_or_rule_shape`) |
| pixel given shape correct | 0.408 | — |
| queries reaching pixel >= 0.95 | 0/419 | — |

The three defects are D1 (the output is generated from scratch, with no reference
to the query input), D2 (79.2% of predicted shapes are square because two 30-way
softmaxes off one shared hidden collapsed onto the same function), and D3 (a
rank-16 CP colour tensor cannot express ARC grids).

## 2. Blast radius decision

Fold the whole new decoder inside `compact_readout`, reading the model's own
captured query grid, so the compact vector carries already-final
`height(30) | width(30) | colors(30*30*10)` = **9060** rather than 1180.

Consequences, verified against the code:

- `OutputLogits`, `decode_candidates`, `score_query_candidates`,
  `_candidate_log_probability` and `analyze_latent_trajectory` all consume
  `(30)/(30)/(30,30,10)`: **no changes required**.
- `_training_row`, `_TrainingTensors`, `_CHUNK_ARRAY_FIELDS`, `_stacked_chunk`,
  `_training_chunks` and `train_all`'s signature are **unchanged**, so
  `test_chunked_training_reproduces_unchunked_losses_bitwise` and the
  full-size-draws-up-front constraint hold by construction.
- Everything trainable stays inside `model.update`, so the ETP compiler registers
  it in `learner.param_states` and Adam actually updates it. A decoder applied at
  expansion time would silently never train.

## 3. Layout

`ModelConfig` gains `decoder_mode: Literal["legacy_cp", "edit_rule"] = "legacy_cp"`,
`copy_prior_logit`, `shape_prior_logit`, and four `query_*` slice starts required
iff `decoder_mode == "edit_rule"`. `compact_output_width` and
`expand_compact_logits` keep their exact current bodies; a new
`decoder_output_width` / `expand_decoder_logits` pair dispatches on the mode.

New heads, all `braintrace.nn.Linear` so they route through the same ETP `matmul`
as the existing heads:

| head | shape | init |
|---|---|---|
| `shape_rule_head` | `readout_width -> 2*13` | bias `shape_prior_logit` on the `same` slot |
| `shape_absolute_head` | `readout_width -> 2*30` | zero bias |
| `copy_gate_head` | `readout_width -> 30*30*4` | bias `copy_prior_logit` on the `copy` slot |
| `color_palette_head` | `readout_width -> 10` | zero bias |
| `color_explicit_head` | `readout_width -> 30*30*10` | zero bias |

All new construction sits **strictly after** the four existing heads and inside
the flag branch. Any `random.randn` draw inserted earlier would change every
legacy weight and break the three byte-identity tests.

Query capture uses **`brainstate.ShortTermState`**, not `HiddenState`, so the
buffers land in `_other_states` and cannot form a hidden group or attract an
eligibility trace.

## 4. Shape as a rule

`SHAPE_RULE_MATRIX (12, 30, 30)` and `SHAPE_RULE_SOURCE (12, 2)` are fixed
module-level constants: rule *r* maps input size *i+1* to output size *s+1*, and
which input axis each rule reads. The head produces a gate over 12 deterministic
rules plus one learned absolute fallback, and the output is a normalised
log-probability via `logsumexp`.

This is what kills the `h == w` collapse: the height and width gates select over
*different* deterministic bases, so identical gate vectors still produce
different outputs whenever `H_in != W_in`. Collapse now requires both axes to
pick `absolute` **and** the two absolute heads to coincide.

The candidate shapes are computed from the demonstrations deterministically and
fed in as features. The head only *selects*. It must not be asked to induce them
from the recurrent state: demonstration pairing has never been represented above
chance (0.5107 against a 0.5072 null) and BPTT fails at it too.

## 5. Colour as an edit

Per output cell, a mixture over `{copy, copy_transpose, palette, explicit}`.
Alignment when shapes differ is nearest-neighbour on the **predicted** output
shape: `src_row(i) = clip(floor(i * H_in / H_out), 0, H_in - 1)`, likewise for
columns. Train and decode use the same rule, so there is no train/test gap.
Teacher-forced alignment was rejected: it would move the colour mixture out of
`model.update` and thereby out of `learner.param_states` entirely.

## 6. Post-scan readout (required, not cosmetic)

`compact_readout` currently runs at every one of ~330 ticks while only 33 are
kept. At width 9060 and batch 419 that is a ~500 MB scan carry and ~10x wasted
FLOPs across five arms. Record the decoder inputs at the selected ticks and apply
the readout once after the scan, branching on `memory_enabled` for the carrier
(`voltage_buffer` when memory is on, `spikes_buffer` otherwise). Wiring
`voltage_buffer` unconditionally silently changes legacy numerics. Land this as
its own commit: it is independently valuable and independently verifiable against
byte equality with the per-tick path.

## 7. Prerequisite gate — run this first

**The ETP-classification probe must run in the container before anything else in
Stage 2.** Static evidence says a `ShortTermState` lands in `_other_states` and
cannot form a hidden group, and that the new heads share the existing heads'
`excluded_weights` classification, but this was never executed. Assert, for an
`edit_rule` model: every new head path in `report.excluded_weights`, none in
`report.etrace_weights`, and `report.hidden_groups` identical to the legacy
model's.

If it fails, the fallback is threading the query grid as an extra `xs` through
`etrace_grad` — a larger diff that loses the "training stream unchanged" property
currently protecting the bitwise chunking test.

## 8. Acceptance

- Shape >= 0.85 (now 0.348); `shape_square_fraction` falls from 0.792; the top-8
  predicted-shape histogram contains at least one `h != w` entry.
- Pixel given correct shape >= 0.75 (now 0.408); overall pixel >= 0.634; the
  count of queries at pixel >= 0.95 exceeds 0 (now 0/419).
- `norm(g(readout_projection)) / norm(g(color_explicit_head)) < 50`, against the
  measured ~2800.
- Decoder stays under ~1 M parameters so Stage 3 can replicate it 400x.

## 9. Honest expectation

Stage 2 raises the *diagnostics*. It does not by itself add exact solves: exact
match needs per-task induction, which is Stage 3. The value of Stage 2 is that it
makes the model stop losing to a predictor that uses no neurons, and it is the
prerequisite for Stage 3 having anything worth adapting.
