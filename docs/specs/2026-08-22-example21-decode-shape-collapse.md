# Example 21: the protocol-v2 "regression" was the decode rule

Date: 2026-08-22
Branch: `feat/ex21-decoder-shape-fix`

## Summary

Two independent defects were masking each other in Example 21's model-only
channel.

1. **The protocol-v2 decoder collapses every grid onto one by one.** Ranking
   all 900 shapes by the shape log probability *plus the summed included-cell
   colour log probability* subtracts a non-positive term for every cell, so
   the score falls monotonically with area. Shape accuracy 0.5752 -> 0.0072.
   This is the whole of the recorded "v2 regression"; the model was never
   involved.
2. **The strongest recorded model-only base is an identity map.** Measured at
   113,982 of 113,982 overlapping cells at efforts 30 and 60, its colour argmax
   equals the test input exactly. The mechanism is `--row-head-carrier-scale 0.0`.
   `_refinement_head_input` concatenates the workspace carrier with the
   event-derived blocks (row-position one-hot, the query colours of the row
   being transcribed, the query's grid dimensions), and **the carrier is the
   only pathway that carries the demonstrations**. Scaling it to zero leaves the
   row head reading nothing but the query's own input row, so the best map it
   can learn is the identity — and training finds it. The `copy_residual_gain`
   bias does not compound across refinement ticks (`next_row = row_proposal`
   replaces rather than accumulates); the collapse is the learned head, not the
   residual.

## Proof of (1)

Replay the strongest recorded logits (`var/logits-s31337.npz`, the s31337
cr2/cs0 full-scale run, `EXAMPLE21_LOGITS_DUMP`) through both decoders:

| decoder | shape acc | pixel acc | q@1 | q@2 | strict@1 | strict@2 |
|---|---|---|---|---|---|---|
| pre-v2 (`bde13ba`) axis argmax | 0.5752 | 0.5484 | 1 | 2 | 0 | 1 |
| protocol-v2 joint sum | 0.0072 | 0.0332 | 1 | 1 | 1 | 1 |

The protocol-v2 column reproduces `var/example21-rtm-main-full`'s recorded
shape 0.0072 / pixel 0.0160 on logits from a model that scores 0.5752 / 0.5484.

Shape-scoring variants on the same logits at effort 60:

| rule | shape acc | top-2 | top-5 |
|---|---|---|---|
| shape heads only | 0.5752 | 0.6086 | 0.6563 |
| shape heads + summed cells (v2) | 0.0072 | 0.0072 | 0.0406 |
| shape heads + mean cell | 0.5776 | 0.6110 | 0.6587 |

### Why the shape heads alone are the correct rule

`_arc_loss_vectors` masks the colour cross entropy to cells inside the target
shape (`jnp.where(valid, color_nll, 0.0)`). The colour head is never supervised
on whether a cell lies inside the grid, so its summed log probability is not a
calibrated quantity to compare across shapes. The factorization is a valid
normalized distribution over grids, but its MAP is dominated by the
area-monotone term, and empirically the shape marginal it induces is 80x worse
than the heads trained for the job.

## Consequence for the recorded run history

**Every Example 21 result recorded after `e338e84` was scored through the
broken decoder and says nothing about its model.** The 114-run survey in
`var/` must be re-read with that in mind. Re-scoring three full-scale
checkpoints offline (`--training-updates 0` eval-only dump, 32 s each) gives:

| checkpoint | recipe | shape | pixel | input echo | small-grid cell acc | cumulative |
|---|---|---|---|---|---|---|
| `example21-rtm-s31337` | cr2 / cs0, muon 1e-3, u260 | 0.5752 | 0.5484 | **1.000** | 0.268 | 4 |
| `example21-full-muon-cr2g` | cr2 + carrier gate, muon, u260 | 0.5346 | 0.5382 | 0.993 | 0.285 | 2 |
| `example21-full-cr2g-sub30` | cr2 + carrier gate, adam 3e-3, u260 | 0.2959 | 0.3490 | 0.879 | 0.260 | 0 |
| `example21-full-default-u390` | no copy residual, muon, u390 | 0.5442 | 0.4425 | **0.652** | 0.336 | 0 |
| `example21-rtm-main` | main defaults: latent_row_decode, learned_update, cr2 / cs0 | 0.1408 | 0.3524 | **1.000** | 0.268 | 2 |

Cumulative is `q@1 + q@2 + strict@1 + strict@2` in absolute counts.
`example21-full-default-u390` is the only checkpoint on disk that is not
substantially a copy of the query input, and it is also the only one with
above-copy small-grid accuracy.

## Baselines and ceilings (evaluation split, 419 queries / 400 tasks)

- Emitting the query input verbatim: shape 0.6611, pixel 0.6032, **cumulative 0**.
- Oracle shape + top-left crop of the query input: **cumulative 6**, and the two
  solved queries are exactly the two the model already solves (`bbb1b8b6` q1,
  `e872b94a` q0). The crop-of-input architecture is fully exhausted at 2.
- Under an oracle shape, the colour argmax is exact for 2 of 419 queries,
  within 1 cell for 5, and within 2 cells for 9. No decode policy over these
  logits exceeds cumulative 6.
- Per-cell colour accuracy by output area: 1-9 cells 0.229 (n=26), 10-16 0.321
  (n=19), 17-36 0.292 (n=49), 37-100 0.588 (n=96), 101-900 0.757 (n=229). Exact
  match lives in the small grids, which is where the model is weakest, because
  its accuracy there *is* the copy baseline.

## Fix

`decode_candidates` ranks shapes by the height and width heads alone. The
runner-up compares its two alternatives (second-ranked shape, single
second-colour substitution) as log likelihood ratios against candidate one so
they share units, and the reported `log_probability` is monotone in rank.

`input_echo_fraction` reports the share of rank-one cells that repeat the query
input at the same coordinate, per query in `result.json` and averaged per
effort in `report.txt`, so a copy-collapsed run can never again be reported as
a score without the number beside it.

## Families measured, and what is not worth building

Measured on both splits (test pairs: 416 training, 419 evaluation):

| family | training | evaluation |
|---|---|---|
| output is some window of the input | 32 (7.7%) | 19 (4.5%) |
| output equals a demonstration output | 17 (4.1%) | 10 (2.4%) |
| output is the input top-left window | 7 (1.7%) | 2 (0.5%) |
| single dihedral transform fixed by the demonstrations | 7 (1.7%) | 0 |
| colour permutation fixed by the demonstrations | 4 (1.0%) | 0 |

A learned pointer into the demonstration outputs was scoped and rejected: among
the 17 training collisions the correct demonstration is the first for 3 and the
last for 6 out of a mean 6.9 demonstrations, so no query-only signal selects it.
Choosing correctly needs the pairing sensitivity that the width, load,
learned-keys, `etp_outer_write` and exact-D-RTRL probes each failed to produce
(see `docs/specs/2026-08-21-example21-drtrl-arc-pilot.md`). 17 supervised
examples is also far below what the shape head needs (416) to reach 0.575.

## Acceptance run

`var/ex21-fixed-controls-s31337` reproduces the s31337 cr2/cs0 recipe on the
fixed branch with all five evaluation control arms enabled (protocol v2
default), 260 updates, seed 31337.

- Effort 60 (primary submission): query pass@1 = 1, pass@2 = 2, strict task
  pass@1 = 0, pass@2 = 1. **Cumulative 4.** Shape 0.5776, pixel 0.5489.
- `model-only input echo` prints **1.0000** at efforts 30 and 60 and 0.4410 at
  effort 0: the refinement sweep is what drives the collapse, and the artifact
  now says so on its own.
- `required_controls_executed: True` and no "recomputed intact control metrics
  are inconsistent". The new decode rule threads through intact, repeat_intact,
  no_context, shuffled_demonstrations and slot_ablation consistently.
- `repeat_intact_deterministic`, `slot_ablation_pre_intervention_matched` and
  `associative_diagnostics_complete` are False, exactly as they are in the
  pre-fix controls-on run `var/example21-rtm-main-full`. That is GPU
  non-determinism against a 1e-6 RMS tolerance, unchanged by this work;
  `metrics exact` is True for the repeat arm.

The offline replay in the session scratchpad reproduces this run's four-tuple
and both diagnostics to the digit, so decode policies can still be evaluated
without the GPU.

## Where the cumulative score stands

| channel | q@1 | q@2 | strict@1 | strict@2 | cumulative |
|---|---|---|---|---|---|
| `main` before this branch, effort 60 | 0 | 0 | 0 | 0 | **0** |
| `main` after this branch, effort 60 | 1 | 2 | 0 | 1 | **4** |
| identity baseline (emit the query input) | 0 | 0 | 0 | 0 | 0 |
| oracle shape + top-left crop (architecture ceiling) | 2 | 2 | 1 | 1 | 6 |

No decode policy over the recorded logits exceeds 6, and the copy path that
produces the two solved queries is exhausted at 2 of 419. Raising the score
further requires a model that is not a crop of its input, which is the
demonstration-pairing problem that remains open.

## The update count is not what is binding

`var/ex21-default-u700` runs the non-degenerate recipe (no copy residual,
carrier at full scale, `row_refinement`, muon 1e-3 cosine, seed 9999) for 700
updates:

| | shape | pixel | input echo | small-grid cell acc | ham=0 under oracle shape | cumulative |
|---|---|---|---|---|---|---|
| `ex21-default-u700`, 700 updates | 0.5919 | 0.4769 | 0.654 | 0.317 | 0 | 0 |
| `example21-full-default-u390`, 390 updates | 0.5442 | 0.4425 | 0.652 | 0.336 | 1 | 0 |

**Do not read the two rows as a curve.** `example21-full-default-u390` came off
a different source tree with `training_chunk_size = 0` against this run's 5, so
the pair differs in more than the update count, and a 0.336 -> 0.317 move on a
45-query band is inside seed noise.

What the run does establish, on its own:

- At 700 updates **no query reaches Hamming zero under an oracle shape**, and
  small-grid per-cell colour accuracy is 0.317 — the same regime as every other
  checkpoint measured here (0.260 to 0.336).
- Shape accuracy is 0.5919. `example21-training-regime-was-the-blocker` records
  shape 0.59 at **6,000** updates. Reaching the 6,000-update shape accuracy at
  700 says the update count is not the binding constraint.
- Exact match on a 9-to-16 cell grid needs roughly 0.85 per-cell accuracy. The
  model sits at 0.32 wherever it is measured.

Measured throughput on the RTX 3080 Ti laptop, for planning: **1.65 s/update**
at 4096 neurons / 4.19M edges / batch 32 (700 updates = 1151.6 s training,
1188.6 s wall for the whole run). A 260-update run with all five control arms is
716.9 s. An earlier 1.0 s/update figure was read off a partially warm cache and
is wrong.

## What would have to change

Every route measured here is closed:

- **Decode.** No policy over the recorded logits exceeds cumulative 6.
- **More training.** 700 updates already reaches the shape accuracy that 6,000
  updates reached, colour accuracy on small grids stays near 0.32, and no query
  reaches Hamming zero under an oracle shape.
- **Augmentation.** Dihedral, colour-permutation and demonstration-shuffle
  augmentation are already on by default (`AugmentationConfig`).
- **A pointer into the demonstration outputs.** No query-only signal selects the
  right demonstration (see the family table above).

What remains is the open problem the `var/example21-drtrl30*` pilots left: the
model does not condition on the demonstration pairing, so it can only learn a
marginal prior over ARC outputs, and that prior is approximately the identity.
Raising the exact-match count requires making the pairing usable, not tuning
what is here.
