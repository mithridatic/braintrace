# Example 21 surface-diversified curriculum V47

Status: approved for implementation

Date: 2026-08-24

Branch: `feat/example21-direct-generation`

## Objective

Test the primary-break hypothesis of
`docs/specs/2026-08-24-example21-causal-phase-map.md`: the direct model's
operator recognition is keyed to the v3 synthetic generator's surface
statistics and therefore never fires on real ARC instances of the same
transformation. V47 changes only the training data's surface statistics. The
architecture (V44 phase-separated gated memory), the trainer (single-step
PP-prop, `task_gated_fourth_root_v42` loss), the answer path, and the
scoring code remain exactly fixed. This is one falsifiable mechanism, not an
architecture version.

## Predeclared change: curriculum schema v4

New module `latent_workspace_diverse_curriculum.py`, schema
`direct_synthetic_curriculum_v4`, task identifiers
`synthetic-v4:{family}:{index:06d}`. The v3 module and every existing runner
stay byte-untouched; a zero-v4 configuration remains behaviorally identical
to the ordinary path.

The eleven v3 operator families are retained with unchanged operator
semantics: copy, recolor, dihedral, crop, upscale, count, pattern_label,
select_marked_region, project_marker, complete_corner, mirror_concat. Within
each family, only the surface statistics are diversified to span the
measured public-training distribution:

- side lengths sampled from 3 through 30 (v3 capped at 9; real inputs have
  median side 10, 34.75% exceed 12, maximum 30);
- foreground density spanning roughly 0.05 through 1.00 (v3 fixed 0.28 or
  0.35; real median 0.30 with p10 0.05 and p90 1.00);
- one through six distinct foreground colors per input (v3 recolor used
  exactly one; real median 3, p90 6);
- scene structure drawn from four painters — Bernoulli noise, solid
  rectangles, straight lines, random-walk blobs — instead of independent
  noise only, so inputs include coherent objects;
- demonstrations per task vary from 2 through 6 (v3 fixed 4; real tasks
  span 2 through 10);
- recolor maps one through three source colors and may leave other colors
  unchanged; crop keeps the foreground bounding box of a sampled color or of
  all foreground, with padding margins beyond one cell.

Every task samples its transformation parameters once and applies the same
operator to all demonstrations and the query, keeps every grid within 30x30
and colors within 0..9, uses only `brainstate.random`, and records exact
family counts and the ordered canonical task digest. Family weighting
doubles `pattern_label` as in v3. The generators never read validation or
evaluation tasks; the fixed-validation examples justify operator classes
only, never grids, colors, sizes, or outputs.

## Experiment contract

One score-ineligible run, artifact `var/ex21-online-v47-diverse-curriculum-v1`:

- model: `phase_separated_gated_memory_v44`, memory width 128, twelve
  experts, model seed 2108, built fresh;
- data: 1,400 v4 tasks, seed 33108; no public-ARC fitting of any kind;
- training: 800 updates in chunks of 20, batch size 8, learning rate 0.001,
  trace decay `2 ** (-1 / 40)`, single-step PP-prop, augmentation off (the
  diversification replaces it);
- synthetic holdout: 120 fresh v4 tasks, seed 44108;
- real scopes, scored target-free after training, never fitted:
  1. `in_library_public_training`: every public training task whose
     operator hypothesis (copy, recolor, dihedral, crop, upscale,
     downscale) holds for all of its pairs, selected by the deterministic
     declared classifier — 51 tasks at analysis time;
  2. `held_out_public_training` fold zero: the same canonical fingerprint
     positions 0..79 that V45 scored 0/80, for direct comparability.

## Gates

1. Mechanism gate (unchanged): finite losses, zero recurrent exclusions,
   every parameter group and ordered leaf moves, candidate bytes change.
2. Synthetic gate: at least one strict task in at least two distinct
   families on the untouched v4 holdout, proving the path still learns under
   diversification. This gate is necessary but non-predictive.
3. Decisive L4 gate: strict count above zero on `in_library_public_training`.
   A pass shows the recognition gap closes under surface diversification and
   justifies a scaled, separately specified follow-up. A fail promotes the
   row-event encoding (map layer L3) to primary suspect and closes the
   curriculum-diversification hypothesis; no rerun with more updates or
   seeds is permitted without a new mechanism.
4. The fold-zero scope is diagnostic context only; its count does not open
   or close anything by itself.

## Tests

Co-located `latent_workspace_diverse_curriculum_test.py` must prove, with
meaningful coverage above 90 percent of the new module:

- byte-identical output and digest for a repeated seed and a different
  digest for a different seed;
- exact round-robin family counts including doubled `pattern_label`;
- every grid within 1..30 per side and every color within 0..9;
- surface diversity is real: across a fixed seed, generated side lengths
  exceed 9, densities span at least 0.10..0.90, and inputs with three or
  more distinct colors occur;
- operator semantics per family: copy equality, consistent recolor mapping,
  exact dihedral transform, exact sub-rectangle crop, exact integer
  upscale, count color equals object count in a 1x1 output, 1x1 label
  outputs, marker projection, corner completion, exact mirror
  concatenation;
- transformation parameters are constant across all pairs of a task;
- unambiguous targets (for example unique argmax in select_marked_region);
- ArcTask structure: demonstrations in `train`, query as the sole `test`
  pair.

## Non-goals

No architecture, loss, trainer, optimizer, seed, or decoder change. No
public-ARC fitting. No evaluation-manifest access. No complete-manifest run
follows from this experiment regardless of outcome; a pass earns one
separately specified scaled arm.
