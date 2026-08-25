# Example 21 scene-summary decode channels V50

Status: drafted, conditional on the V49 scaling readout

Date: 2026-08-24

Branch: `feat/example21-direct-generation`

## Objective

Attack the two nearest measured walls with one mechanism. On the V48
holdout, count and select_marked_region tasks are shape-solved but
label-wrong (7/10 and 9/10 within one cell), and crop is 0/10 on output
shape. Both failures are exact-discrete-readout failures: the answers are
determined by simple scene statistics (object counts, bounding-box
extents) that the 128-float recurrent state cannot hold precisely. V50
adds fixed, task-neutral, input-only scene-summary channels to the decode
path. The full lossless encoding is retained; the summaries are redundant
fixed arithmetic channels of the same class as the existing validity-mask
and coordinate planes, computed from query inputs only — never from
targets. Every logit still comes from checkpoint-owned heads consuming
recurrent state and these channels; no channel emits an answer by itself.

## Evidence

V48 (artifact `var/ex21-online-v48-query-routing-v1`): count 1/10 exact
with 7 near, select_marked_region 1/10 with 9 near, crop 0/10 shape; on
the real fold-zero scope, `27a28665` is exact on two of three queries.
Wrong labels are half misindexed associations and half degenerate
query-colour echoes. Probe D shows the encoder represents operator
identity; the failure is precise discrete readout, not representation.

## Predeclared change (conditional on V49 not clearing the label wall)

Architecture identifier `scene_summary_routing_v50`, a V48 successor with
identical encoder, routing head, loss, trainer, curricula, seeds, and
schedule. One addition: fixed scene-summary channels appended to the
query-grid block, consumed by the cell and shape heads:

- valid-cell count, one-hot over 0..29 plus an at-or-above-30 bucket (31);
- per-colour valid-cell counts, ten channels with the same 31-bin
  one-hot scheme (310);
- foreground bounding-box height and width, each one-hot over 1..30 (60),
  computed from the query validity mask; empty grids decode as zero
  channels.

These channels are exact fixed arithmetic on the lossless query-input
block, appended to it (401 new block features; event width 13,731 to
14,132). The cell head and shape heads consume them alongside the existing
features through their unchanged checkpoint-owned Linears. No new
trainable primitives beyond the widened head weights; the
weight-to-weight invariant is preserved because the channels are inputs,
not stacked trainables.

## Gates

1. Mechanism gate: unchanged V48 definition (all leaves move, no
   weight-to-weight exclusion, finite losses, candidate bytes change).
2. Summary gate (decisive, measured on the same fresh v4 holdout):
   count exact at least 5/10 (from 1/10) AND crop shape-exact at least
   4/10 (from 0/10), with no strict-family regression below V48's 12/120
   total and at least one routing-family strict retained.
3. Anti-collapse gate: unchanged thresholds.

A pass promotes the mechanism into the nomination line. A fail closes the
fixed-summary direction; the next mechanism would be demo-association
readout (per-demonstration exact-match channels), specified separately
with its own evidence.

## Conditional clause

If V49's 2400-update readout already clears count exact to at least 5/10
and crop shape to at least 4/10, this arm is withdrawn: duration, not
summary channels, was the constraint, and the next arm instead scales what
V49 proves.

## Non-goals

No demo-output value channels, no learned gate projections, no curriculum,
loss, seed, or schedule change, no real-task fitting, no complete-manifest
evaluation regardless of outcome.
