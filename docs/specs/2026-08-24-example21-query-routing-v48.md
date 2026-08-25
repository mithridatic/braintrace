# Example 21 query-routing head V48

Status: approved for implementation

Date: 2026-08-24

Branch: `feat/example21-direct-generation`

## Objective

Test the primary-break conclusion of
`docs/specs/2026-08-24-example21-causal-phase-map.md`: the direct model
cannot express spatial routing, so no rearrangement operator (dihedral,
upscale, shift, mirror) is learnable on-generator. V48 adds exactly one
mechanism — a checkpoint-owned query-routing head — to the frozen V44
architecture. The encoder, experts, shape heads, loss, trainer, curriculum,
seeds, and schedule remain exactly fixed. This is one falsifiable mechanism.

## Evidence that selects this mechanism

The matched BPTT arm (artifact `var/ex21-online-v47-bptt-matched-arm-v1`)
reproduced the PP-prop family profile exactly (11/120: six copy, three
count, one pattern_label, one select_marked_region; zero routing families),
settling that credit assignment is not the binding constraint. The V1
architecture sketch sanctioned a checkpoint-owned cross-spatial attention
from query cells to output coordinates; it was never implemented in the
online line. Demonstration-output value transfer remains prohibited; only
query-input content is routed.

## Predeclared change: query-routing head

Architecture identifier `query_routing_gated_memory_v48`, answer-head
identifier `query_routing_hierarchical_decoder_v48`.

1. Event extension. Every decode-step event carries an additional fixed
   lossless block of 9,900 features: the complete query-input colour
   one-hots (900x10) and query-input validity mask (900), reconstructed
   from the decode-step replay rows of the same fixed encoding. The block
   is zero at every non-decode step. It contains only query-input
   information that already appeared earlier in the episode; no
   demonstration output, target, or derived selector is added. Event width
   becomes 3,831 + 9,900 = 13,731.
2. Routing sources. At each decode row, for each of its 30 output cells and
   each of the 900 query positions, a fixed 76-feature vector: the query
   colour one-hot (10), validity (1), and 65 fixed geometric match
   features — identity (1), integer shifts with dr,dc in -3..3 (49), the
   eight dihedral maps (8), integer upscale factors 2/3/4 (3), and the four
   mirror-concatenation maps (4). Match features are exact fixed arithmetic
   on position indices and the episode's lossless query height/width
   one-hots; no trainable or target-derived quantity enters them.
3. Routing attention. The BrainTrace ETP `attention_residual` primitive
   mixes the 900 sources per output cell with a checkpoint-owned query
   table of 16 programs x 76 features, initialized to exact zeros. A fixed
   softmax gate per output cell — softmax of a predeclared fixed seeded
   projection of the recurrent hidden state and the row/column one-hots —
   mixes the 16 program outputs. The gate projection is a declared
   constant, not a trainable leaf. The first ten mixed features (routed
   colour votes) are added to the cell logits at the predeclared fixed
   scale 4.0. The query table is the only new trainable leaf and reaches
   the logits through no other trainable primitive, satisfying PP-prop's
   weight-to-weight invariant by construction.
4. Everything else is unchanged: phase-separated MiniLSTM populations,
   twelve gated experts, shape heads, hierarchical serialization,
   `task_gated_fourth_root_v42` loss, single-step PP-prop trainer, v4
   curriculum, model seed 2108, synthetic seed 33108, holdout seed 44108,
   800 updates in chunks of 20, batch size 8, learning rate 0.001, trace
   decay `2 ** (-1 / 40)`.

## Experiment contract

One score-ineligible run, artifact `var/ex21-online-v48-query-routing-v1`:
fresh V48 model, identical v4 training curriculum and holdout to V47,
identical schedule. Scored on the same fresh 120-task v4 holdout (seed
44108) and, for context, the same 51-task real in-library scope and
fold-zero scope used by V47. The evaluation role is never opened.

## Gates

1. Mechanism gate: finite losses; zero compiler recurrent exclusions and
   zero weight-to-weight exclusions including the routing table; every
   parameter group and every ordered leaf (now 19) moves; candidate bytes
   change.
2. Routing gate (decisive): at least one strict task in at least one
   routing family (dihedral, upscale, mirror_concat, project_marker, or
   shift-covered crop) on the same-generator holdout. Any strict routing
   task proves the family is expressible and trainable. A zero result
   closes the query-routing mechanism and moves the break below the decode
   path — the next suspect is the recurrent encoder's operator
   representation, which would be measured by a probe that reads operator
   identity out of the frozen hidden state.
3. Anti-collapse gate: unchanged V47 thresholds on holdout diagnostics.

The real in-library and fold-zero scopes are diagnostics only; their
counts open or close nothing while gate 2 is unmet.

## Tests

Co-located tests with more than 90 percent meaningful coverage of the new
modules must prove: exact event-block reconstruction from replay rows and
its causality (the block at every step depends only on query inputs, never
on targets); block absence outside decode steps; exact match-feature values
for representative shift, dihedral, upscale, and mirror-concat cases,
including boundary behaviour; sources layout and validity masking; zero
query-table exclusion in the compiler report; finite nonzero gradients and
movement for every one of the 19 leaves including the table; deterministic
checkpoint roundtrip under the exact schema loader; candidate dependence on
the table (perturbing it changes emitted bytes); unchanged V44 behaviour
when the routing block is zeroed at construction.

## Non-goals

No curriculum, loss, trainer, optimizer, seed, or schedule change. No
demonstration-value routing. No complete-manifest evaluation regardless of
outcome; a passed routing gate earns one separately specified scaled arm.
