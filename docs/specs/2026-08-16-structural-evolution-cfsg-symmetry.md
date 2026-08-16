# Example 19 · Symmetry analysis of evolved recurrent wiring

Date: 2026-08-16
Status: implementation specification

## Question

Does Example 18's adaptive structural evolution change exact neuron
interchangeability or coarser structural roles, and do any exact
interchangeability classes align with task-attribution boundaries better than
the frozen-topology control?

This is a post-hoc descriptive analysis. It does not claim that the complete
automorphism group was computed, that a CFSG family caused learning, or that
the observed topology generalizes beyond the executed Example 18 run.

## Inputs and execution

Example 19 imports Example 18 and invokes its public command-line entry point
without changing the training configuration. It consumes each arm's recurrent
edge rows and columns, per-edge attribution labels, per-task gradient mass,
task names, and recurrent-neuron count.

All Example 18 command-line flags remain available. Repeated model execution
stays inside Example 18's `brainstate.transform` drivers; Example 19 performs
only one-shot NumPy analysis after the run.

## Exact twin classes

Two vertices are twins when transposing them preserves the directed binary
adjacency matrix, including their mutual edges and self-loop entries. Twin
classes generate a directly represented subgroup of the graph automorphism
group, `product(S_k)`, where each class has size `k`.

The report gives the base-10 logarithm of that subgroup's order and the
Jordan-Hoelder composition factors of each symmetric-group factor:

- `S_2`: `Z2`
- `S_3`: `Z3 . Z2`
- `S_4`: `Z2 . Z2 . Z3 . Z2`
- `S_k`, `k >= 5`: `A_k . Z2`

Singleton classes contribute no factors.

## Coarse role refinement

Directed one-dimensional Weisfeiler-Lehman refinement groups vertices using
their current color plus the multisets of outgoing- and incoming-neighbor
colors. Its stable color classes provide a coarse partition. The product of
factorials of their sizes is reported only as an upper bound on the full
automorphism-group order; it is not presented as the automorphism group.

## Attribution summaries

Each neuron receives the set of attribution labels on its incident edges.
Wired twin classes of size at least two are counted as pure only when every
wired member has the same singleton label set. All other wired classes are
mixed. This avoids traversal-order-dependent modal-label ties and does not
hide heterogeneous incident attribution behind a majority label.

For edges classified as shared, the report also counts the two tasks with the
largest gradient-mass contributions. Zero-mass edges are excluded. The task
pair matrix is symmetric with a zero diagonal; each qualifying edge contributes
once to an unordered pair.

## Reporting contract

For evolving and control arms, print:

- exact twin-class count and largest class;
- lower and coarse upper log-order bounds;
- exact subgroup composition factors;
- pure and mixed wired-twin counts;
- nonzero shared-edge task-pair counts.

The entry point returns Example 18's result unchanged. Example 19 does not add
or modify plot outputs. Topology analysis operates on sparse neighbor sets and
must remain O(neurons + edges) in storage; it must not materialize an
`n_rec`-squared adjacency matrix.

## Required tests

Co-located tests cover exact twins and non-twins, directed role refinement,
known symmetric-group factors and orders, isolated-neuron handling, pure versus
mixed wired classes, zero-mass filtering and pair counting, report output, and
a smoke entry-point run. Invalid shapes, endpoints, task counts, and labels must
raise clear `ValueError` exceptions rather than producing misleading output.

## Release boundary

This example is complete when its specification, implementation, co-located
tests, and README catalog row are committed; focused example tests and the
repository's normal example gate pass; and the branch is clean and pushed.
Generated plots and scratch experiment directories are development artifacts,
not release files.
