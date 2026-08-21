# Example 21 Attention Residuals

## Status

Approved implementation specification. This change is developed on
`feat/example21-attention-residual` and does not modify, merge into, or push
`main`.

## Motivation

The existing Example 21 row-refinement decoder exposes two residual-style
ablations: a fixed additive copy-logit bias and a learned carrier gate. Neither
implements Attention Residuals. The Attention Residuals paper instead assigns
one learned pseudo-query to each depth position, RMS-normalizes each prior
representation only for its key, applies a softmax over depth, and returns a
convex weighted sum of the original representations.

For Example 21, one complete 30-row refinement sweep is the decoder's closest
analogue to model depth. This change applies the paper equation across sweeps
without replacing the compiled per-tick `brainstate.transform.for_loop`.

## Public operator

Add the reusable function:

```python
attention_residual(
    sources,
    query,
    *,
    source_mask=None,
    query_index=0,
    epsilon=None,
) -> output
```

`sources` has shape `(..., source_count, hidden_size)`. `query` has shape
`(hidden_size,)` or `(query_count, hidden_size)`. A scalar query index applies
one query everywhere; an index array must match the leading source dimensions.
The mask is either `(source_count,)` or `(..., source_count)`.

For source `v_i`, key `k_i`, selected pseudo-query `q`, and valid-source set
`M`, the operation is:

```text
k_i = v_i / sqrt(mean(v_i ** 2) + epsilon)
alpha_i = exp(q . k_i) / sum(j in M, exp(q . k_j))  if i in M
alpha_i = 0                                          otherwise
output = sum_i alpha_i * v_i
```

The values are deliberately not normalized. `epsilon=None` resolves to the
source dtype's machine epsilon. An all-false mask returns exact zero weights
and output. The API rejects non-floating values, invalid ranks or dimensions,
non-boolean masks, non-positive/non-finite epsilon, and invalid query indices.

The implementation is a fused, gradient-enabled ETP primitive. It exposes the
query table as its one trainable input and packs the source positions, validity
mask, and query selection into the non-trainable input trace. This preserves
source positions instead of prematurely reducing them.

## Eligibility-trace rules

### pp-prop

The input trace retains the complete source axis. The solve rule evaluates the
query VJP against that retained representation. With zero trace decay, the
instantaneous query gradient is exactly the JAX VJP. Across a finite window,
pp-prop still applies its documented input/output factorization and smoothing;
outside its guaranteed regime bounded divergence from BPTT is expected rather
than element-wise equality.

### D-RTRL

The D-RTRL trace keeps both query-coordinate and output-coordinate axes. The
instantaneous rule inserts the full `d output / d query` Jacobian, the
recurrence rule propagates it position-wise, and the solve rule contracts only
the retained output axis. Under diagonal hidden recurrence this matches the
BPTT query gradient element-wise.

The primitive is intentionally unanchored for SnAp-n. One query coordinate can
influence every output coordinate, so sparse-position traces are rejected
rather than silently approximated.

## Module

Add `braintrace.nn.AttentionResidual(hidden_size, query_count=1, epsilon=None)`.
It owns a `ParamState` query table initialized to exact zeros and routes calls
through `braintrace.attention_residual`. Its pure `attention_weights` diagnostic
returns the same weights without mutating module or random state. Public
documentation uses NumPy-style sections and self-contained doctest examples.

## Example 21 configuration

Add one explicit decoder choice:

```python
refinement_mixer: Literal[
    "linear",
    "carrier_gate",
    "attention_residual",
]
```

The initial default remains `"linear"`. Existing `copy_residual_gain`,
`row_head_carrier_scale`, `row_head_carrier_gate`, and
`shape_head_carrier_scale` arguments remain supported as named ablations.
`row_head_carrier_gate=True` is accepted as a compatibility spelling of
`refinement_mixer="carrier_gate"` only when the new option is left at its
default. Explicit conflicting mixer/legacy settings fail with an actionable
error. Copy bias and carrier gating are documented as ablations, not paper
implementations.

## Same-seed initialization

For each row or shape proposal, draw one matrix at the complete linear input
width and scale it by that complete width. The linear and Attention Residual
proposal heads use the complete matrix. The carrier-gate event and carrier
heads use literal row slices of that matrix. Consequently a zero carrier gate
is exactly equal to the carrier-free event slice of the linear head, with no
approximately 3.4-times event-only initialization scale confound.

## Refinement integration

Let `max_sweeps = refinement_steps // 30`; create one pseudo-query per sweep.

- Row sources are the original query-row representation, proposals from all
  completed sweeps for that row, and the current row proposal.
- Shape sources are the original query height/width representation, proposals
  from all completed sweeps, and the current partial-sweep proposal.
- Existing RMS-balanced identity-source scaling is applied before stacking.
- At zero query initialization, every valid source has equal weight, so the
  initial output is the exact uniform mean.

Row and shape proposal heads write to dedicated `HiddenState` objects before
mixing. This gives the current proposal an ordinary direct pp-prop path.
Completed-sweep proposals are stored in resettable, snapshot-compatible
`ShortTermState` caches. Historical cached sources participate in the exact
forward equation but are stop-gradient inputs for retrospective proposal-head
credit. Extending the compiler with position-preserving scatter traces is
outside this change and this online-learning limitation must remain explicit in
user-facing evidence.

All refinement ticks continue to run through the existing compiled
`brainstate.transform.for_loop`; no repeated Python model loop is introduced.

## Qualification routing

Replace row/shape qualification path literals with one architecture-aware
manifest derived from model configuration. The manifest includes the linear
head paths, all three carrier-gate paths, or the proposal and pseudo-query paths
for Attention Residuals, as applicable. Every required path must compile and
all of its hidden-state route classifications must be `all_direct`.

## Evidence artifact

Large raw benchmark artifacts remain ignored. Commit one compact JSON manifest
containing revision, exact command, deterministic environment, seeds,
configuration, source artifact hashes, metrics, runtime, peak VRAM,
structural-gate results, zero-query counterfactual, repeatability
classification, and final qualification/promotion status. Missing benchmark
fields are represented explicitly as `null` or `not_run`, never inferred.

## Tests

Tests are co-located with the modules they cover. Bug regressions are written
before their fixes for stale qualification paths and unfair same-seed
initialization. New coverage includes:

- forward output and weights against an independent JAX equation;
- query VJP, inactive-query zero gradients, masks, batched indices, leading
  dimensions, JIT, and vectorization;
- zero sources, all-false masks, finiteness, and convex weights;
- zero-initialized uniform averaging and nonzero query gradients;
- pp-prop's finite-window guaranteed regime and bounded divergence elsewhere;
- D-RTRL versus a small BPTT oracle;
- reset/snapshot/restore and refinement source-history indexing;
- compilation and `all_direct` routing for all three mixer modes;
- exact zero-gate/event-slice equivalence and legacy configuration behavior.

Meaningful coverage of new operator/module code must exceed 90 percent.

## Validation and benchmark

Run focused tests first, then core `braintrace` tests with coverage, typing and
build checks, and the complete 19-module Example 21 numerical/oracle gate with
six workers and `--dist=loadgroup` rather than `-n auto`.

If those gates pass and the required GPU/runtime prerequisites are present, run
a paired full ARC-AGI-1 pilot for seed 2108 using 4,096 neurons, 4,096 edges,
batch 32, 260 updates, 60 latent steps, learning rate `1e-3`, and the fixed
raw-JAX sparse backend. Continue with seeds 31337 and 7777 only if both pilot
arms compile, preserve valid artifacts, and finish within 15 minutes each.
Stop the full benchmark at 45 minutes.

Attention Residuals may become the default only if all structural gates pass,
no seed loses more than 0.02 query-weighted pixel score, exact-answer total is
noninferior, copy and shape means regress by no more than 0.02, and either:

1. mean pixel and rule-at-oracle both improve by at least 0.01; or
2. exact-answer total increases while mean pixel stays within 0.001.

Otherwise it remains opt-in and the negative or incomplete result is recorded
without an unplanned tuning campaign.
