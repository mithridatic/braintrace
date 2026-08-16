# Example 21 · In-context rule binding and iterated latent computation

Date: 2026-08-16
Status: implementation specification

## Question

Can a recurrent spiking network trained by pp_prop acquire a rule from
demonstrations at inference time, hold it in a contextual memory written without
changing any parameter, and compute with it across repeated latent iterations —
and does iterating that latent workspace change the result over reading the
memory directly?

This instantiates the system-level interface published in *BDH-CQ: In-Context
Learning with Recurrent Latent Reasoning* (arXiv:2608.09888), Eqs (1)–(4). It is
not a reproduction of that system. Section 3.4 states that its dimensions and
update rules are proprietary and Section 4.1 states the same of its training
recipe; none of that is replicated or inferred here. No ARC-AGI-1 score, no
inference-cost figure, and no benchmark claim of any kind is made. The example
also asserts nothing about pp_prop's gradient estimate.

## The interface, as instantiated

| Published | Here |
| --- | --- |
| `S_t = U_θ(S_{t-1}, D_t)` | Hebbian write `S_t = S_{t-1} + v_t k_tᵀ`, held factored as `(A, B)` over `M` slots. `k_t`, `v_t` are rectified projections of the ingestion population's rate. |
| `H_0 = E_θ(x*, S_K)` | latent state after the query span, with the memory read folded in |
| `H_{r+1} = F_θ(H_r, S_K)` | one zero-input tick of the latent population: recurrence plus the memory read |
| `ŷ = G_θ(H_R)` | linear readout over the symbol set |

The linear-attention special case named in Eq (1), `S_t = S_{t-1} + U_θ(D_t)`, is
the case implemented.

## Inputs and execution

One episode is a single contiguous time axis with three spans — `K`
demonstrations, one query, then `R` zero-input latent ticks — and supervision
applied only at the final tick. A per-tick phase vector arithmetically gates
which sub-map is active; there is no inner `scan` and no Python loop driving the
model, so `ControlFlowPolicy.scan_unroll_limit` never applies and neither `K`
nor `R` is capped by it.

Each episode draws a fresh bijection over the symbol set. No rule identifier
reaches the model as input. The step function writes only to state, never to a
parameter; parameters change only through pp_prop's update over a training
episode as a whole. The projections that write memory are trained by that update
like any other parameter — that credit path, spanning the full episode, is the
thing the example exists to exercise.

The contextual memory is a pair of `brainstate.HiddenState` arrays of shape
`(n_lat, M)`. Its read is an ordinary contraction between hidden states, not an
ETP operator — ETP operators mark parameter-times-input operations — so the
compiler absorbs it into the hidden-to-hidden transition, where pp_prop's
factorized trace handles it approximately. That approximation is the reason no
gradient claim is made, and it is stated rather than assumed away.

## Interventions

Models are trained once per latent depth `R ∈ {0, 1, 2, 4, 8}`, all five drawing
from a single shared mixed binding-count distribution so that depth is the only
difference between them, then **frozen**. Every intervention runs against
frozen models with no retraining:

- **Binding count**, two through eight simultaneous bindings.
- **Context support**, matched pairs whose query-span inputs and targets are
  byte-identical and which differ only in whether the queried symbol's binding
  appears among the demonstrations.
- **Shuffled memory**, a column permutation of the memory factors that preserves
  shape and magnitude, isolating the content of context from its presence.

`R = 0` is the no-iteration control and is always reported.

## Latent geometry

Over held-out episodes, four measurements and no more:

- effective dimensionality (participation ratio) of `H_r`, per iteration;
- step-to-step change `‖H_{r+1} − H_r‖`, per iteration — fixed point, cycle, or
  divergence;
- linear decodability of the query answer from `H_r`, per iteration;
- linear decodability of the query answer from the memory read at the query
  encoding, `A(Bᵀq)`.

Probes are linear, fit on one episode set and scored on a disjoint set, with both
counts printed. Decoding this query's answer is reported separately from decoding
the full rule. Analysis is one-shot NumPy after the run and must remain linear in
episodes and latent width.

The comparison between the last two measurements is the experiment. If memory-only
decodability matches or exceeds decodability from `H_R`, the report states plainly
that the two-state separation added no decodable information at this scale. That
is a result to be reported, not a failure to be suppressed.

## Reporting contract

Print, for each trained depth: accuracy overall and per binding count; the
supported-versus-short contrast on byte-identical queries; intact-versus-shuffled
memory accuracy; the four geometry measurements per iteration; probe split counts;
and the claim-boundary paragraph. One Agg PNG carries accuracy versus depth,
accuracy versus binding count under both context conditions, and the
per-iteration decodability curve.

Configuration and seed are printed such that a run is reproducible from the
report alone.

## Required tests

Co-located, suffix style, one `*_test.py` per module. They cover: the factored
memory read against a dense outer-product read (property test); oracle agreement
verified independently of the generator; rule variation across episodes and no
rule leakage through inputs; byte-identical matched queries with the short
condition omitting and the supported condition containing the queried binding
exactly once; phase masking activating exactly one sub-map per tick; ingestion
leaving every parameter bitwise identical; shuffle preserving shape and
magnitude; participation ratio on inputs of known rank; trajectory norm on a
constructed fixed point and a constructed divergence; probe fit/score sets
provably disjoint; the null-separation line firing on constructed data; and a
`--smoke` entry-point run. Malformed configurations, shapes, and labels must
raise clear `ValueError` exceptions rather than producing misleading output.

No test in this example asserts agreement, or bounded deviation, between the
online gradient estimate and a backpropagation-through-time oracle. Per the
known-limitations rule, such an assertion would require the finite-window oracle
path; a whole-sequence VJP returns BPTT for every algorithm and would pass
vacuously. This example measures accuracy and geometry.

## Feasibility gates

Two empirical questions are settled by a throwaway spike before implementation,
because either one failing makes the sweeps measure nothing:

1. The task must degrade across the binding range — supported-query accuracy at
   least 0.9 at two bindings and at most 0.6 at eight.
2. The latent span must sustain activity — mean firing rate at `r = R` at least
   25 percent of its rate at `r = 0`, at the largest swept depth.

If no configuration meets a gate, implementation stops and the range or the
population constants are revised in this document first.

## Release boundary

Complete when this specification, the implementation, its co-located tests, and
the README catalog and axis-map rows are committed; focused example tests and the
repository's normal example gate pass; and the branch is clean and pushed.
Generated plots and the Task 1 spike are development artifacts, not release
files.
