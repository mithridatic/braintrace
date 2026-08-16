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
| `S_t = U_θ(S_{t-1}, D_t)` | Hebbian write `S_t = S_{t-1} + v_t k_tᵀ`, held factored as `(A, B)` over `M` slots. `k_t`, `v_t` are linear projections of the ingestion population's rate whose scaled tick contributions accumulate before the initialized aggregate code projection is read. |
| `H_0 = E_θ(x*, S_K)` | binary subtractive-reset LIF workspace after the query span. The output-only analog pure read `P_0 = A(Bᵀq_next)` includes the final query tick but is not `H_0`. |
| `H_{r+1} = F_θ(H_r, S_K)` | one zero-input tick of the latent population: recurrence plus the memory read |
| `ŷ = G_θ(H_R)` | linear readout over the symbol set for `R ≥ 1`; the no-iteration control uses `G_θ(P_0)`. Every stored `H_0..H_R` remains binary workspace state. |

The linear-attention special case named in Eq (1), `S_t = S_{t-1} + U_θ(D_t)`, is
the case implemented.

## Inputs and execution

One episode is a single contiguous time axis with three spans — `K`
demonstrations, one query, then `R` zero-input latent ticks — and supervision
applied only at the final tick. A per-tick phase vector arithmetically gates
which sub-map is active; there is no inner `scan` and no Python loop driving the
model, so `ControlFlowPolicy.scan_unroll_limit` never applies and neither `K`
nor `R` is capped by it.

During demonstrations only, D3's one-hot phase-local slot address selects the
memory column being written. It is zero in query and latent phases and conveys
demonstration position, not the episode's rule, index, or target.

Each episode draws a fresh bijection over the symbol set. No rule identifier
reaches the model as input. The step function writes only to state, never to a
parameter; parameters change only through pp_prop's update over a training
episode as a whole. The feasibility spike did expose the memory projections to
that update, but its parameter-delta audit measured exactly zero movement in
`W_v`. The passing accuracy gate therefore is not evidence of a learned write
path. The release default uses fixed random memory-write projections, trains the
remaining model parameters with pp_prop, and states plainly in the report that
the write path was fixed. A learned-write arm may be added only as a separately
labelled experiment with a nonzero parameter-delta check.

The mathematical contextual memory is a pair of factors
`A, B ∈ ℝ^{n_lat × M}`. The implementation stores their transposed, slot-major
logical views with shape `(batch, M, n_lat)` rather than allocating two separate
`HiddenState` objects. Its abstract read is an ordinary contraction between
hidden states, not an ETP operator — ETP operators mark parameter-times-input
operations — so the compiler absorbs it into the hidden-to-hidden transition,
where pp_prop's factorized trace handles it approximately. That approximation is
the reason no gradient claim is made, and it is stated rather than assumed away.

For batched execution, the two slot-major factor views and the binary spike
workspace occupy one row-major `HiddenState` of physical shape
`(batch * (2M + 1), n_lat)`. The
latent voltage and pure query encoding are separate `HiddenState` objects with
that same compiler-aligned physical shape; only their workspace row is active.
Their logical views have shape `(batch, n_lat)`. Projection inputs preserve the
grouped row axis as their leading native-batch axis, so BrainTrace dispatches to
`etp_mm` without flattening rows into features or wrapping the model in `vmap`.
The per-tick key, value, and query projections accumulate linearly. Applying a
ReLU per tick would not preserve the intended aggregate code projection because
`sum_t ReLU(W r_t) != ReLU(W sum_t r_t)`; the initialized aggregate projections
are already nonnegative and row-normalized by construction.

## Measured configuration

The throwaway spike selected `C = 10` symbols encoded by 24-wide Bernoulli codes
with activation probability `0.25` per tick and `codebook_seed = 313320`. Its
attempt-zero codebook had realized rate `0.248958333`, 10/10 unique flattened
symbols, augmented design rank 10, and minimum pairwise Hamming distance 29/96.
Production uses that same fixed BrainState draw rather than an equal-weight
surrogate. Each demonstration uses `T_d = 4` ticks with key and value presented
in parallel. The contextual memory has `M = 8` slots, and both the recommended
latent width and key/value projection width are `32`.

The latent population uses a `160 ms` membrane constant and recurrent spectral
radius `0.9`. The feasibility spike's largest-depth firing-rate retention was
`0.751055`. The corrected production subtractive-reset LIF measured
`r0 = 0.353027`, `r8 = 0.279785`, retention `0.792531`, and achieved recurrent
spectral radius `0.9000002`, above the required `0.25`. This retention required
a coupled latent update: recurrent drive and the contextual-memory read remain
active together at every latent tick.

The CLI defaults to `--device gpu` and fails closed when no GPU is visible. Full
runs use the repository's CUDA-enabled Docker environment operationally, but the
CLI enforces the requested JAX platform rather than detecting the container.
The initial XLA compilation is expected to be slow, while the larger
neuron-and-synapse sweeps benefit from GPU execution. Fast tests explicitly
select CPU so their backend does not depend on local accelerator visibility. At
the native default `(batch=4, M=8, n_lat=32)`, the coupled hidden group has three
`(68, 32)` members and materializes `42,614,784` float32 Jacobian elements
(`170,459,136` bytes). The model explicitly raises the compiler ceiling to
`1 << 26` elements for this supported shape.

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

Over held-out episodes, four primary measurements:

- effective dimensionality (participation ratio) of `H_r`, per iteration;
- step-to-step change `‖H_{r+1} − H_r‖`, per iteration — fixed point, cycle, or
  divergence;
- linear decodability of the query answer from `H_r`, per iteration;
- linear decodability of the query answer from the memory read at the query
  encoding, `A(Bᵀq)`.

Here `H_0` is the recorded binary LIF workspace after the query and `H_1`
onward are the recorded binary LIF latent states. The separate analog memory
read used for `R = 0` query-terminal logits is never substituted for `H_0` in
trajectory, firing-rate, or geometry measurements.

Participation ratio avoids a latent-width-squared covariance. Through width 16,
it uses a complete 16-row Walsh/Rademacher design and is exact. Above width 16,
it uses 16 fixed SplitMix64-derived Rademacher probes as a deterministic
Hutchinson estimate with a nontrivial nullspace. The structured geometry result
records the method, probe count, exact-width threshold, and limitation.

Probes are linear, fit on one episode set and scored on a disjoint set, with both
counts printed. Decoding this query's answer is reported separately from decoding
the full rule. Raw-memory-factor probes are also labelled as secondary rather
than being substituted for the primary query-conditioned memory read. Ridge
probes use a residual-controlled matrix-free conjugate-gradient solve with a
dimension-scaled iteration limit capped at 2,048. Analysis is one-shot NumPy
after the run; its storage and each solver iteration remain linear in episodes
and latent width without a dense width-square factorization.

The comparison between the last two measurements is the experiment. If memory-only
decodability matches or exceeds decodability from `H_R`, the report states plainly
that the two-state separation added no decodable information at this scale. That
is a result to be reported, not a failure to be suppressed.

## Reporting contract

Print, for each trained depth: accuracy overall and per binding count; the
supported-versus-short contrast on byte-identical queries; intact-versus-shuffled
memory accuracy; the four primary geometry measurements per iteration; the
secondary full-rule and raw-factor diagnostics; probe split counts; and the
claim-boundary paragraph. One Agg PNG carries accuracy versus depth, accuracy
versus binding count under both context conditions, and the per-iteration
decodability curve.

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

The throwaway spike settled the two initial sizing questions before
implementation; the corrected production model is requalified separately below:

1. The task must degrade across the binding range — supported-query accuracy at
   least 0.9 at two bindings and at most 0.6 at eight.
2. The latent span must sustain activity — mean firing rate at `r = R` at least
   25 percent of its rate at `r = 0`, at the largest swept depth.

For the untrained fixed-write mechanism, supported-query accuracy at binding
counts `K = 2..8` was `[0.989532, 0.622009, 0.391968, 0.280212, 0.217133,
0.169830, 0.142273]`. The curve is monotone and clears both endpoint gates. The
compiled model's brief CPU training check measured `0.9060` at `K = 2` and
`0.5015` at `K = 8`, also clearing the accuracy gate. These numbers belong to
the feasibility skeleton. The corrected production implementation is a
subtractive-reset LIF rather than that spike skeleton and does not inherit those
scores. The grouped row-major state and native batched ETP layout compile as the
supported shape.

**Corrected-production baseline requalification — failed.** On an RTX 3080 Ti
with `width=32`, `batch=4`, `M=8`, `R=8`, four symbol ticks, eight terminal
pp_prop updates, and 512 fresh supported evaluation episodes per endpoint,
`K = 2` was `0.08984375` both before and after training, failing the `0.9` gate;
`K = 8` was `0.099609375` both before and after training, clearing the `0.6`
upper gate. The model predicted
class 9 for every evaluation episode and its terminal workspace saturated near
`0.95`. `W_k` and `W_v` had exactly zero delta as required by fixed-random mode;
`W_f` and `W_o` had L2 deltas `0.0008848` and `0.0007841`, respectively, with
no compiler warnings. This baseline does not qualify release.

**Corrected-production `R = 0` requalification — passed.** A single model on an
NVIDIA GeForce RTX 3080 Ti Laptop GPU (`cuda:0`, JAX `gpu` backend), using
Bernoulli `codebook_seed=313320`, `projection_seed=210848`, recurrent
`seed=2108`, `width=32`, `batch=4`, and `M=8`, received eight terminal pp_prop
updates. On 512 fresh supported held-out episodes per endpoint, `K = 2`
accuracy was `0.994140625` both before and after training, and `K = 8` accuracy
was `0.1640625` both before and after training. The io-factorized/coupled
compile completed with no warning or error diagnostics. Fixed `W_k` and `W_v`
had exactly zero delta; `W_f` also had zero delta at zero latent depth, while
`W_o` moved by L2 delta `0.000603494`.

For `R = 0`, query-terminal logits decode the analog pure contextual read
`A(Bᵀq_next)` after the final query tick. That output is not the workspace
`H_0`: the stored workspace and all `H_0..H_R` geometry remain binary
subtractive-reset LIF states. With the same trained parameters and one actual
LIF latent tick, terminal `K = 2` and `K = 8` accuracies fell to `0.203125` and
`0.103516`, respectively. Latent iteration therefore degraded this endpoint
result, and the example does not claim that iteration helped.

The accuracy result is qualified by the zero `W_v` movement described above:
the fixed-write fallback is the recorded outcome, and neither the spike nor the
example claims that pp_prop learned the memory write. The binding sweep remains
two through eight.

## Release boundary

Complete when this specification, the implementation, its co-located tests, and
the README catalog and axis-map rows are committed; focused example tests and the
repository's normal example gate pass; and the branch is clean and pushed.
The fresh corrected-production `R = 0` run clears both accuracy endpoints; its
one-tick LIF diagnostic remains a disclosed degradation rather than a passing
latent-depth claim. Focused and normal example gates still belong to the release
qualification task. Generated plots, requalification outputs, and the Task 1
spike are development artifacts, not release files.

Release verification completed with `199` focused Example 21 tests passing in
`113.94 s`. Changed production coverage was `94%` overall: task `98%`, model
`93%`, analysis `94%`, and entry point `93%`. The repository's unmodified normal
example gate, `python -m pytest examples/ -n auto --durations=15`, completed with
`574` passed, `5` skipped, and `19` existing compiler/decomposition warnings in
`147.03 s`. The root-level gate requires `examples/pp_prop` on pytest's import
path because older co-located tests use bare sibling imports; that path is now
declared in `pyproject.toml` so the checked command and CI command are identical.
