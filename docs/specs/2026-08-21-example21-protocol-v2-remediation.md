# Example 21 protocol-v2 remediation

Status: approved implementation specification
Date: 2026-08-21
Branch: `fix/example21-paper-audit`
Baseline: `bde13ba5a16a03fcdded6c04b34913bfeb8137d6`

## Claim boundary

Example 21 targets the public latent-workspace interface and evaluation shape
described by arXiv 2608.09888. The paper does not disclose enough of its model,
training recipe, state update, or compute accounting to support an exact
reproduction claim. Protocol v2 therefore makes a narrower claim: BrainTrace
implements a reproducible, equal-effort ARC-AGI-1 latent-reasoning experiment
with strict exact scoring, explicit causal gates, finite-window eligibility
trace checks, and fail-closed evidence.

The source default remains 4,194,304 recurrent edges. The preregistered
4,096-edge run is a reduced-scale diagnostic and cannot satisfy the full-scale
qualification check. A structurally valid zero-exact-score result is retained;
accuracy is evidence, not a reason to change this preregistered protocol.

## Phase gates and arm streams

Protocol v2 defines model work with a JAX-compatible `StepGates` pytree rather
than inferring phase from event contents. Its boolean, batch-shaped leaves are
`advance_physics`, `latent_update`, `decode_row`, `answer_feedback`, and
`recurrent_enabled`. Every production invocation supplies these gates.

An `ArmStream` owns events, gates, named boundaries, and metadata and validates
their shared leading length, batch shape, boundary order, and phase invariants.

| Phase | physics | latent | decoder | feedback | memory write |
|---|---:|---:|---:|---:|---:|
| demonstration or query row | 1 | 0 | 0 | 0 | side-presence gate |
| matched blank row | 1 | 0 | 0 | 0 | 0 |
| recurrent reasoning tick | 1 | 1 | 0 | 0 | 0 |
| decoder row | 0 | 0 | 1 | 0 | 0 |
| padding | 0 | 0 | 0 | 0 | 0 |

Unequal-height demonstrations write memory whenever either input or output side
is present. Missing-side and fully blank rows are never reinterpreted as latent
ticks. The no-context arm zeros demonstration content while preserving physical
timing and the query boundary; all pre-query latent, decoder, write, and feedback
gates are off. Query and post-query schedules exactly match the intact arm.

## Equal recurrent effort and decoding

Effort means recurrent reasoning ticks only and is evaluated at 0, 30, and 60.
Each checkpoint receives the same fixed 30-row decoder sweep. Evaluation order
is context, R0 decode, 30 reasoning ticks, R30 decode, 30 reasoning ticks, and
R60 decode. The decoder consumes the frozen query state, query grid/shape, and
an explicit row index. It cannot mutate physical state `H`, associative memory
`S`, or feed predicted or target rows back into the primary model.

`latent_row_decode` is the protocol-v2 decoder. `row_refinement` remains only
for schema-1 replay and comparative diagnostics. `attention_residual` is invalid
with `latent_row_decode` because it bypasses the intended latent-binding path.

Enabled controls are:

- `state_hold`: identical schedule with physical and latent updates frozen;
  R30 and R60 must equal R0.
- `recurrent_lesion`: identical ticks, memory, and decoder with recurrent
  synapses disabled.
- `feedback_disabled`: legacy diagnostic only; protocol v2 has no answer
  feedback to lesion.

## Learned memory update

`memory_coding="learned_update"` builds a deterministic feature vector in this
order: demonstration identity; input/output side-valid bits; normalized row
position; input/output height and width; flattened input/output validity masks;
and flattened input/output colors. Event-valid and phase indicators are
excluded. The configuration and report record the feature order, indices,
width, projection shape, initialization seed, and projection hash.

`brainstate.random` initializes a bias-free linear projection from the feature
vector to `memory_width ** 2`, scaled by `1 / sqrt(memory_width)`. It is routed
through a registered ETP operation. The update is

```text
S_t = decay * S_(t-1)
      + write_gate * softcap(U_theta(D_t)) * memory_write_scale
```

One-sided rows must create finite, nonzero, distinguishable writes. Retrieval
keys use the same trace-compatible learned path. Historical `frozen`,
`learned_keys`, and `learned_write` modes remain readable but are not protocol-v2
qualification paths.

## Training and trace semantics

Training samples efforts 0, 30, and 60 uniformly and always supervises 30
decoder rows. Each episode's decoder-cell weights sum to one; a batch of size
`B` contributes exactly `1/B` per episode regardless of output shape. There is
one learner invocation per model tick. The objective is the weighted sum of
per-example loss and only the active-tick mask is passed to the trace engine.
A batched gradient must equal the arithmetic mean of independently computed
episode gradients.

`compile_etrace` is the canonical compiler entry point. `compile_pp_prop`
remains a source-compatible wrapper. Documentation describes D-RTRL here as
"per-parameter trace factorization with diagonal hidden recurrence". Exactness
is claimed only for parameter groups and finite-window regimes that match an
oracle element-wise.

Repeated execution is expressed with `brainstate.transform.jit`, `for_loop`,
`scan`, or checkpointed variants as appropriate. Generative randomness uses
`brainstate.random` throughout source, tests, and examples.

## Candidate policy

`decode_candidates(max_candidates=1|2)` computes exact global factorized
ranking with stable log-softmax terms:

```text
log p(height) + log p(width) + sum(included cells, log p(color))
```

Two-dimensional prefix sums provide all 900 shape totals. Candidate 1 is the
best grid over every height/width. Candidate 2 is the better of the best grid
at the second-ranked shape and the best one-cell second-color substitution in
the top shape. Ties resolve by score, shape before cell flip, lower height,
lower width, row-major cell, then lower color. Only the latest completed effort
is submitted. The recorded policy is
`latest_checkpoint_factorized_global_top2_v2`; strict ARC exact match is
unchanged.

## Evidence schema and qualification

Reports, checkpoints, and generated evidence use schema 2. Readers accept
schema 1 without mutation, but versions cannot be mixed or silently upgraded.
Every qualification check is represented as:

```json
{"status": "passed|failed|not_run", "required": true, "reason": "..."}
```

Compatibility booleans are true only for `passed`. A required `failed` check
fails qualification; otherwise a required `not_run` makes it incomplete and
nonqualifying. Disabled controls are `not_run`, incomplete, and false.

Reports include protocol/schema versions; effort, decoder, and gate counts;
output completeness; decoder pre/post `H` and `S` hashes; external-input and
answer-feedback norms; no-context pre-query gates; control execution; candidate
policy; live carrier-consumer routing; parameter movement and trace evidence;
source revision and dirty state; declared-revision mismatches; configuration
hash; image digest; and artifact checksums/sizes.

GPU accounting is fail-closed. It retains the maximum valid allocator,
device-wide, and process readings across samples; requires physical capacity and
a valid device-wide peak; treats process data as supporting evidence; records
transient errors without discarding a later conservative bound; and fails if no
device-wide bound exists or peak usage exceeds 85 percent.

## Immutable historical evidence

The six tracked revision-`353bd46` bundles below are schema-1 historical
evidence. Protocol v2 indexes and labels them but never rewrites their contents:

- `var/example21-default-20260821`
- `var/example21-ei-dale-010-20260821`
- `var/example21-ei-dale-020-20260821`
- `var/example21-ei-dale-070-20260821`
- `var/example21-ei-dale-090-20260821`
- `var/example21-ei-dale-20260821`

Their 24 baseline SHA-256 digests are recorded by the protocol-v2 evidence
index test, which fails on any byte change.

## Promotion gates

Defaults change to `latent_row_decode`, `learned_update`, and enabled evaluation
controls only after all of the following pass:

1. Exhaustive tiny-distribution top-two enumeration, including shape-versus-
   cell-flip and deterministic tie counterexamples.
2. Effort-zero completeness, decoder state immutability, zero feedback, no
   pre-query latent work, matched no-context timing, one-sided memory writes,
   and required-control execution.
3. Trace rungs `64/2`, `128/8`, and `256/16`: memory-group D-RTRL element-wise
   error at most `1e-4`; pp-prop baseline and learned-pairing cosine at least
   `0.90`; norm ratio in `[0.5, 2.0]`.
4. Three deterministic unequal-height binding seeds with adaptation and copy
   residuals disabled: protocol-v2 exact pass@2 at least 90 percent per seed,
   at least 20 percentage points above legacy row refinement, and shuffled
   context at least 25 percentage points below intact.

The complete 19-module numerical/oracle suite runs with six load-grouped
workers. All BrainTrace and example tests follow. Changed critical paths and
the repository overall require meaningful line and branch coverage above 90
percent, plus Ruff, typing, doctests, diff checks, wheel/sdist builds, artifact
exclusion, Docker help/path checks, and test-collection parity.

## Repository conventions

All tests are co-located as `*_test.py`; tracked `tests/` directories and
`test_*.py` names are forbidden. Golden data lives beside the owning test or in
a sibling `_testdata` directory. Pytest, CI, imports, packaging, and collection
metadata must preserve behavior during relocation. A convention check rejects
test-layout violations and direct generative `jax.random`. Public APIs use
NumPy-style docstrings with doctest-compatible `Examples` blocks. Docker
documentation uses `/opt/braintrace`, and every documented in-image path is
validated against the image definition.

## Finding-to-test-to-invariant remediation ledger

| Finding / prior mistake | Regression test | Permanent invariant |
|---|---|---|
| Unequal-height missing rows were treated as latent work | one-sided demonstration stream and memory-write tests | either-side presence writes; blank rows never reason |
| Efforts had different decoder work and effort zero was incomplete | equal decoder-count and R0-completeness tests | effort counts recurrent ticks only; 30 decoder rows each |
| Event contents inferred control flow | adversarial event-content gate test | `StepGates` is the only production phase authority |
| No-context changed timing or allowed pre-query work | paired arm schedule test | physical/query boundaries match; pre-query semantic gates off |
| Decoder feedback contaminated later checkpoints | state hash and feedback-norm tests | decoder sweeps preserve `H`/`S`; feedback is zero |
| Candidate two was only locally or checkpoint ranked | exhaustive enumerator and tie tests | exact global top two from latest checkpoint |
| Union masks overweighted large outputs | unequal-shape gradient equivalence test | each episode totals one and contributes `1/B` |
| One-sided rows produced zero/aliased memory | one-sided nonzero/distinct write test | learned update includes side bits, masks, shapes, colors |
| Disabled controls passed vacuously | disabled-control qualification test | disabled means required `not_run` and nonqualifying |
| Docker examples referenced the wrong root | documented-path existence test | all in-image paths resolve under `/opt/braintrace` |
| A transient monitor error erased later valid evidence | monitor recovery/max test | retain maxima and errors independently |
| D-RTRL wording overstated exactness | documentation/convention test | diagonal-recurrence factorization wording and finite-window claims |
| Historical evidence was at risk of silent rewrite | 24-file digest test | schema-1 bundles remain byte-for-byte immutable |

## Preregistered diagnostic

After a clean promoted implementation commit, run only:

```text
python /opt/braintrace/examples/pp_prop/21-arc-agi-latent-reasoning.py --recurrent-edges 4096
```

The compatibility entry point may forward from the historical filename. No
other semantic override is allowed. The run must exit successfully; cover all
400 tasks and 419 queries; complete efforts 0/30/60 and required controls;
preserve decoder state; show zero primary external and feedback norms; show no
pre-query latent/decode/feedback and zero no-context memory; satisfy state-hold;
use the v2 candidate policy; record finite movement/traces and complete
provenance; and stay below the VRAM ceiling. `actual_full_scale` and overall
qualification remain false because 4,096 edges is below the source default.

Generated schema-2 evidence is committed separately so its manifest names the
clean implementation parent. The identical command is rerun only for invalid
execution, provenance, control, or structural evidence, never to tune a valid
negative score.
