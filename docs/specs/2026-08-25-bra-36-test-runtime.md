# BRA-36 test-runtime profiling and bounded optimization

Status: profiling approved by independent review on 2026-08-25

The reviewer clarified that trainer construction must be timed separately from
JAX tracing/compilation. Cold training and evaluation are therefore recorded
per arm with an explicit synchronization at every timed boundary; warm timing,
when collected, is labeled separately. Approval applies to profiling only. A
measured optimization candidate still requires this specification to record
its target and semantic invariants before product or test behavior changes.

## Revision 2: compile-event instrumentation before candidate selection

The first implementation candidate (Gate C evaluator reuse) was rejected after
matched measurement: its 64.90 s median was 0.15 s (0.2%) slower than the
64.75 s baseline. The candidate was reverted and its negative result is
recorded in `2026-08-25-gate-c-evaluator-reuse.md`.

A fresh-process Gate A run on 2026-08-25 used:

`python -m pytest examples/pp_prop/latent_workspace_ablation_gate_test.py -k
"reduced_gate_a_runs_real_full_and_legacy_pp_prop_arms" -q --durations=8`

It completed with 1 passed and 631 deselected in 51.02 s. Pytest attributed
36.64 s to module-fixture setup, 0.28 s to teardown, and less than 0.005 s to
the selected test body. This confirms that assertion execution and ordinary
pytest harness work are not useful optimization targets for this path.

Before a second implementation candidate, add temporary benchmark-only
instrumentation (kept outside product behavior) around the fixture's four
phases: data regeneration/encoding, model construction, trainer/compiler
construction, and first synchronized train/evaluate execution. For the JAX
phases, capture trace/compile counts as well as elapsed time. Run Gate A and
Gate B in three fresh processes each, after one unmeasured environment warm-up.

Select the second candidate only if one phase accounts for at least 20% of
fixture setup in both the median and every measured run, and the candidate has
a mechanism expected to remove work from that phase (not merely move it into
an unmeasured fixture). Update this specification with the measured target and
its exact semantic invariants, then request approval before changing product
or test code. The acceptance gate remains a material median improvement across
three matched fresh-process runs, with no weakened assertions, tolerances,
coverage, state isolation, or finite-window oracle checks.

## Revision 3: measured phase breakdown

Independent review approved Revision 2 and required synchronized per-arm cold
train/evaluation boundaries. The profiling harness is
`examples/pp_prop/latent_workspace_ablation_gate_profile.py`. Each measured
sample ran in a fresh process after one unmeasured warm-up, using
`PYTHONPATH=.;examples/pp_prop python -m
examples.pp_prop.latent_workspace_ablation_gate_profile <gate>`.

| Gate | Total runs (s) | Median (s) | Model construction median (range) | Cold synchronized execution median |
| --- | --- | ---: | ---: | ---: |
| A | 34.45, 36.52, 34.91 | 34.91 | 14.52 (14.05--15.64), 41.6% | 16.20, 46.4% |
| B | 77.20, 79.24, 79.33 | 79.24 | 15.05 (14.42--16.35), 19.0% | 58.75, 74.1% |

The percentages use each phase median divided by the total median. Model
construction exceeded 20% in every Gate A run, but only one of three Gate B
runs (18.7%, 19.0%, and 20.6%), so it fails the preregistered cross-gate target
rule. Aggregate synchronized cold execution clears 20% in both gates, but the
current measurement does not isolate a removable compilation unit. The prior
compatible-arm evaluator reuse candidate already failed matched timing, so
that mechanism must not be retried without new evidence.

JAX monitoring recorded 97 compilation-cache requests in a subsequent Gate A
validation run and 104 in Gate B. This public event does not distinguish cache
hits from misses, so these counts are diagnostic only. No second optimization
candidate is authorized by this revision. The next bounded investigation is
to attribute compiler requests/misses to individual train and evaluation
boundaries, then propose only a unit whose redundant work can actually be
removed. Product and scientific test behavior remain unchanged.

## Revision 4: per-boundary compiler-event attribution

The profiling harness now snapshots JAX's public compiler/cache event counter at
every synchronized phase boundary. One fresh Gate A and Gate B diagnostic run
on 2026-08-25 produced the following attribution:

| Boundary | Gate A requests / seconds | Gate B requests / seconds |
| --- | ---: | ---: |
| Data | 10 / 2.62 | 8 / 1.95 |
| All model construction | 35 / 12.02 | 35 / 12.18 |
| All trainer construction | 15 / 1.12 | 15 / 1.55 |
| First (`full`) cold train | 1 / 4.95 | 1 / 10.56 |
| First (`full`) cold evaluation | 22 / 3.35 | 22 / 4.44 |
| Each later cold train | 1 / 3.21 | 1 / 3.06--4.76 |
| Each later cold evaluation | 2 / 1.51 | 2 / 2.85--3.43 |

The equal 35-request model-construction and 22-request first-evaluation counts
for a two-arm and five-arm fixture show that JAX already reuses those compiled
units within a process. This explains why evaluator-object reuse did not reduce
runtime: it changed Python object lifetime but did not eliminate the dominant
compiler requests. The public event still reports requests rather than proven
misses, so this is diagnostic attribution, not a speedup claim.

No safe removable unit is demonstrated in the Gate A/Gate B fixture after this
attribution. Reattempting evaluator reuse, merging scientific arms, weakening
fresh state, or moving the same cold work outside the timed boundary is
rejected. The next proposed scope is to profile the broader suite for repeated
same-shape construction across independent tests and select a new target only
where a fresh-process benchmark proves duplicated work. This scope requires a
new approval because the current approval is bounded to the Gate A/Gate B
investigation.

## Objective

Reduce developer and CI feedback time without changing algorithm semantics,
scientific assertions, tolerances, coverage, or the set of tests executed.

## Baseline environment and method

- Worktree: `paperclip/bra-36-speedups` at `0a10127`
- Host: Windows, CPU backend
- Python: 3.14.6
- pytest: 9.1.1
- pytest plugins: hypothesis 6.165.2, xdist 3.8.0, cov 7.1.0
- Command:
  `python -m pytest braintrace/_algorithm/_common_test.py -vv --durations=10`
- Result: 65 passed in 16.94 seconds.
- Largest duration: 6.03 seconds in the first array-shape test. Later comparable
  array shape/dtype/reset tests were generally 0.13--0.22 seconds.

A broader serial run of `pytest braintrace --durations=25 -q` reached about 6%
after roughly two minutes and was interrupted. A three-module algorithm sample
likewise had not completed after roughly 90 seconds. These interrupted runs are
diagnostic bounds, not completed benchmarks, and must not be reported as suite
runtime.

## Finding

The completed sample indicates that cold JAX initialization/dispatch is a large
component of a small module's wall time. It does not yet prove that an
application helper is slow. The CI workflow already uses `pytest -n auto` and
bounds each worker's JAX cache, so adding workers or removing cache management
is not an evidence-backed candidate.

## Approved-scope proposal

1. Add a reproducible focused benchmark that labels process startup, test data
   setup, first/cold JAX execution, and repeated/warm execution separately.
2. Use pytest duration output and fixture instrumentation to locate duplicated
   setup in the slowest algorithm test module(s).
3. If duplication is demonstrated, share only immutable inputs or factory
   metadata at module/session scope. Create fresh mutable BrainState state and
   random streams for every test.
4. If repeated model execution is found in product or benchmark code, migrate
   it to the appropriate `brainstate.transform` primitive; do not introduce or
   retain a bare Python model loop.
5. Accept an implementation only when three baseline and three candidate runs
   show a material median wall-time reduction on the same machine. Report cold
   and warm results independently and include range/variance.

## Correctness and regression gates

- Run the changed module and its sibling tests.
- Preserve all existing assertions and tolerances.
- For exact algorithms, retain element-wise BPTT-oracle agreement.
- For approximate learning-rule properties, retain the finite-window oracle
  path rather than substituting a whole-sequence VJP.
- Run coverage for changed modules and keep meaningful coverage above 90% where
  practical.
- Reject the candidate if it only moves cold initialization into an unmeasured
  fixture, changes random inputs, leaks mutable state, or improves a single run
  outside observed variance.

## Deliverables after approval

- Reproducing benchmark/test adjacent to the affected module.
- Small implementation change on `paperclip/bra-36-speedups`.
- Before/after table with commands, environment, medians, ranges, cold/warm
  labels, correctness results, and tradeoffs.
- Committed candidate SHA for independent review.

## Edge cases to test

- Tests remain order-independent and pass when selected individually.
- Parallel xdist workers do not share mutable state or random streams.
- Different shapes/dtypes still trigger the intended JAX compilations.
- Cache clearing remains bounded and does not cause memory growth.
- CPU-only execution and the supported JAX version matrix retain identical
  numerical behavior.
