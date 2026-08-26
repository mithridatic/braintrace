# BRA-36 test-runtime profiling and bounded optimization

Status: proposed; implementation requires approval

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
