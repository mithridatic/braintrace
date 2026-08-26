# Gate C evaluator compile reuse

## Status

Rejected after matched measurement on `paperclip/bra-36-speedups`; no
implementation retained.

## Objective

Reduce Gate C cold test and formal-run time by reusing evaluation executables
across arms with identical static evaluation geometry, without changing
training, evaluation streams, model parameters, state-reset boundaries,
scientific thresholds, report schemas, or assertions.

## Baseline

The baseline is commit `0a101278c7839cd25f59c303d1510bc75fdf3f79` on
Windows, Python 3.14.6, JAX 0.11.0 CPU, brainstate 0.5.3, and pytest 9.1.1.

- Collection of `latent_workspace_ablation_gate_test.py`: 22.06 s.
- Bounded single-process module run: 96 passed in 237.20 s before a clean
  interrupt at 11%.
- Slow fixture setup: reduced Gate B 71.34 s and reduced Gate A 40.36 s.
- Isolated cProfile Gate B node: 1 passed in 99.79 s, including 85.84 s setup.
- Within the profile, `_evaluate_arm` consumed 21.73 s across five arms,
  `LatentWorkspaceModel.__init__` consumed 15.21 s across the five training and
  five evaluation models, and the five compiled training calls consumed
  16.81 s. JAX backend compilation consumed 23.71 s across the process; Numba
  compilation consumed 22.50 s. cProfile and fresh-process import overhead
  make these cumulative figures unsuitable for addition, but they identify
  evaluation construction and compilation as a material independent target.

The current path constructs and compiles a new validation-batch model inside
every `_evaluate_arm` call. The `full`, `terminal_only`, and `frozen_write`
arms have identical evaluation model configuration and `memory_read_policy`;
only their trained parameter values differ. `query_only` and `legacy` retain
distinct static evaluation programs.

## Scope

1. Extract the validation model and compiled three-stream evaluator into a
   private reusable evaluator object or factory.
2. Before each arm evaluation, copy that arm's trained parameters into the
   compatible validation model. Reset all mutable biological and workspace
   state before every stream exactly as today.
3. Reuse one evaluator only for arms whose complete `ModelConfig`, validation
   batch geometry, regime, and `memory_read_policy` are equal. In the current
   five-arm matrix this permits `full`, `terminal_only`, and `frozen_write` to
   share; `query_only` and `legacy` remain separate.
4. Use the same grouped evaluator path in the reduced module fixtures and the
   formal sequential arm runner. Keep `_evaluate_arm` as a compatibility
   wrapper that creates a one-shot evaluator when no reusable evaluator is
   supplied.
5. Do not cache across processes, regimes, configurations, or incompatible
   arm policies. Do not retain training models or trainers longer than the
   current formal runner does.

## Correctness invariants

- Each evaluation consumes exactly the trained parameters of its own arm.
- The intact, shuffled, and no-context stream bytes and order are unchanged.
- Model state is reset before each stream, and no state or parameters leak
  from a previously evaluated arm.
- Evaluation reports and strict canonical JSON are byte-identical to the
  one-shot path for all five arms in both reduced Gate A and Gate B regimes.
- No learning-rule code, optimizer state, training schedule, tolerance,
  qualification threshold, public API, or report schema changes.
- Repeated model execution continues to use `brainstate.transform.for_loop`
  inside a compiled driver; no Python loop is introduced around timesteps.

## Tests

Add co-located regressions before the implementation change that:

1. compare reusable and one-shot evaluation reports for every arm and both
   reduced regimes;
2. evaluate two compatible arms in both orders and prove byte-identical
   reports, catching parameter or state leakage;
3. reject reuse for a different model configuration, regime, validation batch
   shape, or memory-read policy;
4. retain all existing reduced real pp-prop arm assertions and formal report
   validators.

Run the focused new regressions and existing reduced Gate A/Gate B fixture
consumers. Measure each command in at least three fresh processes after one
untimed environment warm-up. Report median and range separately for collection,
fixture setup/cold compilation, and test-body execution. Run changed-file
coverage above 90% and syntax/style checks without weakening assertions.

## Acceptance

- All correctness invariants and focused regressions pass.
- Median reduced Gate B fixture setup improves by at least 10% over the
  71.34 s baseline, with no regression greater than 5% in reduced Gate A.
- The profile shows fewer evaluation-model constructions and fewer JAX
  compilation cache misses for the compatible three-arm group.
- If the 10% target is not met, retain only changes that independently simplify
  the evaluator lifecycle without increasing runtime or memory; otherwise
  revert the implementation and record the negative result.

## Risks and tradeoffs

The main risk is mutable state or parameters leaking between arms. Exact
one-shot equivalence and order-reversal tests are therefore release-blocking.
Keeping three evaluator programs alive instead of compiling five may modestly
increase the lifetime of evaluation state while reducing executable count;
peak RSS must be observed during the focused benchmark. No scientific claim is
made from this performance work.

## Measurement result

The candidate reused one evaluator for `full`, `terminal_only`, and
`frozen_write`, retained distinct evaluators for `query_only` and `legacy`, and
passed byte-exact reusable-versus-one-shot and compatibility regressions. The
candidate was then rejected because matched fresh-process timing did not show a
material improvement.

Command (baseline and candidate):
`python -m pytest examples/pp_prop/latent_workspace_ablation_gate_test.py -k
"reduced_gate_b_runs_all_five_real_pp_prop_arms" -q --durations=5`. Candidate
runs also selected the two new lifecycle regressions; setup duration below is
the shared reduced Gate B fixture and excludes their 5.68--5.95 s one-shot
equivalence call.

| Revision | Fixture setup runs (s) | Median (s) | Range (s) |
| --- | --- | ---: | ---: |
| Baseline `ac7e2b8` | 64.34, 66.94, 64.75 | 64.75 | 2.60 |
| Candidate | 64.90, 68.29, 59.42 | 64.90 | 8.87 |

The candidate median was 0.15 s (0.2%) slower. This is within observed host and
JAX compilation variance and misses the preregistered 10% improvement gate.
The code and test-fixture changes were therefore reverted. The earlier 71.34 s
profile remains useful bottleneck evidence but was not a stable comparison
baseline. A future attempt should first add lower-variance compile-cache-miss
instrumentation or isolate evaluator construction from arm training.
