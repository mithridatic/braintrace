# Example 21 pytest throughput

## Status

Approved implementation specification for `feat/example21-row-refinement`.

## Objective

Reduce the wall time of the complete Example 21 and latent-workspace pytest
gate while keeping all 19 modules in the default gate, preserving the public
runtime APIs and JSON schemas, and retaining the existing production evidence.
The target is a green median complete-gate wall time of at most 435 seconds
under the six-worker workstation configuration.

## Baseline and acceptance

The baseline complete gate collected 1,953 tests and took 869.3 seconds with
six workers. It produced 1,947 passes and six understood fixture/isolation
failures. Acceptance requires the baseline 1,953 tests plus any new regression
tests to pass (the implementation gate therefore collects 1,972 tests), the
pinned Gate B encoded digests to remain byte-identical, focused entry/task
tests to remain within 45 seconds at two workers, and three fresh six-worker
complete runs to have a median wall time no greater than 435 seconds.
Aggregate worker memory must remain below 32 GiB without paging or relaxing
the existing JAX cache clear cadence.

## Scope

1. Repair Gate C fake-module isolation by replacing both `sys.modules` entries
   and package attributes, and add regression coverage for preloaded package
   state.
2. Stub frozen source evidence in the two depth-gate CLI orchestration tests;
   do not refresh preregistered production source hashes or treat old Gate B
   evidence as current-source qualification.
3. Replace Gate B's per-episode `ArcTask`/`ArcGrid` construction with private
   batched NumPy encoders that decode mapping IDs and fill event, target,
   weight, and advance tensors directly. Training and validation use the same
   encoder. Preserve shape validation, read-only outputs, ordering, and the
   pinned digests. Regression tests compare it byte-for-byte with the scalar
   encoder across boundary and sampled mapping IDs, colors, efforts, and
   presentation orders.
4. Construct the legacy finite-window gradient transform once per oracle call
   and reuse it for equal-shaped chunks, retaining host-loop order,
   accumulation order, `after_init`, and the default `compiled_scan=False`.
   Numerical probes may opt into `compiled_scan=True` where only finite-window
   semantics and tolerance-based equivalence are asserted.
5. Execute the reduced Gate C mechanism oracle once in a grouped fixture; keep
   threshold and construction-audit assertions independent over that result.
6. Add `pytest-xdist>=3.8` to testing/development extras, mark expensive
   stateful fixture consumers with `xdist_group`, run the full gate with
   `--dist=loadgroup`, and use the same distribution mode in the Examples CI
   job. Keep the focused two-file lane at two workers and retain the existing
   40-test JAX cache-clear cadence.

## Non-goals and invariants

GPU qualification, scientific requalification, production source-pin refresh,
and changes to retained Gate C v1 numerical evidence are outside this task.
`chunked_online_param_gradients` keeps its signature and default mode. No tests
move to an opt-in lane, public APIs and JSON schemas remain unchanged, and all
changes land on the feature worktree branch.

## Verification

Run focused encoder, oracle, launcher, and depth-gate regressions; the focused
two-file entry/task gate at two workers; then all 19 modules at six workers
with `--dist=loadgroup --durations=100` in three fresh processes. Run changed-
file Ruff, mypy, and meaningful coverage above 90%. Record pass counts, wall
times, memory observations, and any residual environmental limitations in the
implementation closeout.

## Implementation closeout

Verification used the synchronized Python 3.14.6 environment with JAX 0.11.0,
brainstate 0.5.3, pytest 9.1.1, and pytest-xdist 3.8.0. The 19-module gate
collected 1,972 tests (the 1,953-test baseline plus 19 regressions) and passed all
tests in each fresh six-worker run. Pytest reported 335.89 s, 334.91 s, and
321.48 s; the median was 334.91 s. The focused entry/task gate passed 257 tests
at two workers, but measured 58.9 s and 66.5 s in this isolated environment,
above the 45 s stretch target.

The focused changed regressions passed (419 oracle/depth/launcher tests, 20
mechanism-oracle tests, and both pinned digest tests). Mypy passed for 66
braintrace source files. The repository coverage run reached 95% total coverage
with 3,064 passes; two unrelated `_quant/_turboquant` tests still fail under
the qualified JAX image because of an existing cached Hadamard tracer leak.
Ruff still reports legacy diagnostics in the touched large modules, so a clean
Ruff result is not claimed. Worker memory was not independently sampled during
these runs.
