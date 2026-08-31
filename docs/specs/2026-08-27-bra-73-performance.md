# BRA-73 code and test performance

## Status

Proposed implementation specification; awaiting approval and a qualified
benchmark runtime.

## Objective

Reduce a measured production or test-suite bottleneck without changing learning
semantics, scientific qualification, assertions, tolerances, or the default test
selection. The implementation target will be the largest reproducible hotspot
that can be improved with a small, reviewable change.

## Evidence available before approval

- This execution workspace is on
  `paperclip/BRA-73-speed-up-code-and-tests`, not `main`.
- The workspace has CPython 3.12.3 at `/usr/bin/python3`, but import discovery
  finds no JAX, brainstate, pytest, or pytest-xdist installation and there is no
  `pytest` executable. Fresh local runtime profiling is therefore unavailable
  in this heartbeat; dependency installation is not authorized and no timing
  claim below comes from this container. This was checked on Linux/WSL2 with a
  12th Gen Intel Core i9-12900H exposed as 12 logical CPUs and 24,609,416 KiB
  total memory using:

  ```bash
  python3 - <<'PY'
  import importlib.util
  for name in ('jax', 'brainstate', 'pytest', 'xdist'):
      print(name, importlib.util.find_spec(name))
  PY
  command -v pytest
  lscpu
  awk '/MemTotal/ {print}' /proc/meminfo
  ```
  Docker is also absent, so the repository's Compose configuration cannot
  supply a pre-provisioned benchmark runtime in this workspace.
- Continuation check (2026-08-27): CPython 3.12.3 remains available, while
  `jax`, `brainstate`, `pytest`, and the `xdist` import module remain missing;
  no local virtual environment, `pytest`, `uv`, or Docker executable was found.
  The previous revision incorrectly used the distribution name
  `pytest_xdist` as an import probe. The installed distribution imports as
  `xdist`; future environment checks must map distribution names to import
  module names before recording dependency availability.
- Repository contents are writable, but this run exposes `.git` metadata as
  read-only. `git status` and branch inspection succeed, while staging the
  specification fails when Git tries to create `.git/index.lock`. The initial
  worktree-health check proved read access only; future recovery checks must
  verify index writability before promising a commit.
- The retained Example 21 throughput result reports 1,972 passing tests in
  321.48 s, 334.91 s, and 335.89 s with six workers. That result is historical
  evidence from the qualified environment, not a baseline for this branch.
- The previously reported 37.5 s Gate C2 validation hotspot is already fixed by
  commit `278ce14`; the current implementation converts strict lists once and
  checks their aggregates with NumPy operations. It is not a candidate for
  duplicate optimization.
- The current test harness clears JAX caches every 40 completed tests. Its
  retained measurements identify 40 as the wall-time optimum among 15, 40,
  100, and disabled. That policy remains unchanged unless a fresh full-gate
  profile disproves it.
- A static AST audit of non-test Python modules found one repeated model driver
  in package test support that is both broadly reused and still uncompiled:
  `online_param_gradients_singlestep_naive` constructs a fresh
  `brainstate.transform.grad` inside a Python step loop. The helper has 20
  executable invocations across ten co-located test modules, including the
  operator, exact/approximate algorithm, axis-golden, and while-support gates.
  Its sibling finite-window helper already demonstrates the safer pattern:
  construct the transformed gradient once, then carry model and eligibility
  state through a `brainstate.transform` loop. This is the first measurement
  candidate, not yet an optimization claim.
  The invocation inventory is reproducible with:

  ```bash
  rg -n '= online_param_gradients_singlestep_naive\(|return online_param_gradients_singlestep_naive\(|online_param_gradients_singlestep_naive\($' \
    braintrace examples -g '*.py'
  ```
  A continuation inventory independently confirmed the 20 executable call
  sites across these ten sibling-test modules:
  `axis_golden_test.py`, `ostl_test.py`, `param_dim_vjp_test.py`,
  `three_factor_test.py`, `while_support_test.py`, `conv_test.py`,
  `lora_test.py`, `sparse_test.py`, `oracle_models_test.py`, and
  `oracle_test.py`. The helper itself still creates a new transformed gradient
  function on every step; no intervening branch change has removed the
  candidate.
- The same audit found a Python epoch loop in `train_synthetic_gradient`, but
  each epoch already contains a compiled `brainstate.transform.for_loop` and
  the outer reset reallocates state. It remains secondary unless runtime
  evidence shows material epoch-boundary retracing or dispatch cost and a
  transform-based rewrite can preserve reset behavior.

## First measurement candidate

Measure `online_param_gradients_singlestep_naive` on the representative
fixtures already used by `three_factor_test.py`, `axis_golden_test.py`, and one
shape-changing negative-control fixture. Separate setup/graph compilation,
first gradient trace/compile, and warm per-step execution. Count traces and
synchronize every timed result.

If it is a material hotspot, first add an opt-in compiled qualification path,
mirroring `chunked_online_param_gradients(compiled_scan=True)`, rather than
silently changing every oracle caller at once. That path will replace the bare
Python model-step loop with `brainstate.transform.scan`, use an explicit
gradient-total carry, and rely on `State` for model and eligibility-trace
evolution. Only evidence-compatible call sites will opt in during the bounded
change; a default flip requires all focused semantic checks and the affected
gate to pass. If a fixture's trace state changes shape on its first update and
therefore cannot be scanned, do not silently fall back to a Python model loop;
record the incompatible fixture and seek a separate approved design.

The legacy path establishes these observable contracts for the compiled-path
regression: graph compilation and eligibility initialization happen exactly
once; model and eligibility state persist between steps; parameter gradients
are accumulated in input order with the same pytree keys, shapes, dtypes, and
units; and zero-length input fails at the existing `inputs[0]` boundary rather
than returning a new sentinel or zero tree. Floating-point reassociation may
change low-order bits, so equivalence uses the repository's scientific
tolerances while separately pinning ordering-sensitive effects.

The three-factor acceptance tests add an observable ordering constraint:
`_capture_symmetric_signals` stages an unordered `jax.experimental.io_callback`
whose host callback appends each step's signal to `_SINK`, then consumes that
list in temporal order. The callback is a staged runtime primitive (not a
trace-time Python side effect), but its order must not be assumed under a
compiled scan without measurement. The focused equivalence check must therefore
assert callback count and per-step order as well as gradient values. If scan
reorders those callbacks, reject this candidate rather than weakening the
three-factor assertions; changing the callback contract is a separate semantic
change.

Before implementation, add a sibling regression in
`braintrace/_testing/oracle_test.py` that compares the proposed driver with the
current helper for exact and approximate algorithms, including gradient leaves,
state ordering, the zero-length failure boundary, and the single-step-vs-BPTT
divergence that the helper exists to expose. The test must fail against an
intentionally order-independent or whole-sequence implementation so it cannot
pass vacuously.

## Measurement protocol

Use one CI-equivalent CPU environment and record Python, JAX, brainstate,
pytest, CPU, logical-core count, and available memory. Keep the worker count and
`BRAINTRACE_TEST_JAX_CACHE_CLEAR_EVERY` fixed while comparing revisions.

1. Run the package and example gates separately with `--durations=50`, once to
   identify candidates and three times in fresh processes for the selected
   before/after comparison. Report collection count, passes, failures, wall
   time, peak resident memory, and median/range.
2. For a model-execution candidate, record setup, first-call trace/compile, and
   warm execution separately. Synchronize device work before stopping each
   timer. Use at least five independent process samples for compile time and at
   least 20 iterations per sample for warm execution.
3. Use representative sequence length, hidden width, batch size, chunk size,
   epoch count, and reset mode from an existing test or example. Do not choose a
   toy shape solely because it exaggerates the proposed speedup.
4. Attribute the hotspot before editing. Acceptable evidence is a pytest
   duration concentration, a Python profile, a JAX trace/compile count, or a
   controlled ablation that isolates duplicated setup, synchronization,
   allocation, or dispatch.

## Implementation decision rule

Implement one bounded change only after the baseline identifies it. Prefer, in
order:

1. replace repeated model execution with the appropriate
   `brainstate.transform` primitive while preserving explicit carry and state;
2. reuse an equal-shape compiled transform or expensive immutable fixture after
   proving every mutable model, optimizer, and eligibility-trace state is reset
   at the existing boundary;
3. remove duplicated allocation, conversion, synchronization, or test setup;
4. adjust test distribution only from fresh worker-runtime and memory evidence.

If no candidate improves its focused median by at least 10% without regression,
record the negative result and make no production change. A test-only change
must also improve the affected gate median by at least 3% or remove at least
10 s from a concentrated hotspot. Noise floors and confidence limits will be
reported rather than rounded into a speedup claim.

## Correctness invariants

- Exact algorithm gradients continue to match their BPTT oracle element-wise.
- Approximate learning-rule properties continue to use the finite-window oracle
  and retain their guaranteed regime and bounded divergence behavior.
- No assertion, tolerance, marker, default selection, cache-safety control, or
  scientific artifact is weakened or removed.
- No repeated model execution is introduced in a bare Python `for` or `while`
  loop. Randomness continues to use `brainstate.random`.
- Public APIs, serialized schemas, ordering, dtype behavior, and failure modes
  remain unchanged unless a separately approved specification says otherwise.

## Tests and verification

For a bug or performance regression, first add a sibling `*_test.py` test that
reproduces it. Add semantic equivalence and edge coverage for empty or minimum
lengths, ragged final chunks, reset enabled and disabled, batch size greater
than one, dtype/unit-carrying values, invalid inputs, and mutable-state
isolation as applicable. Keep changed-line coverage above 90% where practical.

Run the focused regression and benchmark first, then the affected package or
example gate. Run the full default gate only when the changed path is shared
widely enough to warrant it. Report before/after commands, raw samples, median,
range or variance, environment, peak memory, compile and warm timing,
correctness checks, and tradeoffs.

## Approval boundary

Approval authorizes the measurement protocol and one evidence-selected bounded
optimization. A change to algorithm semantics, a broad architecture rewrite, a
new dependency, CI resource spending, or a weaker verification policy requires
fresh approval.

## Current blockers and owners

- Chief of staff: approve or reject this specification before any code or test
  implementation begins.
- Rainbow/runtime owner: provide the assigned worktree with JAX, brainstate,
  pytest, and pytest-xdist (or an equivalent qualified CPU benchmark runtime)
  so the required cold/warm baseline can be recorded.
- Paperclip runtime owner: restore the run-scoped bridge at
  `PAPERCLIP_API_URL`; without it this heartbeat cannot create the required
  approval interaction, delegate the runtime unblock, or verify the issue's
  blocked disposition.
- Workspace runtime owner: expose writable Git index metadata so approved work
  can be staged and committed on the assigned branch. Repository files are
  writable, but both `.git` and `.git/index` fail the workspace writability
  check in this run. The continuation check observed mode `0775` on `.git` and
  `0664` on `.git/index`, but the managed filesystem still reports both paths
  as non-writable; mode bits alone are not a valid writability check in this
  runtime.
