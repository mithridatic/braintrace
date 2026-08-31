# BRA-86 performance recovery

## Status

Proposed implementation specification. Awaiting approval before implementation.

## Objective

Reduce the wall time of the current default BrainTrace test gate and, where the
same profile identifies it, a production recurrent path. Preserve scientific
semantics, the complete default test selection, assertions, tolerances, and the
bounded JAX cache policy.

The most recent durable baseline in this repository is 2,208 selected tests in
913.42 seconds with six xdist workers on Python 3.14.6, JAX 0.11.1,
brainstate 0.5.4, and pytest-xdist 3.8.0. That baseline predates the current
Example 21 additions and cannot be represented as the current result. The
present execution sandbox has Python 3 only; JAX, NumPy, pytest, and
pytest-xdist are absent, so it cannot produce a valid new runtime baseline.

## 2026-08-27 environment qualification

The execution workspace is on the assigned
`paperclip/BRA-86-speed-up-code-and-tests` branch. A read-only qualification
check found:

- Python 3.12.3 on Linux 6.18.33.2 WSL2 with glibc 2.39;
- 12 logical CPUs from an Intel Core i9-12900H and 23 GiB of visible RAM;
- no importable `jax`, `jaxlib`, `numpy`, `pytest`, `xdist`, `brainstate`, or
  `brainunit` package;
- no repository virtual environment and no available `uv`, `pip`, or `pip3`
  executable.

The qualification commands were:

```console
python3 -c "import importlib.util, platform, sys; ..."
lscpu
free -h
command -v uv
command -v pip
command -v pip3
find . -maxdepth 3 -type f -name python -o -name pytest
```

This environment cannot collect the suite, separate JAX compile from warm
execution, or support a scientifically valid before/after claim. Installing an
unversioned substitute stack would also make the prior Python 3.14/JAX 0.11.1
baseline incomparable. The next measurement run must therefore use the project
testing extra in a fresh, recorded environment.

## Static audit boundary

A source search confirmed that repeated-loop candidates still exist in tests,
examples, oracle helpers, and training drivers. That search is not a profile:
many hits are host-side data construction, expected-reference implementations,
compile-time graph analysis, or intentionally retained legacy oracle paths.
No candidate is selected from the static result alone. In particular, the
existing `compiled_scan=False` oracle default and strict-order tests are not to
be changed merely because they contain Python loops.

## Existing work not to repeat

The current tree already includes the earlier throughput-v2 changes:

- Gate C2 floating-difference validation converts strict 512-element lists to
  arrays and validates aggregates with vector operations.
- Selected tolerance-based finite-window probes use `compiled_scan=True`.
- Expensive stateful Gate C fixtures share an xdist load group.
- JAX executable caches are cleared every 40 tests and glibc arenas are
  trimmed to bound worker memory.

These are part of the baseline, not candidates for a new claimed speedup.

## Measurement plan

Use a fresh qualified CPU environment with the project testing extra and record
the exact Python, JAX, jaxlib, brainstate, pytest, pytest-xdist, CPU, and memory
versions. Run the complete default selection with six workers and
`--dist=loadgroup --durations=50`. Capture separately:

1. collection and import time;
2. cold JAX trace/compile time;
3. warm execution time after explicit blocking;
4. fixture and data setup time;
5. wall time and peak RSS for each worker.

Run the baseline in three fresh processes. Compare medians; report all three
observations and their range. Do not compare a warm candidate with a cold
baseline or include diagnostic tests in only one arm.

## Change selection

Choose one small optimization only after the profile shows that it is material.
The preferred decision order is:

1. Convert a repeated model rollout still driven by Python into the appropriate
   `brainstate.transform.for_loop` or `scan`, retaining explicit carry and
   finite-window boundaries.
2. Reuse an identical compiled transform or immutable fixture only when state
   reset tests prove that model, optimizer, random, and eligibility-trace state
   cannot cross test boundaries.
3. Vectorize host validation or setup loops only when the profile attributes
   material wall time to them and strict schema/type/failure behavior remains
   unchanged.
4. Adjust xdist grouping only from fresh worker contention and RSS evidence.

Reject a candidate that merely moves time between phases, increases median
wall time, causes unbounded worker RSS, or produces less than a 10% improvement
in its targeted path without a measured full-gate benefit.

## Correctness and regression protection

Add a reproducing sibling `*_test.py` test before a bug fix. For a pure
performance change, add or extend co-located semantic tests that prove:

- exact algorithms still match the BPTT oracle element-wise;
- approximate learning-rule claims use the finite-window oracle and retain
  their guaranteed regime;
- compiled and legacy paths agree within the test's existing tolerance;
- ragged final windows, length-one sequences, nested state trees, and seeded
  random state are preserved;
- invalid types, booleans in numeric evidence, NaN/Inf, empty sequences, and
  malformed shapes still fail exactly as before;
- no mutable fixture or compilation reuse leaks state between tests.

Run the focused changed module first, then the relevant oracle cluster. Run the
complete default selection three times only after focused correctness is green.
Run coverage on the changed modules and keep meaningful coverage above 90%.

## Acceptance criteria

- No test is removed, deselected, weakened, or given a looser tolerance.
- No production API, algorithm, evidence schema, or scientific artifact changes.
- The selected hotspot improves by at least 10% in median wall time.
- The complete default gate is no slower in three fresh runs; report median,
  range, pass/deselect counts, and peak worker RSS.
- Cold compile and warm execution results are labeled separately.
- The final report contains commands, environment, before/after measurements,
  variance, correctness checks, edge cases, and tradeoffs.
