# BRA-117 code and test performance

## Status

Proposed implementation specification, awaiting Chief of staff approval.
Read-only profiling may proceed before approval; code and test implementation
require approval under the repository working agreement.

## Objective

Reduce developer and CI feedback time without removing tests, weakening
assertions, changing numerical tolerances, or changing learning-algorithm
semantics. Adopt a change only when a reproducible profile identifies the
bottleneck and the candidate improves its median wall time by at least 10%.

## Existing evidence

The 2026-08-19 Example 21 throughput work reduced a 1,953-test, six-worker
baseline from 869.3 seconds to a 334.91-second median after the selection grew
to 1,972 tests. Its focused entry/task lane still missed the 45-second stretch
target at 58.9 and 66.5 seconds. The later 2,208-test selection was measured at
913.42 seconds with six workers before the 2026-08-22 throughput-v2 changes.

The current execution image cannot refresh those measurements. The measured
environment is Python 3.12.3 on a 12-logical-CPU Intel Core i9-12900H with 23
GiB RAM and 8 GiB swap. It has no pytest, pytest-xdist, JAX, brainstate,
brainunit, or project virtual environment. Docker is installed, but access to
its daemon is denied, and network access is restricted. The next profiling
heartbeat must therefore run in the project benchmark environment; historical
numbers are context, not a new performance claim.

This was verified on 2026-08-27 with the following read-only baseline attempt:

```text
python3 --version
python3 -m pytest --version
python3 -c "import jax, brainstate, brainunit, pytest, xdist"
docker info --format '{{.ServerVersion}}'
getconf _NPROCESSORS_ONLN
awk '/MemTotal|SwapTotal/ {print}' /proc/meminfo
lscpu
```

The first command returned Python 3.12.3. Pytest and every listed project
dependency failed to import, Docker reported permission denied for its daemon,
and the hardware commands produced the CPU and memory facts above. Consequently
there is no valid collection, compile, warm-execution, or total-runtime number
from this image.

The previously reported Gate C2 validator hotspot is not a candidate for this
issue: commit `278ce14` already replaced its scalar aggregate checks with NumPy
array operations. Candidate selection must start from new timings on current
HEAD rather than reusing that stale profile.

A continuation recheck on 2026-08-27 found the same environment blocker on the
assigned worktree branch. `/usr/bin/python3` is Python 3.12.3; no Python 3.14,
project virtual environment, pytest executable, or repository package runner
(`uv`, `pixi`, `conda`, `rye`, `hatch`, `pdm`, or `poetry`) is available. The
repository declares the needed packages, but none is installed. No new
performance number or speedup claim was produced.

Correction record: the first draft stated an assumed Python version before it
was measured. It was corrected from the captured `python3 --version` output;
future environment facts must likewise come directly from captured commands.

## Measurement plan

Use Python 3.14, the CI JAX/brainstate dependency set, pytest, and
pytest-xdist. Record exact package versions, CPU, logical core count, available
memory, worker count, and cache-clear cadence with every result.

1. Run collection separately and record its wall time.
2. Run the complete `braintrace/` and `examples/` CI selections in fresh
   processes with `--durations=100`. Use the CI distribution modes and a fixed,
   memory-safe worker count. Record pass/deselect counts, wall time, aggregate
   worker peak RSS, and page/swap activity.
3. Repeat each selected baseline three times and use the median. Do not mix
   cold compilation and warmed execution.
4. For the slowest test or fixture accounting for at least 5% of the selected
   gate, split collection, data setup, JAX trace/compile, first execution, warm
   execution, synchronization, and teardown. Count compilations and synchronize
   device results at timing boundaries.
5. Profile host-only setup with a deterministic profiler. Inspect JAX trace and
   compilation behavior separately; do not infer either from total wall time.

Use the unresolved Example 21 entry/task gate and its expensive stateful
fixtures as the first measured lane because existing measurements show a
concrete target miss. This is a profiling priority, not a predetermined
optimization target. If fresh durations place another test above it, follow
the measured ranking instead.

## Implementation constraints

- Write a reproducing performance regression test or focused benchmark beside
  the affected module before changing behavior.
- For repeated model execution, use `brainstate.transform.for_loop` or
  `brainstate.transform.scan`; use checkpointed variants only when measured
  reverse-mode memory requires them. A one-shot model call may use
  `brainstate.transform.jit`. Never introduce a bare Python model loop.
- Use `brainstate.random` for random generation.
- Preserve exact-algorithm equality with finite-window/BPTT oracles as
  appropriate. Preserve the guaranteed regime and bounded behavior of
  approximate algorithms.
- Keep every current default-gate test, assertion, marker, schema, source pin,
  and tolerance. Fixture reuse must reset model, optimizer, random, hidden, and
  eligibility-trace state at existing test boundaries.
- Prefer one small hotspot fix. Do not tune xdist or cache eviction without
  fresh wall-time and memory evidence.

## Acceptance

The optimized hotspot must improve median wall time by at least 10% over three
fresh-process runs. Report cold compile, warm execution, setup, and total time
separately, plus variance and peak RSS. The relevant complete CI selection must
remain green, meaningful coverage must remain above 90%, and focused scientific
checks must show unchanged results within their existing contracts. Reject the
candidate if it shifts cost elsewhere, increases peak RSS by more than 5%
without an explicit tradeoff approval, or makes the result less maintainable.

## Approval gate

No code or test implementation starts until this specification is approved.
After approval, refresh the baseline in an executable project environment,
select the measured hotspot, add the regression benchmark/test, implement the
smallest qualifying change, and append before-and-after results here. If an
executable benchmark environment remains unavailable, stop without changing
code and record that first-class blocker instead of optimizing from static
inspection.

The named unblock owners and actions are:

- Chief of staff: approve or reject this specification.
- Rainbow: provide the qualified Python 3.14 benchmark environment with the CI
  dependency set and an executable pytest/xdist lane.
- Paperclip runtime owner: restore the local API bridge so the approval request,
  durable issue update, and final disposition can be recorded.
