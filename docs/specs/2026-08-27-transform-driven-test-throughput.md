# Transform-driven recurrent test throughput

Status: completed; transform candidate rejected by the approved performance
gate and removed

Date: 2026-08-27

Issue: BRA-125

Branch: `paperclip/BRA-125-speed-up-code-and-tests`

Inventory source revision: `0a10127`

## Objective

Reduce the runtime of recurrent and online-learning tests without changing the
algorithms, test selection, assertions, numerical tolerances, or scientific
oracles. Repeated model execution must use `brainstate.transform` primitives,
as required by the repository working agreement.

This is a measure-first optimization. The first implementation-stage action is
to benchmark the candidate tests in a dependency-complete environment. If the
focused median does not improve materially, the code change is abandoned and
the negative result is recorded.

## Evidence and measurement limitation

Eight successful daily scheduled Continuous Integration runs on upstream
`main`, from 2026-08-20 through 2026-08-27, took between 15 minutes 46 seconds
([run 33051348334](https://github.com/chaobrain/braintrace/actions/runs/33051348334))
and 17 minutes 58 seconds
([run 32610039372](https://github.com/chaobrain/braintrace/actions/runs/32610039372))
end to end. This is a directional system baseline only: the workflow includes
dependency installation, four JAX test matrix jobs, examples, type checking,
and builds, so it does not attribute time to the candidate tests.

The execution workspace has Python 3.12.3 on a 12-logical-CPU Intel i9-12900H,
but it does not contain JAX, brainstate, pytest, or pytest-xdist. Docker is
installed but the runtime cannot access its daemon. Consequently, no valid
local cold-compile or warm-execution number is available yet. Invented or
cross-environment timings are not acceptable baselines.

Static inspection found 39 syntactic candidates under a broad AST filter for
bare Python loops containing calls whose names resemble a model, cell, readout,
or online learner. Manual attribution confirmed 13 fixed-shape repeated-model
loops in `braintrace/nn/_rnn_test.py` and `braintrace/nn/_readout_test.py`.
Together they drive 179 stateful steps in sequences of 5--50 timesteps. The
broad filter also finds shape sweeps, constructors, metadata access, and
deliberately scalar oracles, so its total is not an optimization target by
itself.

### Confirmed candidate inventory

The exact test nodes and static step counts are:

| Module | Test method | Steps |
| --- | --- | ---: |
| `_rnn_test.py` | `test_valina_rnn_sequential_updates` | 5 |
| `_rnn_test.py` | `test_gru_sequential_updates` | 10 |
| `_rnn_test.py` | `test_mgu_sequential_updates` | 5 |
| `_rnn_test.py` | `test_lstm_sequential_updates` | 15 |
| `_rnn_test.py` | `test_minimal_rnn_sequential_updates` | 7 |
| `_rnn_test.py` | `test_minigru_sequential_updates` | 12 |
| `_rnn_test.py` | `test_minilstm_sequential_updates` | 10 |
| `_rnn_test.py` | `test_cfn_sequential_updates` | 10 |
| `_rnn_test.py` | `test_lru_sequential_updates` | 20 |
| `_rnn_test.py` | `test_sequence_processing` | 50 |
| `_rnn_test.py` | `test_reset_during_sequence` | 10 |
| `_readout_test.py` | `test_leaky_rate_readout_sequential_updates` | 5 |
| `_readout_test.py` | `test_rate_readout_temporal_dynamics` | 20 |

The AST inventory also finds four non-candidates in these modules:
`test_different_batch_sizes`, the two nested loops in
`test_batch_size_consistency`, and
`test_leaky_rate_readout_different_batch_sizes`. They sweep incompatible shapes
or model instances and must remain host-driven. Reproduce this attribution by
parsing both modules, selecting `For`/`While` nodes containing `cell(...)` or
`readout.update(...)`, and then manually excluding shape/model sweeps. The
result is 17 matching loop nodes, of which 13 are fixed-shape sequence drivers.

### Correction record

An initial text-pattern count reported 43 candidate loops. That number mixed
unverified syntax matches with performance targets and was not reproducible
enough to retain. The count above comes from an AST-shaped filter plus manual
attribution. Future profiling notes will label heuristic matches separately
from confirmed repeated-model loops and record the selection method before
using a count to define scope.

On the 2026-08-27 continuation wake, the first-class Paperclip blocker was
reported resolved, but the execution environment was still not benchmark- or
commit-ready. Treating dependency-edge completion as evidence that the local
capabilities had changed was incorrect: control-plane state and execution
environment state are separate facts. Future continuations must re-run the
dependency-import and Git-mount probes before declaring the implementation
gate open.

### Validation record

On 2026-08-27, the inventory was reproduced in the execution workspace with
Python 3.12's standard-library `ast` module. For each `For` or `While` node in
the two focused modules, the audit found the enclosing test and selected loop
bodies containing a call to `cell(...)` or `readout.update(...)`. It reported
14 matching nodes in `_rnn_test.py` and three in `_readout_test.py`. Manual
classification retained 13 fixed-shape sequence drivers and excluded the four
shape/model sweeps listed above. The retained literal and named sequence
lengths sum to 179 steps.

The same workspace reported 12 logical CPUs on a 12th Gen Intel Core
i9-12900H and Python 3.12.3. Dependency probes failed for `pip`, pytest, NumPy,
JAX, brainstate, and brainunit, while `docker version` reached the client but
was denied access to `/var/run/docker.sock`. These results reproduce the
measurement limitation rather than providing a performance baseline.

An attempt to stage this specification then failed because the harness mounted
`.git` read-only. Future runs must check that both the Python dependencies and
Git metadata are writable before beginning benchmark or commit work. The file
remains in the execution workspace but is not committed.

The continuation workspace reproduced both blockers. `/usr/bin/python3` is
Python 3.12.3 but has no `ensurepip`, `pip`, or `pytest`; imports of JAX,
brainstate, brainunit, pytest, and pytest-xdist cannot be run. No compatible
wheel archives were present under the system or user pip-cache locations, and
the sandbox could not resolve `pypi.org`, so a local dependency environment
could not be bootstrapped. `findmnt -T .git` reported the repository metadata
as a dedicated read-only ext4 mount, while the worktree itself was writable;
`git add --dry-run` failed while creating `.git/index.lock`. Therefore the
approved measure-first implementation still cannot begin: there is no valid
baseline, test path, or commit path in this workspace.

### Dependency recovery and measured result

On 2026-08-28, the execution workspace exposed `uv`, outbound dependency
resolution, and writable Git metadata. A local `.venv` was built with Python
3.12.3, JAX/jaxlib 0.11.1, brainstate 0.5.4, pytest 9.1.1, and pytest-xdist
3.8.0. The CPU was a 12-logical-CPU Intel Core i9-12900H. No JAX, XLA,
`BRAINTRACE_TEST_*`, OMP, or MKL environment overrides were set. The
read-only-home sandbox required `BRAINEVENT_CACHE_DIR`, `XDG_CACHE_HOME`, and
`MPLCONFIGDIR` to point into the workspace; these cache-path changes were held
constant for baseline and candidate measurements.

Collection/import attribution, measured in three fresh processes with
`pytest --collect-only`, was 4.86--5.32 seconds wall (median 4.99 seconds) and
353,576--356,980 KiB peak RSS. Pytest itself reported 3.04--3.36 seconds to
collect 92 tests.

The candidate replaced all 13 fixed-shape Python sequence drivers with one
stacked `brainstate.random` input and `brainstate.transform.for_loop`. It left
the four varying-shape/model sweeps unchanged. Results were:

| Measure | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Fresh-process wall median | 91.13 s | 78.43 s | -13.9% |
| Fresh-process wall range | 84.62--93.28 s | 76.53--80.69 s | narrower by 4.50 s |
| Pytest runtime median | 88.83 s | 76.28 s | -14.1% |
| Peak RSS median | 967,656 KiB | 950,992 KiB | -1.7% |

All three baseline runs and all three candidate runs passed the same 92 tests.
The candidate also passed a separately selected run of all 13 changed tests.
No assertion, tolerance, marker, or test selection changed.

A separate MiniLSTM same-shape transform microbenchmark used
`block_until_ready` at both timing boundaries. The non-jitted `for_loop` path
had a 152.8 ms cold median and 147.2 ms second-call median. Composing an outer
`brainstate.transform.jit` had a 176.0 ms cold median and 1.83 ms warm median.
This demonstrates a 98.8% warm execution reduction after the outer driver is
compiled, but these pytest nodes each use a distinct cell/shape once, so the
cold compile cost is the relevant suite tradeoff. The non-jitted path was
initially described as an eager path; that label was incorrect because its
driver still used `for_loop`. Future benchmark notes must name the measured
driver directly rather than infer an execution mode from its timing.

The 13.9% focused wall-time improvement missed the pre-approved 15% retention
gate by 1.1 percentage points. The candidate was therefore removed. No
production or test code change is retained, and a full-suite/coverage run was
not performed because gates 3--5 apply only after gate 1 admits a candidate.
This prevents spending suite time or retaining transform complexity for a
rejected patch.

## Scope

### Phase 1: reproduce and attribute

In one dependency-complete CPU environment, record:

1. environment: Python, JAX, jaxlib, brainstate, pytest, pytest-xdist, CPU, and
   relevant XLA/JAX environment variables;
2. collection/import time for the two focused modules;
3. cold process wall time and pytest's slowest-test durations;
4. warm execution time after the same-shape transform has compiled, with
   `block_until_ready` at measurement boundaries;
5. peak resident memory when the environment supports reproducible sampling.

Run at least three fresh processes and report the median and range. Use one
pytest worker for attribution; xdist throughput is a separate whole-suite
measurement. Do not mix compilation and steady-state execution in one number.

The initial candidates are only the fixed-shape repeated execution tests in:

- `braintrace/nn/_rnn_test.py`;
- `braintrace/nn/_readout_test.py`.

Profile algorithm-test loops only if these modules do not provide a material
target or if pytest durations identify a larger repeated-execution hotspot.

### Phase 2: smallest measured change

For fixed-shape sequences, generate the complete input tensor with
`brainstate.random`, then execute the stateful cell with
`brainstate.transform.for_loop`. Return stacked outputs directly from the
transform. Use `brainstate.transform.jit` only for a one-shot call, and use
`scan` only where an explicit non-State carry is genuinely required.

Do not transform:

- loops over different batch sizes or model geometries, which necessarily
  compile different shapes;
- Python loops that inspect metadata, trees, or assertions rather than drive a
  model;
- scalar BPTT or finite-window oracle loops whose independent implementation is
  the purpose of the test;
- loops whose host-side reset or construction semantics cannot run inside a
  transform.

No production algorithm change is in scope for the first patch. If profiling
identifies a production hotspot, supersede this specification with measured
compile-time, warm-time, and memory evidence before changing it.

## Correctness invariants

- The same number and order of recurrent timesteps execute.
- Inputs keep the same shapes, dtypes, and brainstate random source.
- Every existing shape, state, convergence, and reset assertion remains.
- A regression test compares stacked outputs and final State from the transform
  path against a small eager reference for each affected cell family.
- Exact online-learning algorithms continue to match their BPTT oracles.
- Approximate-algorithm learning-rule assertions remain on finite-window oracle
  paths.
- No test is deselected, weakened, re-marked, or given a looser tolerance.

## Acceptance gates

1. The focused three-process median improves by at least 15%, with cold compile
   time and warm execution reported separately. A smaller result does not
   justify the added transform complexity and the patch is dropped.
2. The transformed warm path performs one trace for the loop body and one
   compiled execution for the sequence; evidence may be a trace counter or
   lowered-jaxpr inspection in a sibling regression test.
3. All focused tests pass in the supported JAX-version matrix available to the
   implementation environment.
4. Meaningful coverage for changed test-support code remains above 90%.
5. A representative full `braintrace/` run retains the complete default
   selection and records wall time, slowest durations, and peak memory. Overall
   CI improvement is reported separately from focused-module improvement.

## Benchmark commands

The exact executable is environment-specific, but the measurement shape is:

```bash
python -m pytest \
  braintrace/nn/_rnn_test.py \
  braintrace/nn/_readout_test.py \
  -q --durations=0

python -m pytest braintrace/ -n auto --durations=25 \
  --cov=braintrace --cov-report=term-missing
```

Run the focused command in at least three fresh processes for both baseline and
candidate revisions. Preserve `BRAINTRACE_TEST_JAX_CACHE_CLEAR_EVERY=40` for the
full-suite comparison unless memory measurements justify a separate approved
change.

## Edge cases and test cases

- sequence lengths 1 and greater than 1;
- batch size absent and present;
- LSTM-style tuple/structured State as well as single-array hidden State;
- random and constant input sequences;
- output stacking shape and dtype;
- final hidden State equality with the eager reference;
- reset after transformed execution;
- deterministic replay under the same brainstate seed;
- a failing shape or State mutation remains a failure rather than being hidden
  by tracing.

## Tradeoffs

Transforming a short test loop may reduce dispatch while increasing cold XLA
compile time. That tradeoff is why both components are measured and why the
15% focused-median gate is mandatory. A transform-based test can also be less
obvious than a scalar loop; the patch must stay small and use a local helper
only when it removes real duplication.
