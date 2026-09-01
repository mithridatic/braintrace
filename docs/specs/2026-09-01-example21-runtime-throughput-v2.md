# Example 21 runtime throughput v2

## Purpose

Profile the current BrainCell Example 21 evolution workflow at commit `c9caf3d`
and reduce its measured end-to-end runtime without changing its learning,
scoring, structural-selection, or persistence semantics.

The compact scoring implementation already present at the baseline commit is
part of the baseline. Its earlier qualification is not evidence for this
change. This work records a fresh profile and a separate candidate comparison.

## Scope

Measure these phases independently:

- checkpoint load and runtime construction;
- immutable training-payload preparation and device transfer;
- compilation and execution of one 128-update PP-Prop block;
- compilation and execution of the 64-task and 400-task score paths;
- structural evidence and topology mutation;
- checkpoint serialization, hashing, and validation; and
- complete candidate and one-round evolution wall time.

The implementation target is the largest repeatable phase that can be changed
without altering the invariants below. No optimization is selected until the
fresh baseline profile identifies that phase.

## Selected design

The fresh profile identified direct scoring as the largest repeatable phase.
The selected implementation changes only complete-corpus scoring:

- preserve each query as one compiled recurrent scan;
- accumulate spike activity in the scan carry instead of retaining the full
  time-by-neuron spike history;
- retain the voltage history so the final 31 readouts use the same recurrent
  execution boundary and floating-point path as the baseline;
- route the 416 training queries into the corpus-optimal static boundaries
  `(193, 257, 321, 449, 705)`, reducing executed slots from `178,935` to
  `119,200`; and
- retain the baseline `(320, 705)` buckets and full-history kernel for every
  task subset, because compiling five shapes regressed the 64-task screen path.

The bucket choice is fixed by score scope and cached with the existing
role-and-task identity. It does not reorder queries or advancing events. The
training path, PP-Prop trace evolution, update schedule, optimizer, mutation
policy, persistence, and evaluation isolation are unchanged.

## Baseline protocol

Use the local image
`sha256:4a7546a80f424ad79b0a950e3683ebf214aab164bf2b1d58f50ad939293bfb90`
with the worktree mounted read-only and `PYTHONPATH` pinned to that mount. Use
the embedded ARC corpus, the RTX 3080 Ti Laptop GPU, the default sparse backend,
Muon, exactly 128 updates per training block, 64 screen tasks, and the current
topology limits.

Record cold compile time separately from warm execution. Synchronize device
results at phase boundaries. Record host peak RSS, device peak bytes, exact
task counts, unresolved loss, neuron count, recurrent-edge count, and executed
updates alongside every timing.

Use an isolated output directory and compilation cache. Do not read timings
from an earlier run as the baseline for this change.

## Correctness invariants

For identical source checkpoint, corpus, update schedule, seed, and operation:

- every training candidate executes exactly 128 PP-Prop updates;
- score-only and terminal-evaluation stages execute zero updates;
- scheduled query order and advancing-event order are unchanged;
- decoded predictions, per-query exact flags, per-task exact flags, owners,
  owner codes, and selected structural arm are identical;
- per-query and per-task loss differ by at most `1e-6`;
- neuron and recurrent-edge counts, Dale assignments, and mutation topology
  digests are identical;
- parameter and optimizer identity is unchanged by scoring;
- checkpoint lineage, resume state, and terminal-only evaluation remain valid;
- no evaluation payload becomes reachable during training; and
- repeated model execution uses `brainstate.transform` primitives, never a
  bare Python `for` or `while` loop.

Use `brainstate.random` for any new random generation.

## Tests

Write co-located suffix-style tests beside every changed module. Cover normal,
empty, boundary, malformed, cache-invalidation, topology-change, schedule-change,
and state-isolation paths relevant to the selected optimization. Changed code
must exceed 90 percent meaningful branch coverage.

The focused Example 21 adapter, evolution, structural, and BrainCell ARC tests
must pass. Run Ruff, formatting, BasedPyright on changed production modules,
and `git diff --check`.

## Performance promotion

Compare baseline and candidate in fresh, otherwise identical Docker processes.
Use one cold call and at least three warm calls for the affected phase. Promote
only when all correctness invariants pass and:

- the affected repeated phase improves by at least `1.20x` in warm median;
- representative candidate wall time does not regress;
- one-round end-to-end evolution is measurably faster than the fresh baseline;
- host and device peak memory do not materially regress; and
- the improvement remains after including preparation, transfer, compilation,
  execution, scoring, and persistence costs that the normal command pays.

If no candidate clears these gates, retain the baseline and document the
profile and rejected attempt. Do not report a partial microbenchmark as an
Example 21 speedup.

## Non-goals

This work does not change ARC capability, update count, optimizer, loss,
training schedule, topology policy, Dale policy, screen membership, evaluation
isolation, checkpoint schema, or the retired latent-workspace implementation.
