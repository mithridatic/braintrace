# Example 21 scoring throughput

## Purpose

Speed up direct ARC scoring in the current BrainCell Example 21 evolution
workflow without changing training, structural selection, checkpoint identity,
or score meaning.

The optimization applies to
`Example21ArcAdapter._score_runtime`. It does not revive or modify the retired
latent-workspace implementation.

## Profiled baseline

The implementation baseline is `main` commit `25f84bc`. The scoring adapter is
byte-identical to the adapter used by the latest durable chained-operation GPU
canary.

That canary recorded these stage times on the RTX 3080 Ti Laptop GPU:

| stage class | scope | seconds |
|---|---:|---:|
| parent training and score | 400 tasks | 94.09 |
| screen-scope transition | 64 tasks | 13.33 |
| structural operation | 64 tasks per sibling | 80.40 to 107.45 |
| round-boundary score | 400 tasks | 78.99 |
| terminal evaluation | 400 tasks | 73.08 |

The complete round used 1,013.34 recorded stage seconds, including terminal
evaluation. Scoring is therefore repeated on every trained candidate and is a
material part of both screen and full-corpus runtime.

A read-only current-source diagnostic used the accepted
`var/example21-ops-canary/checkpoints/r000-round-score.npz` checkpoint and the
embedded training corpus. It measured:

- 400 tasks and 416 supervised queries;
- fixed event shape `(705, 441)`;
- 293,280 scheduled event slots;
- 116,128 advancing event slots;
- 177,152 non-advancing slots, or 60.40 percent of all slots;
- 1.38 seconds to load and encode the corpus;
- 7.41 seconds to construct the checkpoint runtime;
- 12.80 seconds for the first 64-task score;
- 11.39 seconds for the second identical 64-task score;
- identical `3/64` exact tasks and unresolved loss
  `2.2336226926434386` in both scores; and
- 213,863,424 bytes of observed peak device use in that diagnostic process.

A second diagnostic compared manifest-ordered task subsets. Thirty-two
low-advance queries averaged 193 advancing events and took 6.27 seconds.
Forty-two high-advance queries averaged 445.95 advancing events and took 8.00
seconds. Both cost approximately 0.19 seconds per query despite a 2.31-times
difference in useful recurrent work. The fixed 705-iteration masked loop is the
primary scoring bottleneck.

## Current semantics

Each encoded query has 705 positions. A false advance mask returns exact zero
output and leaves biological state unchanged. The final 31 advancing events are
the direct shape and row requests used to construct `(31, 360)` logits.

Scoring currently:

1. stacks every query at the fixed 705-event length;
2. resets model and eligibility state for each query;
3. executes all 705 positions with a compiled BrainState `for_loop`;
4. decodes the final 31 voltage features;
5. averages absolute spike activity over the original 705 positions;
6. restores manifest order for task exactness, loss, ownership, and structural
   ranking evidence; and
7. verifies that scoring did not change trainable parameters.

The optimization must preserve those semantics. It may remove a false advance
from physical execution because that event has no state transition, but it may
not remove, reorder, or alter an advancing event.

## Design

### Compact query representation

Build an immutable scoring payload from each `SupervisedQuery`:

- retain `events[advances]` in original order;
- retain the original fixed event count for activity normalization;
- require at least 31 advancing events;
- require the original final 31 positions to advance;
- require the retained final 31 events to be those request events; and
- reject a query longer than the largest declared bucket.

Right-align the retained events in the smallest bucket that can hold them.
Leading bucket positions contain zero events and false advance masks. The
retained request events therefore remain the final 31 positions, so the direct
decoder boundary does not change.

### Bucketed compiled execution

Group compact payloads by a small, fixed sequence of bucket lengths. Each
bucket executes one `brainstate.transform.jit` containing one
`brainstate.transform.for_loop` over its queries. The model still resets inside
that compiled query loop.

The considered bucket schemes are:

- `(320, 705)`, which retains 178,935 slots and removes 38.99 percent;
- `(256, 384, 705)`, which retains 164,967 slots and removes 43.75 percent; and
- `(256, 320, 448, 705)`, which retains 149,844 slots and removes 48.91
  percent.

More buckets reduce executed slots but add topology-specific JAX compilations.
Ship `(320, 705)` when it clears both promotion thresholds. It is the smallest
scheme, requires only two fixed compiled programs, removes 38.99 percent of
full-corpus slots, and achieved the required fixed-checkpoint speedup. Treat a
larger scheme as a separate optimization that must improve the median beyond
timing noise before accepting its additional compilation and cache cost.

Bucket results carry their original query indices. Reassemble logits and
activity in exact manifest query order before calculating any host-side metric.

### Activity normalization

Non-advancing events currently contribute exact zero spike values to a mean
whose denominator is 705. The compact scorer must calculate spike activity as
the accumulated absolute spike mass divided by the original fixed event count,
not by the selected bucket length or advancing-event count.

### Deterministic structural evidence

Changing the compiled loop length changes low-order GPU floating-point results
even though false advances are mathematical no-ops. A fixed-checkpoint
diagnostic measured at most `2.75e-8` difference in normalized task-neuron
evidence. Those differences can change exact maximum comparisons and therefore
owner labels, Dale candidates, and mutation tie-breaking.

After per-task normalization, round the task-by-neuron evidence matrix to six
decimal places. Use that same quantized matrix for task owners, owner codes,
activity maxima, and every neuron, recurrent-edge, source, and target structural
score. Do not quantize logits, decoded predictions, exact flags, or task loss.

Six decimals is the declared structural-evidence precision. It exceeds the
observed compact-versus-legacy numerical drift while retaining materially more
precision than structural selection needs. The source checkpoint remains
immutable, but rescoring an older checkpoint under this policy can change its
persisted owner labels: the qualification checkpoint changes 30 historical
owner sets relative to the former unquantized policy. This is an approved
semantic boundary, not a claim of byte identity with historical evidence.

### Cache boundary

Prepared compact host payloads may be cached by corpus role and ordered scored
task IDs. The cache must not load evaluation data during training. Device
arrays, compiled functions, runtime state, parameters, and optimizer state must
not cross checkpoint or topology boundaries through this cache.

## Invariants

For the same checkpoint, corpus, and ordered task subset:

- advancing event bytes and their order are unchanged;
- the final 31 request events are unchanged;
- decoded predictions and per-query exact flags are identical;
- task IDs, query order, task exact flags, and exact-task counts are identical;
- per-query and per-task loss differ by at most `1e-6`;
- legacy 705-position execution and compact execution produce task-neuron
  evidence within `1e-6` under the common six-decimal policy, identical owner
  codes and task-owner sets, and identical mutation topology for neuron add,
  neuron prune, edge add, edge prune, Dale-positive, and Dale-negative stages;
- derived structural ranking evidence differs by at most `1e-6`, retains the
  same finite/nonnegative properties, and does not change selection;
- scoring leaves parameter and optimizer identities unchanged;
- evaluation remains terminal-only; and
- every repeated model loop uses BrainState transform primitives rather than a
  Python `for` or `while` loop.

If compaction under the common six-decimal policy changes a decoded answer, an
exact flag, an owner, a structural mutation topology, a non-finite
classification, or persistent state, reject the optimization.

## Tests

Co-located tests in `example21_arc_adapter_test.py` must cover:

- no-op events before, between, and after advancing events;
- right alignment and exact advancing-event preservation;
- every bucket boundary and the 705-event upper bound;
- a fully occupied 705-event query;
- fewer than 31 advancing events;
- a non-advancing event in the final request tail;
- multiple queries belonging to one task;
- manifest-order reconstruction across buckets;
- the fixed 705-event activity denominator;
- unchanged predictions and exact flags, loss tolerance, and identical owners
  and mutation topology against legacy 705-position execution under the common
  six-decimal evidence policy;
- deliberate owner-policy changes against historical unquantized evidence;
- unchanged parameter and optimizer identities; and
- lazy evaluation-corpus access.

Meaningful branch coverage for the changed scoring code must exceed 90 percent.

## Performance qualification

Run benchmarks in a fresh Docker process with:

- the current worktree mounted read-only;
- `PYTHONPATH` pinned to that worktree;
- the embedded ARC corpus;
- the accepted round-score checkpoint;
- the same GPU and sparse backend for baseline and candidate; and
- quality and memory fields recorded beside every timing.

Measure one first call and three subsequent calls for both the first 64
manifest tasks and the complete 400-task training corpus. Compare the warm
three-run median. Report query count, advancing slots, executed bucket slots,
exact count, unresolved loss, host peak, and device peak.

Promotion requires:

- at least 1.20-times warm-median speedup for the 64-task screen;
- at least 1.20-times warm-median speedup for the 400-task score;
- no correctness-invariant failure;
- no material host or device peak-memory regression; and
- a successful one-round GPU evolution canary with literal update counts,
  durable checkpoint lineage, full-corpus round scoring, and terminal
  evaluation.

The fixed-checkpoint comparison proves score equivalence. The evolution canary
proves integration and lifecycle behavior; its structurally adapted score may
vary under documented GPU arithmetic and is not substituted for the fixed
checkpoint correctness gate.

If the compact bucketed scorer misses the performance gate, do not land it as a
speedup. Re-profile compilation and execution separately before considering a
streaming carry or batched-state redesign.

## Non-goals

This change does not:

- alter the 128-update training block;
- reorder update schedules or structural siblings;
- change model parameters, optimizer policy, topology mutation, Dale policy,
  selection thresholds, screen membership, or checkpoint schema;
- relax exact ARC scoring or evaluation isolation;
- claim improved ARC capability; or
- optimize the retired latent-workspace Example 21.
