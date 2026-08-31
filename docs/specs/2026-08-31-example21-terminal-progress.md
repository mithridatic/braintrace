# Example 21 terminal progress reporting

## Purpose

The resumable Example 21 ARC evolution command can run for many hours. Its
durable artifacts describe completed work, but the command does not currently
show which candidate or stage is running. Add live, GEPA-style terminal
reporting without changing evolution policy, compiled training, checkpoint
identity, or the meaning of any durable artifact.

## Scope

This change adds an optional reporting boundary to `run_evolution` and enables
one console implementation in the numbered Example 21 CLI. Library callers
that omit the reporter remain silent. The change does not alter candidate
construction, selection, update schedules, checkpoint persistence, resume
validation, terminal scoring, or plotting.

`progress.jsonl`, `run-state.json`, accepted checkpoints, `topology.png`, and
`score-history.png` remain the authoritative record. Terminal output is an
operator view of the same lifecycle. It is not a second persistence channel.

## Reporting interface

`run_evolution` accepts an optional progress reporter. The reporter consumes
immutable progress events. Each event has an event name and a read-only field
mapping. A missing reporter uses a no-op implementation.

Events use one-based round numbers for display and retain canonical stage IDs
for correlation with durable progress. Model-stage events include the stage,
stage ID, candidate arm, configured round budget, and the fields available at
that boundary.

The lifecycle emits these events in order:

1. `resume`, only when an existing run state is restored.
2. `candidate-start`, immediately before each compiled candidate block.
3. `candidate-result`, immediately after that candidate returns. The event
   reports `completed`, `blocked`, or `failed`, its reason when present, and its
   literal executed-update count.
4. `selection`, after all siblings are scored and the selected checkpoint is
   durably committed. It identifies the accepted arm or retained parent and
   reports the best accepted training score and topology.
5. `round-end`, after the round transition is durable.
6. `terminal-start`, immediately before terminal-only evaluation.
7. `terminal-result`, after the evaluation result and closed state are durable.

Training, edge add and prune, neuron add and prune, optional edge revisit, Dale
excitatory and inhibitory assignments, edge and neuron compression, round
completion, and terminal evaluation all pass through this event sequence. A
Dale stage therefore reports both sign attempts before it reports the accepted
assignment or retained parent.

Candidate result fields are literal. A completed candidate reports exact tasks,
total tasks, unresolved-task loss, neurons, recurrent edges, and 128 executed
updates. A blocked candidate reports its reason and zero executed updates. A
failed candidate reports its reason and the adapter-provided partial update
count. No reporter infers work that the adapter did not report.

On resume, one `resume` event reports the restored round, next stage, accepted
score, neuron and edge counts, checkpoint path, and checkpoint SHA-256. It does
not replay historical candidate or selection events.

## Console behavior

The console reporter writes only to stderr and flushes every write. The CLI
continues to write exactly one final machine-readable JSON document to stdout.

When stderr is a TTY, a running candidate or terminal evaluation owns one
replaceable line:

```text
Example 21 ARC | Round 1/8 | edge:add | running 02:14
```

A lightweight reporter thread refreshes elapsed wall time while the compiled
block runs. It does not call, inspect, or iterate the model. The reporter clears
the temporary line before every permanent result.

When stderr is redirected, including ordinary Docker logs, no carriage-return
animation is used. The reporter emits one permanent, immediately flushed start
line instead. Candidate completion uses this form:

```text
[02:14] Round 1/8 edge:add completed | score 3/400 | loss 0.8412 | neurons 2048 | recurrent edges 17203 | updates 128
```

Selection uses this form:

```text
[04:29] Round 1/8 edge selected add | best 3/400 | neurons 2048 | recurrent edges 17203
```

Retained-parent, blocked, failed, round-end, resume, and terminal lines use the
same elapsed-time prefix and explicit status language. Loss formatting is
stable and concise; reasons are normalized to one physical output line.

## Behavioral invariants

- Candidate calls retain their existing order.
- Every PP-Prop candidate block still executes exactly 128 updates.
- The adapter remains responsible for compiling each repeated update block with
  BrainState transforms. Reporting does not enter the compiled model call.
- Reporter timestamps and terminal I/O do not enter selection or persistence.
- Adding a reporter cannot change candidate scores, selected checkpoints,
  checkpoint digests, topology identities, Muon state, update cursor, or
  terminal evaluation.
- A reporter is closed in a `finally` path so a failed or interrupted model call
  cannot leave a live terminal-refresh thread behind.

## Verification

Co-located tests must cover:

- Exact event ordering and required fields for candidate start, result,
  selection, round end, and terminal evaluation.
- TTY replacement lines and redirected permanent start lines.
- Immediate stderr flushing and stdout containing only the final JSON document.
- Accepted candidates, retained parents, blocked and failed arms, optional edge
  revisit, Dale assignment, compression, early mastery, stable termination,
  round-budget termination, and interrupted resume without historical replay.
- Identical adapter call ordering, schedules, accepted scores, checkpoints,
  checkpoint identities, topology, optimizer identity, and final state with and
  without a reporter.
- More than 90 percent branch coverage for the added reporting implementation,
  followed by the relevant Example 21 coordinator, adapter, and CLI tests.

The production acceptance run rebuilds `braintrace-example21:latest` from
`feat/example21-iterative-pipeline`, verifies the embedded 400-task training and
400-task evaluation corpus, runs the existing GPU/Muon 128-update canary, then
starts or resumes the normal eight-round, patience-two run in
`var/example21-evolve`. Existing incompatible evidence is preserved and causes
a fail-closed report rather than automatic deletion.
