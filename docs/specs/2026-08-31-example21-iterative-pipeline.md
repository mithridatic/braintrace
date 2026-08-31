# Example 21 iterative ARC evolution

Status: approved implementation specification

## Goal and claim boundary

Example 21 must expose one command that repeatedly trains and structurally
evolves one untyped `BrainCell` network across the complete ARC training
corpus. Every selected topology, parameter set, and optimizer state becomes the
parent of the next stage. A structural candidate does not need to solve another
task immediately: a protected improvement in the supervised ARC objective is
enough to carry it forward.

The mastery target is direct exact pass@1 on all 400 ARC training tasks. This is
an optimization target, not a guarantee that a bounded run will attain it. The
400 ARC evaluation tasks are held out from training, ranking, promotion, early
stopping, and configuration choices. They are scored exactly once when a run
ends. Training mastery must not be reported as held-out ARC mastery.

The pipeline compares two trained mutation siblings with their immutable
stage parent. It deliberately has no matched, unchanged, equally trained
control arm. Results therefore support an optimization decision, not a causal
claim that topology alone produced the change. Resource tie-breaking finds the
smallest encountered equally capable state; it does not prove a globally
minimal topology.

## Command and defaults

The primary command is:

```text
python examples/pp_prop/21-braincell-arc.py evolve --device gpu --arc-root /datasets/arc/raw --output-dir var/example21-evolve
```

The command owns the complete lifecycle: corpus loading, training, scoring,
structural stages, checkpoint handoff, progress reporting, plotting, terminal
evaluation, and interruption recovery. Reusing an output directory containing
a compatible unfinished run resumes it automatically. A configuration mismatch
or a closed run fails closed instead of silently starting another lineage.

Defaults are:

- Optax Muon for rank-two parameters, with Muon's built-in AdamW fallback for
  non-matrix parameters and decoupled weight decay `0.1`. Existing input,
  recurrent, and readout learning rates remain authoritative.
- `128` PP-Prop episode updates for ordinary round training and for every edge,
  neuron, edge-revisit, and Dale candidate. Proof mode remains exactly eight
  updates.
- Eight resumable rounds.
- Early stability after two consecutive rounds without an improvement under
  the selection order below.
- At most 4,096 neurons, 65,536 recurrent edges, the existing dynamic
  biological-connection limit, and the existing 32 MiB checkpoint limit.

Every 128-update schedule in a sibling comparison uses the same ordered tasks,
queries, seeds, and random inputs. The parent snapshot is immutable during the
comparison. There is no separately trained control candidate.

## Corpus and scoring contract

The loader builds a reproducible manifest from exactly the 400 ARC training
tasks. Task paths and IDs are sorted, every query is included, and the manifest
persists task IDs, source digests, query order, and the resume cursor. A resume
must verify these values before executing another update.

Training progress has two supervised measures:

1. Direct exact ARC pass@1 for each task.
2. A target-aware loss made from height cross-entropy, width cross-entropy, and
   valid output-cell color loss.

The scorer is a compiled forward-only path over the complete training corpus.
It reports exactness per task and the mean per-task loss over unresolved tasks.
Mutation ranking may use pre-clip gradient mass, task ownership, spike
evidence, and the existing structural evidence rules. Gradient mass is not a
progress score and cannot replace direct exactness or the supervised
shape-and-cell loss.

No evaluation task bytes, identifiers, labels, predictions, or aggregates may
enter a training update, mutation rank, promotion decision, stopping decision,
or retry. At terminal success, stability, or round-budget exhaustion, the
accepted checkpoint receives one direct exact pass@1 evaluation over all 400
evaluation tasks. The run is then marked closed so resume cannot turn the
evaluation result into tuning feedback.

## Round and stage lifecycle

Each round executes this order:

1. Continue PP-Prop for 128 updates from the accepted checkpoint and score the
   resulting parent on all 400 training tasks.
2. Build an edge-add sibling from the highest-ranked recurrent-edge candidates
   and an edge-prune sibling from the lowest-ranked 5% of recurrent edges.
   Train each for the same 128-query schedule, compare them with the stage
   parent, and carry the best improving state.
3. From that carried state, build a neuron-add sibling by twinning the
   highest-ranked 5% of neurons and a neuron-prune sibling from the
   lowest-ranked 5%. Train, compare, and carry the best improving state.
4. If the neuron topology changed, revisit recurrent edges with another
   matched add/prune sibling comparison and carry the best improving state.
5. From the carried state, compare measured excitatory and inhibitory Dale
   assignments over the selected 5% of neurons. Train both siblings, enforce
   their signs, and carry the best improving state.
6. Persist the round result, refresh progress artifacts, and continue until
   training mastery, two stable rounds, or the eight-round budget.

The existing deterministic 5% structural selection and sparse mutation
contracts remain authoritative. A stage may retain its parent when neither
sibling improves. A candidate may be carried without an additional exact task
when it meets the protected loss-improvement rule. Thus the topology can follow
useful gradient-ranked changes instead of waiting for a discrete pass@1 jump.

## Selection and compression

The stage parent and both trained siblings are ordered lexicographically:

1. Reject a non-finite state or any candidate that makes a previously exact
   training task inexact.
2. Prefer the candidate with more exact training tasks.
3. At equal exact count, accept lower mean per-task loss on unresolved tasks
   only when the decrease is at least
   `max(1e-6, 1e-4 * parent_unresolved_loss)`.
4. When exact count and objective loss tie, prefer fewer persistent bytes.
5. If persistent bytes also tie, prefer fewer neurons and then fewer recurrent
   edges.

An exact/loss tie that does not improve resources retains the parent. Candidate
comparison is deterministic, including stable index tie-breaking in structural
ranking.

Once all 400 training tasks are exact, the objective switches to
compression-first edge and neuron pruning. Compression may remove structure
only while preserving 400/400 direct exactness. It retains a state only when
persistent bytes decrease, with neuron and edge counts as the remaining
tie-breakers.

A no-progress round is one whose final state does not improve exact count,
protected unresolved-task loss, or mastery-preserving resource usage relative
to the accepted state at round entry. Two consecutive no-progress rounds end
the run as stable.

## Checkpoint handoff and compatibility

Every carried state includes the complete topology, model parameters, active
Muon state, stable neuron IDs, task-owner codes, Dale labels, round and stage
cursor, and checkpoint ancestry. A selected add, prune, or Dale child is saved
before it can become a downstream parent. A rejected child cannot mutate the
parent and its temporary files are removed after the comparison is durably
recorded.

Topology rebuilding must:

- Canonically sort input and recurrent CSR entries and apply the identical
  permutation to edge values and every corresponding optimizer array.
- Preserve parameter and optimizer values for surviving items, including Muon
  and AdamW-fallback step state, and initialize new-item optimizer state
  according to the existing structural contract.
- Assign twins new monotonic neuron IDs while retaining stable IDs for all
  surviving neurons.
- Recompute task ownership from the selected state before saving.
- Reset eligibility state at every topology-shape change while preserving the
  persistent model and optimizer state.
- Enforce every accepted Dale sign after updates and structural operations.
- Resume partially typed parents without discarding existing assignments.

Checkpoint validation must use the authoritative dynamic and configured limits
rather than the baseline-only 30,496-connection assumption. Existing format-1
checkpoints remain readable. A rewritten selected checkpoint uses the current
format and must round-trip topology, parameters, optimizer state, stable IDs,
owners, signs, cursor, and ancestry exactly.

## Execution and resource bounds

Repeated model execution must not be driven by a bare Python `for` or `while`
loop. Each 128-update block uses `brainstate.transform.for_loop`, composed with
`brainstate.transform.jit` where appropriate. Full-corpus forward scoring is
compiled as well. Python may coordinate stages whose topology shapes differ and
may perform bounded artifact I/O; it may not dispatch model steps or task
screens one operation at a time.

Every candidate is validated against all limits before construction and again
before promotion. The implementation remains sparse and must not materialize a
dense neuron-pair array. It fails closed on connection-limit, neuron-limit,
biological-limit, checkpoint-size, non-finite, malformed-corpus, or incompatible
resume conditions.

Persistent bytes are the authoritative resource tie-break and include the
selected topology, parameters, optimizer state, stable IDs, ownership, and Dale
state needed to continue the run. Checkpoint bytes, host RAM, and device memory
are recorded engineering measures but do not replace persistent bytes in
selection.

## Durable progress and visual artifacts

The output directory contains:

- `run-state.json`, the current configuration, cursor, lineage, and closed/open
  state.
- Versioned accepted checkpoints addressed by round and stage.
- `progress.jsonl`, an append-only record of every completed sibling comparison
  and round boundary.
- `topology.png`, the latest accepted graph.
- `score-history.png`, the accepted score and resource history.
- One terminal evaluation result after the run is closed.

State JSON, accepted checkpoints, and plots are written through sibling
temporary files and atomic replacement. A progress record is flushed durably
before `run-state.json` advances past that stage. Every record has a stable
stage ID so resume can reconcile an interrupted write without repeating an
accepted mutation or terminal evaluation.

Progress records include the stage and round; parent and child checkpoint
digests; disposition of both siblings; exact count and solved task IDs;
shape-and-cell loss; update count; neuron and recurrent-edge counts; persistent
and checkpoint bytes; elapsed time; peak host RAM; and available device-memory
evidence. They must distinguish an accepted mutation, retained parent, blocked
candidate, failed candidate, and terminal state.

`topology.png` and `score-history.png` refresh after every selected stage and at
completion. The topology image shows stable neuron lineage, Dale groups, task
ownership, and every executed recurrent edge. It is explicitly a graph view of
the learned system, not an anatomical or spatial brain image. The history image
shows exact tasks, unresolved-task loss, topology size, and persistent bytes by
accepted stage.

## Verification

Implementation begins with a failing regression test proving that the current
Example 21 runner does not consume a selected structural child as the next
stage parent. The fix is complete only when the child's topology, parameters,
and active Muon state are consumed downstream.

Co-located suffix-style tests cover:

- Muon and 128-update defaults in every non-proof phase, with proof mode fixed
  at eight updates.
- Sorted 400-training-task manifests, all-query scheduling, resume cursors, and
  strict terminal-only isolation of all 400 evaluation tasks.
- Direct exact scoring, height/width/cell loss, minimum loss improvement,
  solved-task protection, persistent-byte tie-breaking, and mastery-preserving
  compression.
- Equal sibling schedules, deterministic randomness, parent immutability,
  downstream edge-to-neuron-to-edge-revisit-to-Dale handoff, and rejected-child
  cleanup.
- Growth checkpoint round trips, canonical CSR and optimizer alignment,
  surviving and new optimizer state, stable IDs, owner refresh, partially typed
  continuation, and Dale enforcement.
- Zero-edge, one-neuron, unavailable-donor, configured and biological
  connection-cap, neuron-cap, checkpoint-cap, non-finite, malformed-corpus, and
  all-solved cases.
- Atomic interruption and resume at each durable boundary, config-drift
  rejection, idempotent reconciliation, closed-run behavior, exactly one
  terminal evaluation, complete progress records, and automatic plots.
- Static and runtime protection against repeated bare Python model loops.

All new or changed public APIs use NumPy-style docstrings. Changed Example 21
modules require more than 90% meaningful combined line-and-branch coverage,
with edge cases and critical paths represented rather than trivial line hits.
Qualification includes focused tests, the complete co-located Example 21 gate,
relevant `examples/pp_prop` and package tests, Ruff, mypy, strict OpenSpec
validation, and `git diff --check`.

A real GPU canary runs one complete default round: 128 ordinary updates, all
400 training tasks, every eligible structural stage, and an interruption/resume
boundary. The canary must verify checkpoint handoff, artifact production, host
and device resource evidence, and literal training and held-out evaluation
scores. Passing the canary does not by itself establish 400/400 mastery or a
globally minimal topology.

All implementation and qualification stay on the dedicated feature worktree.
Integration into `main` requires separate approval.
