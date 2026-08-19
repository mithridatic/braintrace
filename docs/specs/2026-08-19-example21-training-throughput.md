# Example 21 — compact-horizon streaming training

Status: implemented; focused verification passed; 2x aspiration not met
Date: 2026-08-19
Branch: `feat/example21-row-refinement`
Follows `2026-08-19-example21-batched-training-results.md`.

## 1. Objective

Make long Example 21 pp-prop pretraining runs substantially faster without
changing the admitted data, episode schedule, optimizer, loss, or learned
parameters. The change has two parts:

1. derive the smallest safe compiled training horizon from the admitted dataset
   and its enabled augmentation contract instead of always scanning 390 ticks;
2. replace the reusable in-memory episode bank with one bounded asynchronous
   producer that prepares the next chunk while the GPU trains the current one.

This is a performance change, not a new training method. No score improvement is
claimed by this specification.

The next stage adds deterministic multi-worker row construction. The training
thread owns the random schedule and assigns one immutable seed to each episode;
workers only materialize rows from those descriptors. Results are restored to
ordinal `(update, batch-slot)` order before batching, so worker count and
completion timing cannot change tensors, metadata, or optimizer updates.

## 2. Current evidence

The production training path currently pads and scans every episode to 390
ticks. Across every leave-one-out fold and both dihedral orientations in the
pinned ARC-AGI-1 training split, the measured compact horizon has a maximum of
180 ticks and a median of 93. Therefore 210 of the 390 compiled ticks, 54%, are
a guaranteed suffix for this admitted split even in its longest episode.

Fresh synchronous episode production also starves the accelerator. In the
measured 600-update run without an episode bank, host encoding took about 217
seconds while GPU training took about 73 seconds. The retained 6,000-update run
reduced repeated encoding by building a reusable bank, but took 2,167.6 seconds
end to end, including about 180 seconds of bank encoding, and the bank occupied
14.76 GiB of host memory. That bank cannot scale with the materially longer,
fresh episode stream indicated by the still-descending loss curve.

These timings identify separate costs. Prefetch can overlap host construction
with device execution and remove bank growth. The compact horizon removes
unnecessary recurrent steps from the compiled program itself.

## 3. Compact training horizon

Before compiling the training driver, compute the maximum encoded length over
all tasks admitted to training and every augmentation orientation the configured
training sampler may emit. Allocate training tensors and compile the scan at
that bound. For the current pinned split and augmentation contract, the expected
bound is 180 rather than 390.

The derivation must be fail-closed:

- the bound is computed from the resolved, admitted dataset, not copied as an
  unexplained constant;
- it includes every enabled transformation that can change encoded length;
- each encoded episode is checked before it enters a batch, and a length above
  the bound raises an error rather than truncating, wrapping, or recompiling
  silently;
- an empty admitted dataset, an unsupported augmentation, or an indeterminate
  bound is an error;
- evaluation and adaptation retain their independently required capacities;
  this optimization must not narrow those paths merely because the training
  split is shorter;
- the resolved horizon and the evidence used to derive it are recorded in the
  run report.

Changing the dataset or augmentation configuration may produce a different
safe horizon and a corresponding compilation. No episode may be shortened to
preserve the 180-tick value.

## 4. Bounded asynchronous prefetch

Training uses one producer and one consumer. While the GPU executes chunk `n`,
a worker prepares chunk `n + 1` into host NumPy buffers. The queue capacity is
exactly one complete chunk beyond the chunk being consumed. Once a chunk is
consumed, its host buffers become reclaimable; no reusable episode bank is
retained.

The producer owns episode construction only. Device placement, compilation,
optimizer updates, loss collection, and parameter mutation remain on the
training thread. The random schedule is derived before asynchronous execution,
or otherwise partitioned deterministically, so worker timing cannot change
which tasks or augmentations are selected.

Episode materialization may use a bounded thread pool inside the producer. At
most `2 * training_workers` row futures are in flight, and the producer still
publishes at most one complete chunk to the outer queue. Futures are collected
by ordinal rather than completion order. A worker count of one is the serial
oracle for the worker-count equivalence gate.

The lifecycle must propagate failures rather than conceal them:

- a producer exception is re-raised on the training thread with its original
  cause and stops further optimizer updates;
- a consumer/device exception signals cancellation so the producer cannot hang
  while trying to publish a chunk;
- normal completion, early return, and exceptions all join the worker and
  release queued buffers;
- the producer must not outlive `run_experiment` or leave a blocked non-daemon
  thread behind;
- requesting zero updates performs no production work.

## 5. Acceptance gates

### 5.1 Exact-equivalence gate

For a fixed deterministic fixture and seed, compare the legacy 390-tick,
synchronous path with the compact-horizon, prefetched path. Schedule and tensor
semantics must match exactly. Losses and parameter leaves must match within the
repository's established `1e-6` numerical tolerance because shortening a JAX
scan changes the zero-gradient reduction tree and therefore need not be byte
identical:

- sampled task, fold, effort, and augmentation schedules;
- per-update losses, including chunk boundaries, within `1e-6`;
- every final parameter leaf within `1e-6`, with digests recorded separately;
- optimizer-update count and report accounting.

The comparison covers at least two equal-sized chunks so that production
overlaps device work and crosses a queue hand-off. Ragged final chunks remain
fail-closed under the existing divisibility contract. Any difference blocks
rollout.

### 5.2 Horizon and failure gates

Tests must prove that the bound includes the longest admitted transformed
episode, that shorter episodes preserve the same masks and values after removal
of the dead suffix, and that an over-bound episode fails before an optimizer
update. Unsupported or indeterminate training inputs must fail closed.

Injected producer and consumer failures must reach the caller, cancel the other
side, join the worker, and leave no further parameter updates or live worker.

### 5.3 Memory gate

Peak host memory must remain bounded by the active chunk plus one prefetched
chunk and fixed model/runtime overhead. It must not grow with total update count
or total generated episodes. A long synthetic run must demonstrate a flat
steady-state host-memory envelope after warm-up, and inspection must confirm
that no episode-bank collection remains reachable.

Existing device-memory policy gates continue to apply. Batch-size exploration
may try 16, 32, and 48 only when each configuration stays inside those gates;
the fastest safe batch is selected from measurement rather than assumed.

### 5.4 Throughput gate

The performance aspiration is at least 2× end-to-end episodes per second against
the current representative GPU path, with matched model, dataset, seed, batch,
update count, and warm compilation state. Report encoding time, accelerator time,
end-to-end wall time, horizon, and peak host/device memory for both arms.

The 2× result is not yet achieved or claimed. If the representative benchmark
does not reach it, retain only changes that pass the exactness and resource gates
and continue profiling before describing the run as substantially faster.

## 6. Required tests

Tests are co-located with the implementation modules and cover:

- bound derivation over all admitted tasks and enabled orientations;
- exact 180-tick derivation for the pinned fixture used by qualification;
- exact equality with the legacy schedule, losses, parameters, and digest;
- multi-chunk queue hand-offs and the existing ragged-chunk rejection;
- bounded queue and absence of episode retention after consumption;
- producer failure, consumer failure, cancellation, and worker cleanup;
- zero-update behavior and over-bound fail-closed behavior;
- a representative throughput benchmark recorded separately from the unit-test
  gate.

## 7. Measured result

The implementation derives a 180-tick training horizon for the pinned split,
threads it through fresh and banked schedules, and overlaps one fresh CPU chunk
with the current compiled GPU chunk. A short RTX 3080 Ti Laptop benchmark used
512 neurons, 65,536 recurrent edges, 40 updates, fresh augmented episodes, and
20-update chunks:

| path | batch | horizon | episodes | wall seconds | episodes/s |
|---|---:|---:|---:|---:|---:|
| legacy synchronous | 16 | 390 | 640 | 29.385 | 21.780 |
| compact + prefetch | 16 | 180 | 640 | 22.538 | 28.396 |
| compact + prefetch | 32 | 180 | 1,280 | 34.723 | 36.863 |
| compact + prefetch | 48 | 180 | 1,920 | 47.836 | 40.137 |

At matched batch 16 the implementation is 1.30x faster end to end. The matched
final loss differed by `4.77e-7`, within the declared numerical-equivalence
gate. Batch 48
raises delivered episode throughput by 1.84x relative to the retained batch-16
legacy pipeline. This is a material improvement, but it does not meet the 2x
aspiration, so no 2x claim is made. The next bottleneck is fresh host episode
construction rather than the compiled training horizon.

The deterministic row-worker sweep on the smoke fixture (30 fresh episodes,
10-update chunks, warm process) took 1.061 s with one worker, 0.144 s with two,
and 0.128 s with four. This is a CPU-construction signal, not a production GPU
throughput claim; the production-sized 1,024-neuron benchmark remains a separate
qualification run.
