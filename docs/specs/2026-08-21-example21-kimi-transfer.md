# Example 21 Kimi-transfer campaign

## Status

Accepted for staged implementation. Scientific promotion remains evidence-gated.

## Objective

Evaluate transferable Kimi K3 mechanisms in BrainTrace and Example 21 without
claiming that Kimi's full architecture or training system has been reproduced.
Generic BrainTrace operators may survive a failed Example 21 arm when their
operator-level gates pass. Example 21 defaults change only after the full
reduced-topology promotion gate passes.

## Isolation and provenance

- Branch: `feat/example21-kimi-transfer`.
- Worktree: `.worktrees/example21-kimi-transfer`.
- Base revision: `7843012e546a2ab288900bf2d6bf9e3a88fb5711`.
- No merge, push, publication, registry upload, or `main` modification is in
  scope.
- ARC protocol: Example 21 protocol v2 with fixed source manifest and identical
  task/data schedules within every matched comparison.
- Canonical reduced topology: 4,096 neurons, 4,096 recurrent edges, memory
  width 32, 60 latent ticks, 260 updates, batch 32.
- Full evaluation: 419 queries at seeds 2108, 31337, and 7777.
- A 1,048,576-edge seed-2108 run is a final nonblocking sensitivity diagnostic,
  not a promotion requirement. A 4,194,304-edge run requires separate approval.

## Stage order

0. Revalidate optimizer, schedule, and softcap defaults.
1. Add gated and RMS-normalized associative-memory reads.
2. Alternate local recurrence with periodic global memory reads.
3. Apply Attention Residuals across latent-depth blocks.
4. Add a progressive effort curriculum.
5. Compare KDA-style delta writes and a full SiTU-GLU updater independently.
6. Add effort self-distillation only when effort 60 is a valid teacher.

Every stage has its own specification and commit. A stage starts from the last
accepted stack, changes one mechanism, and records explicit missing evidence.

## Shared configuration surface

The following options are opt-in and must round-trip through configuration,
CLI, reports, result JSON, checkpoint compatibility, and model architecture
manifests:

| Setting | Values | Default |
| --- | --- | --- |
| `memory_read_transform` | `linear`, `gated`, `gated_rms` | `linear` |
| `memory_read_interval` | positive integer | `1` |
| `latent_residual_mixer` | `none`, `attention_residual` | `none` |
| `latent_residual_block_size` | positive integer | `10` |
| `effort_schedule` | `uniform`, `progressive` | `uniform` |
| `lr_warmup_fraction` | finite `[0, 1)` | `0.0` |
| `memory_coding` additions | `delta_write`, `situ_glu_update` | unchanged |
| `effort_distillation_weight` | finite nonnegative | `0.0` |

Legacy defaults must be function-identical unless a full promotion gate changes
them in a later, explicitly evidenced commit.

## Shared structural gate

Before GPU evaluation, each changed mechanism must have independent forward and
gradient references; honest pp-prop and D-RTRL classification; moving intended
parameters; reset/snapshot/restore and checkpoint compatibility coverage; no
repeated bare Python model loop; `brainstate.random` for randomness; focused and
full regressions; and greater than 90% meaningful coverage for new code.

## Shared pilot gate

The 100-task seed-2108 pilot at 4,096 edges advances only when exact query count
does not fall; pairing is positive or improves by at least 0.005; effort-60
minus effort-30 pixel/rule improves by at least 0.005 or becomes nonnegative;
pixel or rule-at-oracle improves by at least 0.005; no secondary metric falls by
more than 0.01; outputs are finite; allocator use stays below 85% VRAM; and
runtime is at most 1.25 times the matched baseline.

## Shared full gate

Across 419 queries and all three seeds, promotion requires pooled exact
non-inferiority, positive pairing improvement in at least two seeds,
non-regressing effort monotonicity in at least two seeds, no mean secondary
regression larger than 0.01, and a positive 95% lower confidence bound from a
10,000-resample seed-stratified paired bootstrap for at least one of binding,
effort improvement, pixel, or rule-at-oracle. Resource and structural gates
must remain satisfied.

## Evidence contract

Every stage retains one compact JSON manifest containing revision, source
hashes, commands, initial/shared parameter hashes, metrics, paired deltas,
runtime, VRAM, promotion decision, and explicit missing evidence. Synthetic
operator tests, smoke plumbing, pilots, and full ARC qualification remain
separately labeled.
