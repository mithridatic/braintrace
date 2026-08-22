# Example 21 Kimi Transfer Stage 2: Memory-Read Cadence

## Status

Approved for implementation on `feat/example21-kimi-transfer`. Promotion is
not evaluated until the preregistered GPU evidence exists.

## Objective

Alternate local recurrent computation with periodic associative-memory reads
without adding a Python model loop or changing the interval-one behavior.

## Configuration

Add `memory_read_interval`, a positive integer with default `1`, to the model
and experiment configurations, CLI, result configuration, architecture report,
and checkpoint compatibility surface. It is meaningful only when associative
memory is enabled.

## Semantics

- Every valid query-input event performs a memory read and resets that lane's
  latent-read index to zero.
- Latent tick numbers are one-based after the last query input. A latent tick
  reads when `tick % memory_read_interval == 0`.
- Interval `1` therefore preserves an every-latent-tick read. Interval `4`
  performs three local ticks then one global read; interval `8` performs seven
  local ticks then one global read.
- `query_only` suppresses all latent reads independent of the interval.
- A local-only tick sends exact-zero raw read and neuron drive. It does not
  overwrite the retained read, drive, or gated-channel diagnostic values.
- Read scheduling is lane-local for packed batches and is implemented inside
  the existing BrainState `for_loop`/`scan` drivers.

## Evidence Interface

Packed and selected packed trajectories expose the exact boolean read mask for
every executed tick and the final per-example read count. Model diagnostics
also expose the interval and final count.

## Structural Tests

- default and interval-one equivalence;
- positive-integer validation;
- query reads and counter reset;
- interval-four and interval-eight masks/counts;
- exact-zero drive and retained diagnostics on local ticks;
- `query_only` negative control;
- independent lanes with different query boundaries;
- reset/snapshot/restore and checkpoint mismatch;
- configuration, CLI, report, result JSON, and architecture round trips;
- JIT and compiled packed/selected drivers.

## Promotion

Compare intervals `1`, `4`, and `8` on the accepted Stage 1 stack. Structural
acceptance does not promote an Example 21 arm without the umbrella pilot and
full reduced-topology gates.
