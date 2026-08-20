# Non-compiler JAX open-jaxpr input compatibility

## Problem

JAX 0.11 represents the external values of an unattached open jaxpr in the
same `invars` sequence as its runtime inputs. BrainTrace's compiler already
records the logical external-input prefix separately, but the VJP algorithm's
pp-prop safety check passed every jaxpr input to the position-preservation
proof as if it were a hidden position.

For a normal recurrent matrix operation, that makes the proof inspect the
external input and weight alongside the hidden output. It consequently rejects
valid standard `tanh_rnn` models with a false "mixes hidden positions"
failure before the algorithm can compile. The same input conflation makes
SnAp-n's position analysis widen valid sparse graphs to all-to-all.

The isolated configurable sparse benchmark has a second boundary of the same
kind: its worker loads an example by file path in a child process, where the
checkout root is not automatically on `sys.path`. The worker must establish
that import boundary explicitly before loading the example.

## Contract

The pp-prop position-preservation proof must trace reachability from the
runtime `y` variable produced by the ETP primitive. SnAp-n must likewise seed
its position analysis from the compiled group's `hidden_invars`. External
values remain available to the transition jaxpr but are not hidden positions
and must not be used as proof seeds.

The isolated sparse benchmark worker must add the repository root to its import
path before executing the file-based learning example.

The graph change is input-seed metadata only. It does not alter the compiled
graph's transition mathematics, recurrence scope, or approximation semantics.

## Acceptance gates

1. A standard `tanh_rnn` pp-prop model compiles successfully on JAX 0.11.
2. A sparse ring retains its exact SnAp-n neighbourhood widths and does not
   become conservatively all-to-all.
3. The isolated sparse benchmark's tiny CPU worker completes and honors an
   inherited-platform CPU pin.
4. Existing rejection tests for mixing and non-position-preserving tails still
   reject those models.
5. The focused algorithm regressions pass.
6. The default non-compiler test gate is run; any remaining failures are
   reported separately from compiler failures.
