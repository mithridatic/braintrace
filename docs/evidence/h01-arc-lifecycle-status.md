# H01 ARC integration: measured status, 2026-09-09

The full training/evolution lifecycle is **blocked, not complete**. The actual
multicompartment forward interface is implemented, but pp-prop rejects its
dependency graph. No learning result, ARC score, or physiological qualification
is claimed.

## Source and execution

The implementation includes the complete committed `feat/h01-braincell` source
at `914b32bb11fbee3d09d8fc5fcf8147d01e017a24`. That work and the tested calcium
cache correction are also integrated into main at `469f7ac2afb29277a55c500c5aaee4ab394edc6c`.
The original H01 branch and its dirty worktree are retained. The incomplete ARC
interface remains on `feat/h01-example21-lifecycle`.

Use `h01-population-components-soma.json`, including the four corrected soma
components, rather than the old largest-component manifest. All 104 selected
components imported successfully in 304.37 seconds, covering 2,786,225 source
nodes (`h01-arc-import-current-preflight.json`). Disconnected fragments remain
documented. The two constructible anatomical contacts remain enabled and the
third fragment contact remains blocked.

The forward probes use frozen donor parameters, the local staggered implicit
calcium solver, and 20 cable steps of 0.005 ms per 0.1 ms event. Their driven
input is an all-ones 441-feature vector, **not an encoded ARC episode**.

| Cells | Compartments | Build seconds | Initialize seconds | Native compile/event seconds | Adapter compile/event seconds | Warm event seconds | Peak MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4 | 75,605 | 48.60 | 39.61 | 12.95 | 11.99 | 2.10 | 1,397.67 |
| 12 | 84,097 | 52.71 | 51.98 | 33.12 | 34.04 | 2.28 | 2,109 |
| 40 | 165,092 | 105.37 | 148.07 | 135.18 | 137.30 | 5.54 | 4,635.63 |
| 104 | 807,588 | 700.31 | 469.95 | 280.46 | 194.13 | 13.91 | 14,326.70 |

Each completed probe matched native zero-input soma voltages exactly and had
finite driven soma voltages. The 40-cell probe additionally inspected all 1,127
model states and found no nonfinite values. The 104-cell check passed under its
separately approved 2,700-second wall limit and 16 GiB RSS limit. It inspected
all 2,391 registered states with no nonfinite values, matched native soma voltages
with maximum error 0.0 mV, and measured a 3.0447 mV maximum encoder-driven change.
Its authoritative report is `h01-arc-probe-104.json`. All these probes used two
anatomical contacts and zero synthetic contacts. These are short-window forward
checks, not long rollout or physiological evidence. Phase timings exclude reset
and miscellaneous reporting overhead and should not be summed as total wall time.

## Learning blocker

The 12-cell compiler emits incompatible shape warnings for encoder/contact
dependencies on full cable/channel arrays, then rejects a non-position-preserving
path from `etp_sp_mv` to a synaptic ring buffer. Small actual cable fixtures also
reproduce rejection at both scan unroll limits 16 and 32 (respectively `scan` and
`squeeze` on a path to voltage state). Merely increasing the unroll threshold does
not solve the dependency problem.

`h01-arc-interface-tests.xml` records both expected failures. They are strict
xfails, not passed learning gates. The implementation does not remove cable or
delay state from eligibility, relax compiler guards, or substitute a soma model.
The next prerequisite is compiler/executor support for these heterogeneous cable
dependencies with bounded sparse eligibility allocation, followed by finite-window
`chunked_online_param_gradients` checks.

## Validation and remaining gates

Focused interface/native-step tests: 16 passed, 2 strict expected compiler
failures; 98% combined statement coverage (model 97%, stepper 100%). Probe runner
failure/limit tests: 4 passed. Existing affected H01 import/network/calcium/CLI
checks: 115 passed. Position-graph tests: 41 passed with the installed CSR
primitive CPU backend explicitly set to `jax_raw`; the environment lacks numba.

The calcium correction has a sibling reproduction: a cached dynamic reversal
potential retained a JAX tracer across native execution and adapter tracing.
The fix removes that unused cached quantity and avoids caching traced constants;
the solver equations are unchanged. Future compatibility checks must exercise
reset and retracing, not only a first native execution.

Still unpassed: 0.005/0.0025 ms numerical comparison, finite-window learning,
active input/contact updates and descent, full causal controls, real mutations,
optimizer remapping, H01 checkpoints/recovery, full 104-cell encoded ARC episode
with pp-prop update and restored continuation, and the complete affected Example
21/lifecycle regression gate. The evolution adapter and backend CLI are not yet
implemented. Physiology remains unqualified and ARC scores are unmeasured.

No campaign estimate is defensible from these forward probes: real scoring and
learning costs are unavailable while compilation is blocked. Campaign execution
must wait for those measurements and a separate costed configuration. Do not
interpret a synthetic warm-event time as an ARC training or scoring cost.
