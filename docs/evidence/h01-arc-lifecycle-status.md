# H01 ARC integration: measured status, 2026-09-09

The full training/evolution lifecycle remains **in progress, not complete**.
The new explicit sparse pp-prop path now compiles actual multicompartment state
dependencies. Four source cells completed two finite Muon updates; twelve cells
compiled dependencies but hit the 900-second limit during update compilation.
The original pp-prop path retains its existing guards. Physiological qualification
and the 104-cell encoded ARC learning/continuation gate remain outstanding.

## Learning continuation

`h01-arc-sparse-learning-4-implicit-solve.json` records 75,605 compartments,
44,201,520 eligibility bytes, a 238.28-second cold update and a 51.67-second warm
update. Both losses and gradient norms were finite, all four parameter groups
changed, and peak RSS was 12,720.79 MiB. This is a synthetic all-ones event loss;
contact changes may include weight decay and are not evidence of active synaptic
learning by themselves. Active-contact gradients and controlled descent passed
separate finite-window multicompartment fixture tests.

The twelve-cell attempt is retained in `h01-arc-sparse-learning-12.json` and
`h01-arc-sparse-learning-12.limit.json`: dependency compilation succeeded with
47,664,752 factor bytes; update compilation reached 900.17 seconds overall at
6.95 GiB current RSS. It did not complete a learning update.

Subsequent code adds exact implicit derivatives for the unchanged tree solve,
cable-substep rematerialization, and omission of incoming state values proven
unread by the compiled program. These changes have dense-gradient, forward
equivalence, and finite-window regression tests. Their new population costs
must be measured rather than inferred from the earlier reports.

The current joint-JVP implementation completed the four-cell GPU probe in
`h01-arc-sparse-learning-4-gpu.json`: cold update 177.56 seconds, warm update
56.84 seconds, 44,201,360 eligibility bytes, 4,057.40 MiB peak host RSS,
678,986,240 peak live device bytes. Loss decreased from 0.49625749 to
0.49244804. The cached GPU stack uses JAX/JAXlib 0.11.0 and BrainState 0.5.3;
it is separately pinned from the CPU 0.11.1/0.5.4 stack. This is still a
synthetic single-event learning measurement, not an ARC episode score.

The local training corpus contains 400 tasks and 416 queries. Its shortest
encoded query advances 193 events (705 including padding), for example task
`025d127b`, query 0, source SHA256
`4cd7de8df11d7d9d9a8f118ad73702506cda133641e6b1af17b4a240fddf251c`.
Multiplying the measured four-cell warm synthetic-event cost by 193 gives
about 3.05 hours, before construction and compilation. This is a planning
estimate only: actual request-loss execution has not yet been measured over
that full episode. A source-population ARC learning run therefore needs a
separately costed wall-time approval under the existing 15-minute rule.

The twelve-cell GPU probe also reached its wall limit: see
`h01-arc-sparse-learning-12-gpu.json` and its `.limit.json` sidecar. It completed
construction, initialization, forward comparisons and dependency compilation
(47,663,568 factor bytes, three colors, 362 state blocks). It stopped at
900.29 seconds during first-update compilation, with 5.437 GiB current host
RSS. The incremental report's `running` status is superseded by the sidecar;
there is no completed twelve-cell learning update on this stack.

Current bounded validation: 90 backend tests passed, two legacy-path tests
remained expected failures, and backend coverage was 92% (896 statements).
The source-equation/learning subset passed 57 tests. Additional native delivery
tests passed for two simultaneous arrivals at the retained 0.5 ms delay, both
E and I receptors, and native-versus-adapter execution at both step sizes.
Those checks do **not** establish cross-step numerical agreement: a strong
passive-cable fixture differed by 15.95 mV (E) / 3.68 mV (I) between 0.005 and
0.0025 ms, while matching native execution at each size. Diagnose that result
and run the actual H01 operating-window numerical gate before qualification.

`examples/h01_arc_episode_probe.py` now implements unchanged ARC encoding,
pre/post direct scoring, one ordered durable pp-prop update, exact parameter
and optimizer restoration, repeated scoring, and episode-boundary resume.
Its full compiled fixture test passed in 90.60 seconds, including a second
invocation that retained the original parent and did not repeat the committed
update. This fixture uses real H01 donor dynamics on a small morphology; it
does not satisfy the 104-source-cell integration gate.

The wider oracle/Example 21/coordinator regression run initially had 245 passes
and 20 failures from missing Numba CPU kernels. After the user's authorized
installation of Numba 0.67.0 / llvmlite 0.49.0, all 44 targeted tests passed
(including every previously failing case) in 129.89 seconds with the normal
CPU backend. NumPy 2.5.2, JAX/JAXlib 0.11.1 and BrainState 0.5.4 were retained.

The worktree now also contains H01 topology mutation, versioned asset-verified
checkpoints, actual optimizer restoration/remapping, and an evolution adapter.
Small multicompartment fixtures have executed clone/add/prune, an encoded ARC
episode/update/score, and update/save/restore/clone. Ordered callback recovery
has replayed an interrupted episode from its last durable parent in a compiled
driver test. Full source-population lifecycle validation and coverage are ongoing.

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

## Historical default-path learning blocker

The 12-cell compiler emits incompatible shape warnings for encoder/contact
dependencies on full cable/channel arrays, then rejects a non-position-preserving
path from `etp_sp_mv` to a synaptic ring buffer. Small actual cable fixtures also
reproduce rejection at both scan unroll limits 16 and 32 (respectively `scan` and
`squeeze` on a path to voltage state). Merely increasing the unroll threshold does
not solve the dependency problem.

`h01-arc-interface-tests.xml` records both expected failures. They are strict
xfails, not passed learning gates. The implementation does not remove cable or
delay state from eligibility, relax compiler guards, or substitute a soma model.
The explicit sparse path described above supplies new compiler/executor support
and finite-window checks; it does not alter these default-path rejection tests.

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

Still unpassed at the required source-population scope: 0.005/0.0025 ms numerical
comparison, full causal controls, all mutation/recovery arms, full 104-cell encoded
ARC episode with pp-prop update and restored continuation, and the complete
affected Example 21/lifecycle regression gate. New small-fixture results are
listed above and do not satisfy the 104-cell gate. Physiology remains unqualified;
source-population ARC scores are unmeasured.

No campaign estimate is defensible yet: full encoded source-population scoring
and learning costs remain unmeasured. Campaign execution must wait for those
measurements and a separate costed configuration. Do not
interpret a synthetic warm-event time as an ARC training or scoring cost.
