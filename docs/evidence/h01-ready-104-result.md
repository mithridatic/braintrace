# H01 all-104 readiness continuation

**Construction, initialization and 1 ms finite runtime PASS: all 104 cells,
808,495 compartments, two supported projections.** Driven activity, controls,
numerical refinement and physiological qualification remain open.

The corrected full build completed in 1,034.034 seconds (1,045.012 seconds process
wall time), exit 0. The independent construction gate verifies exact population
membership, per-cell component and source/archive hashes, donor-record consistency,
positive compartment counts and total, and the supported contacts' endpoints and
Dale signs. Evidence: [gate decision](h01-ready-104-build-r2-decision.json),
[compact build summary](h01-ready-104-build-r2-summary.json), and
[launch/completion](h01-ready-104-build-r2-launch.json). Model source: `b46ee7d`.
The complete raw interval arrays remain at the hash-linked path in the summary.

The preceding [all-cell geometry and soma audit](h01-ready-104-soma-audit.json)
also passes: all 2,804,445 directed source segments preserved, valid electrical
partitions, and strict soma membership for every cell. No identity was removed
to get the full build to pass, and no disconnected fragments were joined.

## Memory measurement limitation

The build supervisor's peak counter is invalid: PowerShell cached the Int32
`Math.Max` overload after small samples, then rejected values above 2 GiB.
Do not use `peak_tree_rss_bytes` from the build launch record. Independent live
worker observations reached at least 4,237,889,536 bytes; that is a lower bound,
not a peak. The free-memory reserve checks continued independently. A regression
using a 1 MiB sample followed by 5 GiB reproduced the failure, and explicit
Int64 casts on both arguments and the initial value pass that regression.

## Full-population initialization and runtime

The [runtime gate](h01-ready-104-ei-1ms-decision.json) passes against the
hash-verified construction reference. All 104 identities, cell settings,
provenance, donors and compartments match. All 317 expected arrays are present
and finite, including soma voltage, output voltage and events for every cell,
and conductance/local voltage for both receptors. There are 200 end-of-step
samples at dt 0.005 ms. No events occurred; both conductances are exactly zero.
Soma voltages range from -84.036919 to -79.279315 mV across the window.

Measured costs: construction 1,126.407 seconds, initialization 836.600 seconds,
compilation plus stepping 266.922 seconds. The process completed with exit 0
in 2,271.368 seconds, within its 3,600-second cap. In-process peak resident
memory was 13,930.617 MiB (13.60 GiB); the corrected Int64 supervisor recorded
zero monitor errors. Its sampled tree peak is lower than the process-lifetime
peak, as expected when samples miss a transient allocation. Evidence:
[compact summary and trace audit](h01-ready-104-ei-1ms-summary.json),
[terminal launch record](h01-ready-104-ei-1ms-launch.json). Raw build metadata
and traces remain at their hash-linked `.cache/h01/readiness/` paths.

This pre-stimulus smoke ends before the assumed 1 nA pulse at 2-5 ms. It does
not test driven firing or synaptic delivery. Next are the driven 10 ms run,
all four matched controls at 104 cells and the dt-halving comparison. The
outstanding physiological gates remain separate. The full goal is not complete.

The first driven `ei` 10 ms run **failed**, exit 1 after 2,704.751 seconds:
`cell_7196644737/output_voltage` became nonfinite. This isolated cell has
1,647 compartments. All 104 cells initialized; the resource limits were not
hit and the supervisor recorded zero errors. The runner raised at the first
nonfinite trace before saving any trace arrays, so first divergence time and
the status of the remaining arrays are unknown. See the
[failure decision](h01-ready-104-ei-10ms-decision.json) and
[terminal launch record](h01-ready-104-ei-10ms-launch.json).

The runner now retains all returned arrays, records every nonfinite array and
its first affected sample, and exits nonzero. Initialization metadata is
persisted before compilation. Twelve regression/CLI tests pass with 100%
coverage. The [exact isolated-cell reproduction and dt-halving comparison](h01-ready-cell7196644737-result.md)
both fail: first nonfinite voltage at 3.2700/3.2675 ms for dt 0.005/0.0025 ms,
with peak finite voltage about 383.8 mV. Complete cell metadata matches the
full construction reference in both runs. The next diagnosis is the first
divergent state and its mechanism; the tested timestep reduction is insufficient.
The failed identity stays in the full population, and a corrected full-104 run
remains required. The completed 1 ms result remains the verified runtime scope.

The [installed-wheel preparation](h01-ready-104-wheel-audit.md) passes source,
installation and isolated-import checks (106 packaged files, 31 H01 modules).
The final full-population execution with imports pinned to that wheel remains
pending; source-run success does not substitute for it.

## Implicit calcium continuation

The [isolated diagnosis and repair](h01-ready-cell7196644737-result.md) now
identifies the negative-calcium numerical failure and verifies finite 10 ms
runs using an opt-in implicit calcium solve. The finer .000625/.0003125 ms
comparison passes the isolated voltage and event timing limits. The full104
[implicit driven run](h01-ready-104-implicit-ei-10ms-plan.json) is registered at
.000625 ms with a 12000-second cap; its construction and runtime results remain
pending. Its solver, morphology, donor and input provenance must pass both the
new solver-transition comparison and the existing source-construction gate.

The preceding installed wheel predates the numerical repair. Final packaging
requires a fresh wheel and full-population execution of its actual installed
code after the remaining runtime, control, refinement and physiological gates.

A fresh [implicit-solver wheel](h01-ready-104-implicit-wheel-audit.md) from
526d57f now passes payload/origin checks and 33 installed regression tests.
This is package preparation only; the full104 source process is still running
and a full104 installed run remains required.

The implicit-solver full104 construction now passes the
[source construction gate](h01-ready-104-implicit-build-decision.json) and
[solver-transition gate](h01-ready-104-implicit-solver-decision.json). It contains
808,495 compartments and both supported projections; every cell field other than
the named solver exactly matches the prior full104 reference. Construction took
1134.833 seconds. The immutable construction checkpoint was copied before the
runner overwrites its working metadata during initialization/runtime; both gate
verdicts bind that saved reference by hash. The
[compact summary](h01-ready-104-implicit-build-summary.json) retains all identities,
donors, compartment counts and contact settings. Initialization is in progress;
no driven runtime or physiological qualification is claimed.

The [implicit-solver initialization checkpoint](h01-ready-104-implicit-init-decision.json)
now verifies all104 population identities and exact construction metadata, with
808,495 compartments. Initialization took 925.829 seconds; process peak resident
memory through initialization was 8722.531 MiB (8.52 GiB). The checkpoint was
preserved before compilation. The same process has entered compilation and
stepping; no driven runtime verdict is available yet.


The first implicit full104 driven attempt terminated at 5639.421 seconds
with a supervisor status-file sharing violation, which killed the worker.
Worker PIDs were checked absent; no runtime verdict was produced. Prior
construction and initialization checkpoints remain valid. A fresh r2 attempt
uses identical numerical inputs and caps, with atomic status publication that
tolerates a locked reader. A real Windows file-lock regression passes creation,
failed replacement with intact old JSON, recovery after release, and cleanup.
The first helper test exposed PowerShell converting a null backup path to an
empty string; explicit NullString fixes that interop issue. Future monitoring
changes must exercise real file handles before an expensive run.

The r2 retry completed all104 initialization in 853.693 seconds after
1230.686 seconds construction, retaining 808495 compartments. Its immutable
checkpoint and exact construction comparison are recorded in
h01-ready-104-implicit-r2-init-decision.json. The worker is now in the combined
compile/run stage; no runtime or physiology pass follows from initialization.
