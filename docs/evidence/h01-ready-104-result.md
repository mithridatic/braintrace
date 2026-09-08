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

## Population soma-component selection finding

A source-only check of all 104 selected components found four cells whose
largest soma-labelled source radius lies in another component. Evidence:
h01-population-stimulus-source-inventory.json (construction hash and exact
selected member source hashes checked). The contrasts are 7196644737: component
0 radius 0.161818 um versus component 6 radius 3.980136 um; 4138580687: component
0 radius 0.122152 um versus component 2 radius 6.788244 um; 4668874666: selected
radius 0.759771 um versus component 1 radius 3.292435 um; 3470629528: selected
radius 1.64 um versus component 1 radius 9.534612 um.

The largest-node component policy does not ensure selection of the largest
labelled soma. All cells receive a 1 nA point clamp at the selected component's
soma location. These source radii are not CV areas or measured membrane current
densities. The finding questions anatomical representativeness of four selected
components, including the isolated calcium-failure cell; it does not undo the
numerical diagnosis or prove that radius alone selects the correct cell body.
The running r2 remains an unchanged numerical test of its recorded components.
Before another population launch, resolve these four selections against source
cell-body evidence and account for omitted branches. Do not join disconnected
components or silently replace them in the existing run's reference.

Explicit component-selection support is implemented and the example now defaults
to h01-population-components-soma.json. The original inventory remains unchanged
for reproduction. The regression first demonstrated selection being ignored;
the corrected network and CLI suites pass 67 tests, with 100% statement coverage
of h01_network.py. Invalid selections, loaded hash/component/node mismatches,
legacy behavior, and contact-selected components are covered. Four actual source
morphologies also load with matching hashes, node counts and valid soma locations
(h01-soma-components-load-check.json; 10.617 seconds). These checks do not perform
electrical construction or simulation. Corrected full-population readiness stays
open. The mistake was treating a soma label on the largest component as sufficient
cell-body evidence; future selection overrides carry independent source evidence
and loaded-source checks rather than relying on node count alone.

The corrected four-cell electrical construction passes: 12563 compartments
(7196644737: 1691; 4138580687: 1510; 4668874666: 3310; 3470629528: 6052),
23.932 seconds construction and 28.964 seconds total subprocess time. The
hash-bound h01-four-soma-construction-decision.json checks all four identities,
selected members and source hashes, unchanged donor parameters, implicit solver,
10 um CV policy, 1 nA inputs, positive CV counts and output-site registration.
No initialization or simulation was performed. This removes the corrected
components' electrical-construction uncertainty; it is not all104 construction
or runtime qualification and does not restore their omitted source branches.

The four corrected components also pass initialization: 22.805 seconds,
507.934 MiB peak interpreter RSS, all four identities present. Exact model
metadata matches their construction reference (apart from execution/timing).
Evidence: h01-four-soma-initialization-decision.json. No compilation or stepping
was performed; the corrected population still needs runtime evidence.

The four corrected components pass the bounded 10 ms runtime at dt .000625 ms:
all saved arrays finite over 16000 steps and model metadata exactly matches the
construction reference. Cells 7196644737, 4138580687 and 4668874666 each emit one
event (soma maxima 48.182, 48.713 and 46.923 mV); 3470629528 emits none and peaks
at -59.093 mV. Evidence: h01-four-soma-runtime-decision.json. The subprocess
completed in 365.704 seconds, below its 600-second cap. These results do not
establish human recording accuracy, dt refinement, or all104 runtime. They
support the corrected components' basic driven execution before a population run.

The corrected package wheel from Git snapshot afbd6c5 passes byte-for-byte
comparison of all108 package files against the source snapshot and a fresh
offline target installation. An isolated-mode installed import also honors an
explicit component selection (h01-corrected-wheel-audit.json). This is package
preparation, not installed all104 runtime. Packaging uses a package-only Git
archive: a full-repository archive unnecessarily included gigabytes of evidence
and was stopped and removed. Await archive completion before extraction.

Corrected all104 construction reached its 2400-second cap after registering
95 cells, while discretizing cell 620880207. No complete build artifact was
produced. h01-104-corrected-construction-decision.json records UNTESTED_TIMEOUT,
not numerical failure. A post-timeout process check found no remaining worker
with this exact output prefix. The automatic comparison correctly refused to
pass the nonzero terminal result. About 30.44 GiB was available near the end;
observed delay is not explained by low available RAM alone. No retry launched.
