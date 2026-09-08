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
coverage. Next is an exact isolated-cell reproduction and one dt-halving
comparison, each capped at 600 seconds. The failed identity stays in the full
population; a corrected full-104 run remains required. The completed 1 ms
result above remains the verified runtime scope.

The [installed-wheel preparation](h01-ready-104-wheel-audit.md) passes source,
installation and isolated-import checks (106 packaged files, 31 H01 modules).
The final full-population execution with imports pinned to that wheel remains
pending; source-run success does not substitute for it.
