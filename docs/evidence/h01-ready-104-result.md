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

The first driven `ei` 10 ms run is now supervised under the
[registered plan](h01-ready-104-ei-10ms-plan.json). Its conservative estimate is
4,673.661 seconds and 18,262,754,560 peak bytes, with a 6,000-second wall cap,
600-second silence cap and 4 GiB free-memory reserve. This is a pending
measurement; the completed 1 ms result above remains the verified runtime scope.
