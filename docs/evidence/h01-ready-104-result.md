# H01 all-104 readiness continuation

**Construction PASS: all 104 cells, 808,495 compartments, two supported projections.**
Initialization and full-population simulation remain unverified at this checkpoint.

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

## Next execution stage

The supervised 104-cell init/1 ms smoke job is live in
`h01-ready-104-ei-1ms-launch.json`. The existing runner initializes each cell
with timing and heartbeat, then executes 200 compiled steps at dt 0.005 ms.
The [registered plan](h01-ready-104-ei-1ms-plan.json) projects 3,058 seconds total
and 22,641 MiB peak memory from measured 40-cell costs and the full build time.
Those are estimates. Limits: 3,600 seconds wall, 600 seconds silence, and 4 GiB
free host memory. The peak counter now uses Int64.

This is a pre-stimulus smoke: the assumed 1 nA pulse starts at 2 ms. Completion
will require auditing all 104 saved traces. Driven activity, four matched
controls at 104 cells, numerical refinement and the outstanding physiological
qualification remain open. The already verified 40-cell 1 ms runtime is a
smaller baseline, not a substitute for those gates. The full goal is not complete.
