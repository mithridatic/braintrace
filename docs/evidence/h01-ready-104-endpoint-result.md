# H01 full-population endpoint correction

The first 104-cell construction attempt failed after 587.575 seconds, exit 1,
on cell 3812058320. This was a placement invariant failure, not a memory limit.
Evidence: [launch record](h01-ready-104-build-launch.json) and
[diagnosis and correction](h01-ready-104-endpoint-decision.json).

The largest-radius soma sample, source node 22316, is the end of branch 2265.
Its computed branch fraction was 1.0000000000000002; the soma interval ended
at exactly 1.0. The cumulative-length and total-length reductions rounded
differently. The strict membership check therefore rejected a source endpoint
that physically belongs to the soma interval.

A fixture using the affected branch's source coordinates reproduced the exact
roundoff error. Canonicalizing the final anatomy-index fraction to 1.0 fixes it
without changing source coordinates, interior fractions, the selected soma,
or the membership check. The real cell then constructs and discretizes to
9,918 compartments with the same L2 donor. Seventy-five anatomy/cell/network
tests pass with 97% anatomy coverage, plus four audit tests that include rejecting
an intentionally misplaced soma probe without moving it.

The failure shows why valid source geometry does not by itself prove electrical
construction. Endpoints must be canonical in both location and interval mappings;
an epsilon-relaxed membership check would hide that inconsistency.

The subsequent audit passed for all 104 unique cells in 478.749 seconds:
all 2,804,445 directed source segments are present, every electrical partition
is valid, and every selected soma lies inside its electrical soma region.
Evidence: [all-cell audit](h01-ready-104-soma-audit.json) and
[launch/completion](h01-ready-104-soma-launch.json), exit 0. Its 1,800-second
wall cap and 600-second silence cap were not reached.

The full build retry is running with the same anatomy, donor mapping, CV policy,
solver and input settings under source commit `b46ee7d`. The retry is recorded
in `h01-ready-104-build-r2-launch.json`; raw output stays in
`.cache/h01/readiness/104-build-r2-build.json`. Full-104 construction, runtime,
matched controls and physiological qualification remain open until measured.
