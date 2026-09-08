# H01 resume checkpoint — 2026-09-08

Worktree: .worktrees/h01-braincell, branch feat/h01-braincell. Main untouched.
Goal remains incomplete. User has almost no weekly usage left: avoid side audits,
repeated status turns, duplicate jobs, and unbounded fitting or data downloads.

## Active work — revalidate processes before relying on this snapshot

- Corrected all104 construction: session 3271, wrapper PIDs 29376/32556.
  Plan: docs/evidence/h01-104-corrected-construction-plan.json; cap 2400 seconds.
  Log: .cache/h01/readiness/104-corrected-soma-construction.log.
  Terminal: same prefix with -terminal.json. Build: same prefix with -build.json.
  Automatic comparison session 13397 runs .cache/h01/readiness/check-corrected-104.py.
  Expected decision: docs/evidence/h01-104-corrected-construction-decision.json.
- Historical all104 implicit runtime r2: worker PID 43860, supervisor 11100,
  original session 79693; cap 12000 seconds. Uses ORIGINAL component selections.
  Log/launch: docs/evidence/h01-ready-104-implicit-ei-10ms-r2.log and -launch.json.
  Outputs: .cache/h01/readiness/104-implicit-ei-10ms-r2-build.json and -traces.npz.
  Automatic checker session 59299 runs .cache/h01/readiness/check-original-104-runtime.py.
  Expected decision: docs/evidence/h01-ready-104-implicit-ei-10ms-r2-decision.json.

Do not restart because an observation times out or session handles disappear.
Inspect actual processes, terminal records, logs and output first. Watchers may
fail independently; run the existing check manually against completed outputs
if needed. Preserve failed/killed runs as such, never as a numerical pass.

## Completed work and authoritative evidence

- Component fix df99731: explicit source-supported soma component selection.
  Four corrections: 3470629528 -> 1, 4138580687 -> 2, 4668874666 -> 1, 7196644737 -> 6.
  Source anchors support these choices; omitted anatomy remains substantial.
  Default inventory: docs/evidence/h01-population-components-soma.json.
  Original h01-population-components.json remains for historical reproduction.
- 67 network/CLI tests pass; h01_network.py statement coverage 100%.
- Four corrected cells pass source import, electrical construction, initialization
  and finite 10 ms runtime at dt .000625 ms, implicit solver, 1 nA pulse.
  Decisions: docs/evidence/h01-four-soma-{construction,initialization,runtime}-decision.json.
  Three emit one event, one remains subthreshold. Not human physiology validation.
- Corrected wheel from snapshot afbd6c5: 108 files match source and fresh target;
  isolated installed selection smoke passes. docs/evidence/h01-corrected-wheel-audit.json.
  Wheel/install under .cache/h01/readiness/wheelhouse-afbd6c5 and wheel-install-afbd6c5.
  No installed all104 runtime yet. Examples are not packaged in the wheel.
- Latest committed status before this note: 1125cbb.

## Outstanding boundaries and next steps

1. Inspect/validate corrected full construction when terminal; compare 100 unchanged
   cells against the r2 construction reference and four corrected cells against
   their separate passing construction. Automatic checker implements that comparison.
2. Inspect historical runtime when terminal; finiteness is not physiology and its
   original selections cannot qualify the corrected full population.
3. Corrected full population initialization/runtime, controls, refinement and installed
   execution remain open. Do not call the goal complete from four-cell results.
4. Only TWO source-supported contacts are enabled. Nine cached shards cover 8,991,719
   records, not the whole export. Exact base ownership found 39 same-cell pairs and
   no between-cell pairs there. Membership/placement and remaining 157 shards unresolved.
   h01-cached-full-base-join.json and h01-connectivity-extraction-comparison.md retain evidence.
5. Example21 implementation remains deferred by the approved programme: interface
   specification only. Updated contract covers real cell dynamics and synapse lifecycle.
6. Original physiological acceptance requirements remain in force. A pending user
   question asks whether to separate physiology as research; no answer has authorized
   changing those requirements. Closed physiological campaign caps are not reopened.

Main status: docs/evidence/h01-ready-104-result.md. Full scope:
docs/specs/2026-09-07-h01-population-programme.md and 2026-09-08-h01-104-readiness.md.

## Environment and cleanup

Use login:false for PowerShell; Python .cache/validation/Scripts/python.exe.
Preload braintrace and braincell before pytest/coverage. fastavro is available in
.cache/h01-inspect/Scripts/python.exe. Git mutations may require escalated git tools.
No agents unless explicitly requested. No model stepping via bare Python loops.
Use rg directory -g pattern, not Windows literal wildcard paths.

A 2.53 GiB incomplete packaging archive was removed. docs/evidence still totals
about 9.93 GiB; unique old traces were NOT deleted. Source/cache data retained.
For packaging use a package-only git archive, await completion, then extract/build.
Do not archive the entire evidence-heavy repository. Preserve unrelated untracked
campaign JSON, profiles, 40-cell records and the live r2 launch record.

## New terminal update

Corrected all104 construction is STOPPED: exit124 after 2400 seconds, 95 cells
registered; no complete build artifact. Automatic comparison session13397 ended
with refusal on the nonzero exit. Decision is UNTESTED_TIMEOUT in
h01-104-corrected-construction-decision.json. No matching worker remained at the
post-timeout check. Do not treat the active-construction snapshot above as current.
The older runtime is a separate job; inspect it independently. No retry launched.
