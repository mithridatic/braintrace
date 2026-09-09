# All 104 H01 population components import without segment loss

2026-09-08, `feat/h01-braincell`. Decision: **PASS for anatomical import**.
Source: [per-cell audit](h01-population-import-104.json).

All 104 largest soma-bearing components load through the H01-local reader.
All 2,804,445 directed source segments are present in the imported geometry,
with zero missing or extra segments. The comparison uses directed proximal and
distal physical coordinates rounded to 1e-9 um. The run took 509.239 seconds.
Each cell record contains its component ID and original SWC SHA-256; the report
also records the release archive SHA-256. Disconnected fragments are not joined.

The old reader's relative coordinate tolerance treated voxel-scale separations
at large absolute coordinates as coincident. Two synthetic regression cases
(a leaf and a branching child) failed with the same `from_points()` exception
before the repair. The H01-local specialization now explicitly retains distinct
attachment points when that approximate comparison would suppress them. The
upstream validation pipeline and branch constructor remain in use. This also
protects multi-point child branches from silently losing their first segment.
Installed BrainCell and the unrelated dirty performance worktree are unchanged.

Validation: 97 importer/anatomy/construction/network/init/audit tests passed in
52.55 seconds; `_h01_reader` and `h01` each measured 100% line coverage. After
adding the multi-point child regression, all six translation/branch-shape tests
passed separately. Cases cover leaf, fork, chain, large coordinate translation,
directed segment multiplicity, invalid source geometry and source provenance.
The new audit test initially used a relative import in a non-package test
directory; it was corrected to the repository's absolute evidence import style
and collection was rerun successfully.

Command for the real audit:

```powershell
.cache/validation/Scripts/python.exe -u -m docs.evidence.h01_population_import_audit --archive .cache/h01/proofread104.zip --components docs/evidence/h01-population-components.json --output docs/evidence/h01-population-import-104.json
```

This closes the seven-component reader blocker. It does not establish population
construction, initialized electrical state, finite simulation, firing, functional
inhibition, or physiological qualification. Those remain gates of the full
104-cell readiness objective. The next run is recorded in
`h01-ready-40-build-launch.json`.
