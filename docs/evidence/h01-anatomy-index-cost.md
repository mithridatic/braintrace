# Anatomy construction cost

The original measured-pair run was live in `_geometry_signature` during CV
construction. Its stack entered `MorphoBranch.index`, which calls
`Morphology._branch_index_map` for each lookup. That method rebuilds the full
branch map. The adapter repeated this work across branches and edge endpoints.

The adapter now reads the documented default branch order once per traversal.
It uses the same indices for geometry serialization and source-node mapping.
It still recomputes the geometry fingerprint when a selection is evaluated.
Thus, an edited morphology does not reuse a stale fingerprint.

| Real component | Branches | Original signature, s | New signature, s | Fingerprint and contact location |
| --- | ---: | ---: | ---: | --- |
| I 5584343344.0 | 6272 | 9.599 | 0.214 | Exact equality |
| E 4157825456.0 | 8367 | 17.733 | 0.304 | Exact equality |

These are single-pass timings on a shared host. They do not measure full
simulation speed. The equality checks, not the timings, establish preservation
of the source identity and contact placements. The [raw record](h01-anatomy-index-cost.json)
retains the hashes and locations.

The regression failed before the change because anatomy traversal repeatedly
called the full-map lookup. After the change, 21 anatomy, CV-boundary, and
contact tests pass. Coverage is 239 of 240 statements across those three
modules. Geometry rejection remains covered.

The updated real-pair run advanced to BrainCell's CV mechanism construction.
A subsequent stack sample was inside `_build_frusta`, through
`build_cv_mechanisms` and the output-site mesh query. This establishes progress
past the earlier adapter work. It does not establish completed construction,
spiking, or inhibitory event delivery. The [launch record](h01-measured-ie-index-fixed-launch.json)
states the unchanged physical settings and the two run handles.
