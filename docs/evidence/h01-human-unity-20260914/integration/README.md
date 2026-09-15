# Clean-main integration record

The user requested all campaign work on main, pushed and synchronized with the
Vast.ai checkout, with clean main and no stale worktrees.

The sole pre-existing local and remote tracked edit was one replacement digest
in `docs/evidence/h01-arc-manifest/manifest.json`. It named
`examples/pp_prop/h01_arc_model.py` as SHA-256
`ed77004c724b5d72ba0cf41e3b2c36823e3880c1b6870e9c4b052d54076d543f`.
The actual committed file hashes to
`2019dfc42be16a12be86731d03b8e04bf702119fa4334e97f5fb10b458010850`,
which is the value in the committed historical manifest.

The edited manifest is preserved byte-for-byte as
[manifest-before-cleanup.json](manifest-before-cleanup.json), SHA-256
`9b5dc72ff9b8d5c3d3a008b4e012f45d3bd0be631574f29f9b7c70a680a17def`.
Both machines had this same edited-file digest before cleanup. Restore the
historical tracked manifest instead of committing an incorrect replacement hash
into it. This does not refresh the historical manifest or qualify a new run.

The mistake in the earlier handoff was leaving this reviewed-but-unresolved edit
behind after the user requested a clean main. Completion requires an empty
`git status --porcelain` locally and remotely, matching main commits on local,
origin, and Vast.ai, and no secondary worktrees. Preserve raw evidence in the
retained cache locations listed in the next-conversation handoff.
