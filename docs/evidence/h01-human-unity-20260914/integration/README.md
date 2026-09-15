# Clean-main integration record: canonical LF digest retained

The user requested all campaign work on main, pushed and synchronized with the
Vast.ai checkout, with clean main and no stale worktrees.

The sole pre-existing local and remote tracked edit was one replacement digest
in `docs/evidence/h01-arc-manifest/manifest.json`. It named
`examples/pp_prop/h01_arc_model.py` as SHA-256
`ed77004c724b5d72ba0cf41e3b2c36823e3880c1b6870e9c4b052d54076d543f`.
The old Windows CRLF checkout hashes to
`2019dfc42be16a12be86731d03b8e04bf702119fa4334e97f5fb10b458010850`,
which is the value in the previous historical manifest. The canonical Git blob,
the LF-normalized Windows file, and the actual Vast.ai file all hash to the
replacement `ed77004c...` value. There is no model-content difference between
these two hashes. The pre-existing edit correctly records canonical LF bytes.

The edited manifest is preserved byte-for-byte as
[manifest-before-cleanup.json](manifest-before-cleanup.json), SHA-256
`9b5dc72ff9b8d5c3d3a008b4e012f45d3bd0be631574f29f9b7c70a680a17def`.
Both machines had this same edited-file digest before cleanup. Retain this edit
in the tracked manifest and normalize the local model checkout to canonical LF
bytes. This single digest correction does not refresh other historical manifest
entries or qualify a new run.

The earlier cleanup diagnosis incorrectly called the replacement digest stale
after comparing only Windows checkout bytes. Checking the Git blob and remote
bytes identified CRLF versus LF as the cause; check both before classifying any
future cross-platform hash discrepancy. Leaving the edit unresolved also failed
the user's clean-main request. Completion requires an empty
`git status --porcelain` locally and remotely, matching main commits on local,
origin, and Vast.ai, and no secondary worktrees. Preserve raw evidence in the
retained cache locations listed in the next-conversation handoff.
