# Installed-wheel preparation

The [audit](h01-ready-104-wheel-audit.json) passes for the wheel built from
commit `4edb51b`: all 106 packaged files match the committed source archive
and installed target byte for byte. The wheel excludes co-located tests and
test-support modules. All 31 H01 modules import from the new installation
target under Python isolated mode (`-I`), with only that target explicitly
prepended to the existing dependency paths. No source-worktree import fallback
was used. The wheel, source archive and imported module hashes are recorded.

Build used the existing build tools with `--wheel --no-isolation` from a fresh
Git archive. Installation used `--no-index --no-deps --no-compile --target` in
a fresh cache directory. Neither the running simulation's interpreter nor its
installed dependencies were modified. The dependencies remain those of the
validation environment; this is not a clean-machine dependency-install test.

The installed target is `.cache/h01/readiness/wheel-install-4edb51b`.
Repository examples are not shipped in the wheel. The final installed runtime
must execute the example script with isolated library imports pinned to this
target, recording the example and wheel hashes with the run. That full-104
execution is pending. Import success does not qualify runtime or physiology.
