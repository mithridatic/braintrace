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

## Installed-library regression tests

The [installed-wheel test decision](h01-ready-104-wheel-tests.json) passes:
51 tests in 2.85 seconds, covering importer provenance, malformed source input,
short attachments, branching/translation invariants, anatomy selection and the
soma endpoint repair. Coverage is 100% for `h01` and `_h01_reader`, and 97.46%
for `h01_anatomy`. All 86 loaded production modules resolve to the installed
target. The source root was appended only for test fixtures after importing
the installed package; production origins were checked again after testing.

The first attempt passed 27 selected tests but failed coverage because H01
imports preceded measurement and the selection omitted validation tests.
The second attempt started coverage before the dependency preload and hit the
known Windows Abseil startup abort before testing. The passing invocation
preloads `braintrace` and `braincell`, starts coverage, then imports H01 modules
and runs the complete selected suite. Preserve this order and verify modules
are not already loaded before starting their coverage. Logs and report hashes
are retained in the decision. This is a targeted installed-library check;
neither the full suite nor full-population wheel execution is implied.

## Launcher for the final installed runs

`docs/evidence/h01_installed_network.py` verifies the wheel hash and installed
payload, requires Python isolated mode, rejects source-import contamination,
and records example/library provenance before and after execution. Its 23
tests pass with 100% coverage, including real subprocess failure cases. The
actual example's help invocation also passes; its
[provenance record](h01-ready-104-wheel-cli-provenance.json) is explicitly a
CLI check, not a runtime result.

Use the existing supervisor with this command shape for the remaining runs:

```powershell
.cache/validation/Scripts/python.exe -I docs/evidence/h01_installed_network.py `
  --audit docs/evidence/h01-ready-104-wheel-audit.json `
  --example examples/h01_verified_network.py `
  --provenance <new-run-provenance.json> -- <registered-example-arguments>
```

Each real run needs a fresh provenance path and its registered limits.
Successful launcher completion must still pass the independent runtime and
applicable control/refinement gates.
