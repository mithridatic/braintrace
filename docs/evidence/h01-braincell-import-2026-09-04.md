# H01 import verification, 2026-09-04

## Delivered interface

`braintrace.datasets.h01` provides `fetch_h01`, `H01Archive.neuron_ids`,
`H01Archive.components`, and `H01Archive.load`. Loading returns a BrainCell
`Morphology`, original H01 rows, converted SWC, diagnostics and source provenance.
`braintrace.datasets.h01_demo` supplies a passive cell factory and installed
command-line demonstration. The tutorial is `docs/tutorials/h01_braincell.rst`.

All implementation was made on `feat/h01-braincell`, based on `624d498`, in
`.worktrees/h01-braincell`. No Example 21 code, main-branch files or shared
runtime packages were changed. Test/build dependencies and source-inspection
tools were installed only in ignored environments inside the worktree.

## Actual source and physical conversion

Downloaded the official H01 `20210601/proofread_104` SWC ZIP: 59,832,937 bytes,
104 cell identifiers, 3,327 component files. SHA-256:
`3e0534df357ef2e92f6e0199962133cc9bc9733eb3a1fd6d1d315208ad63db47`.

H01 analysis code at revision `db223b8d769d8ae046f0026f12b251103f1dbab0`
documents skeleton voxels of 32 x 32 x 33 nm and the nonstandard annotation
vocabulary. Independently fetched the original binary skeleton for cell
546925828 with CloudVolume and compared all 795 vertices in its component 0:

- All scaled SWC positions matched original skeleton vertices exactly (0 nm).
- Maximum radius discrepancy: 0.0000004980468873 nm, consistent with text
  rounding. Radius values must not be multiplied by the position voxel size.
- The production importer does not depend on CloudVolume or its dependencies.

## Actual full-cell simulation

Command, from the worktree using the existing BrainCell runtime:

```powershell
python -m braintrace.datasets.h01_demo --archive .cache/h01/proofread104.zip --output .cache/h01/full-cell-run.json
```

Cell 810151953 has exactly one archived component, `810151953.0.swc`:

| Measurement | Result |
| --- | --- |
| Source points | 14,215 |
| BrainCell electrical compartments | 3,137 |
| Sum of original physical edge lengths | 4,141.525069743235 um |
| Imported morphology cable length | 4,141.525069743186 um |
| Relative cable-length difference | 1.1859e-14 |
| Simulated duration / timestep | 1 ms / 0.025 ms |
| Compiled steps | 40 |
| Root current | 0.001 nA |
| Minimum / maximum voltage | -64.9740448 / -64.9412460 mV |
| Final voltage | -64.9560089 mV |
| All recorded voltages finite | yes |

This is uniform passive cable dynamics, not a human electrophysiology fit.
BrainCell reports `semantics.no_soma_samples` because simulation branch labels
are deliberately neutral; original H01 annotations are preserved separately.
No measured morphology is intentionally removed. One archived component does
not establish anatomical completeness outside the imaged tissue volume.

## Tests and downstream packaging

```powershell
.cache/validation/Scripts/python.exe -m pytest braintrace/datasets -q -s --cov=braintrace/datasets --cov-report=term-missing --cov-fail-under=90
.cache/validation/Scripts/python.exe -m build --wheel --no-isolation --outdir .cache/dist
```

- 37 tests passed in 6.27 seconds; all 173 executable statements in the new
  importer, converter and demonstration covered (100%).
- Meaningful checks include anisotropic coordinate/radius conversion, H01
  annotations not being mistaken for SWC soma labels, topology preservation,
  missing parents, cycles, singleton components, invalid radii, missing cells,
  explicit component selection, corrupt/changed caches, partial downloads,
  archive path rejection, compiled electrical responses, CLI reports and
  non-finite simulation rejection.
- The changed-cache test first failed, then passed after adding verification
  at component-load time. Initial checksum verification alone was insufficient.
- Built `braintrace-0.2.5-py3-none-any.whl`, installed it into an independent
  target directory, and launched Python with `-I` from outside the checkout's
  package root. Asserted that imports resolved to the wheel installation.
- The wheel contains all four production `braintrace/datasets` modules and
  contains none of their co-located tests or downloaded data.
- The installed CLI imported real H01 component `546925828.0.swc` (795 points,
  173 compartments) and ran 40 compiled steps with finite voltages:
  -64.6634598 to -61.1108131 mV.

On this Windows runtime, coverage configured by dotted module names caused a
native Abseil initialization failure. Specifying the source directory, as above,
avoids premature module imports and passed. The shared runtime had no pytest;
verification used an isolated test environment referencing its packages.

## Qualification boundary

Verified: actual H01 import, physical geometry conversion, real-cell forward
simulation, reusable installed package/API, provenance, and the stated tests.
Not claimed: a synaptic connectome importer, biological compartment inference,
calibrated human channel dynamics, PP-Prop gradients, ARC performance, full
repository regression qualification, GPU performance, main integration, or
public publication. A subsequent PP-Prop model must independently qualify its
multicompartment state ownership and finite-window learning oracle.
