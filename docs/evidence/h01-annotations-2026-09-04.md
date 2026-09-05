# H01 annotation integration evidence

Implemented on `feat/h01-braincell`, extending the earlier geometry importer.
All checks below were run against this worktree. No merge or publication is
implied. The main checkout was unchanged.

## Delivered capabilities

- `H01Component.anatomy()` decodes original labels and exposes original-node,
  soma, and label-based locations plus strict or explicitly inferred regions.
- Strict regions require agreeing endpoint labels. Sample-neighborhood regions
  extend each sample halfway along incident edges, retaining unclassified areas.
- Cell tags and numeric annotations load from checksum-pinned official JSON.
  All 104 metadata IDs match the SWC archive. There are 30 cells tagged both
  L2 and pyramidal. No cell type is inferred from SWC geometry.
- All 264090 CSV rows refer to the same 104 cell IDs. The adapter retains source
  line numbers, pre/post role, and all three coordinates in physical units.
- Synapse projections check cell ID, nearest cable segment, maximum distance,
  and equal-distance ambiguity. Rejections remain in the output in row order.
- Source provenance distinguishes source annotations, adapter inference, and
  absent per-item manual verification. Aggregate E/I counts are never assigned
  to individual CSV rows, which lack E/I and partner-neuron fields.
- Both demonstration modules are installable and independent of Example 21.
  The annotated demo records at a soma-labelled sample and stimulates a measured
  postsynaptic coordinate with a caller-selected current clamp.

## Tests

Executed from the worktree:

```powershell
.cache/validation/Scripts/python.exe -m pytest braintrace/datasets -q -s --cov=braintrace/datasets --cov-report=term-missing --cov-fail-under=90
```

**67 passed in 9.43 seconds; 99.80% coverage (505/506 statements).** Each changed
production module exceeds 90%. Tests are co-located `*_test.py` files.

Tests cover sparse, unknown and absent labels; original node coordinates;
non-root soma selection; conservative region lengths; inferred-region coverage;
source/morphology mismatch; shared branch junctions; ambiguous equidistant sites;
distance rejection; wrong cell IDs; pre/post endpoint choice; anisotropic units;
malformed metadata and CSV rows; corrupt/changed caches; gzip download and
staging; unchanged aggregate counts when CSV counts differ; actual compiled
soma response versus zero input; and realized regional leak conductances.

An integration test failed because BrainCell clones its morphology during
`init_state`. The original Python-identity guard rejected that valid clone.
The fix uses geometry/topology fingerprints. Tests now accept the runtime clone
and reject a changed tree. This was reproduced before the fix. A subsequent
test assertion used an incorrect channel attribute; inspection of the installed
declaration corrected it to `class_name` before the final passing run.

## Real source data and forward runs

The actual download API fetched both official annotation assets into a fresh
cache. The JSON transport is gzip compressed and validated after decompression.
See `h01-annotations-download-2026-09-04.json` for file sizes and selection.

At 2 um projection tolerance:

| Component | Source points | CVs | Post CSV rows | Accepted | Too far |
| --- | ---: | ---: | ---: | ---: | ---: |
| 810151953.0 | 14215 | 3137 | 810 | 795 | 15 |
| 546925828.1 | 10766 | 2163 | 550 | 372 | 178 |

Both have L2/pyramidal/neuron tags. Their soma selections use labelled samples,
not the source root. There are 40 strict dendrite intervals in the first and
25 in the second. The selected postsynaptic distances are respectively
0.04220881798 um (CSV row 70173) and 0.02344239273 um (row 111691).

The second cell's 64-bit CLI run completed 40 compiled steps at dt=0.025 ms,
ending at -65.00087688056799 mV. See the corresponding second-cell JSON.
The first cell's 32-bit run also completed but ended at -65.05143737792969 mV.
It is retained as evidence of why finite output alone is insufficient.

An additional zero-input control for 810151953.0 at 64-bit precision had maximum
resting-potential deviation **0.0024100113507188325 mV**, ending at
**-65.00190206016072 mV**. The exploratory **1e-6 mV equilibrium check failed**.
No strict solver-accuracy qualification is claimed, and the BrainCell solver was
not modified. See `h01-annotations-control64-2026-09-04.json` for the full trace.
The nonzero-input 64-bit run ended at -64.99985310265498 mV, a positive response
of 0.00204895750574 mV relative to the control. The repeat CLI run reproduced
that final voltage; see `h01-annotations-primary64-2026-09-04.json`.
Current and leak values are demonstration assumptions.

Reproduce the annotated CLI with local cached inputs:

```powershell
python -m braintrace.datasets.h01_annotated_demo --archive .cache/h01/proofread104.zip --annotations .cache/h01 --max-distance-um 2 --output .cache/h01/annotated-primary64.json
python -m braintrace.datasets.h01_annotated_demo --archive .cache/h01/proofread104.zip --annotations .cache/h01 --neuron 546925828 --component 1 --max-distance-um 2
```

## Installation check

Built with `python -m build --wheel --no-isolation`, then installed with
`pip install --no-deps --target .cache/annotations-wheel-install`.
An isolated `python -I` process ran outside the checkout import directory and
asserted that module paths resolve to the installed wheel. All seven production
dataset modules matched checkout bytes; no tests or datasets were in the wheel.

That process imported 546925828.0 (795 points, 173 CVs), read the installed
annotation API, and projected 7 of 66 presynaptic rows within 2 um. It selected
CSV row 1024 at 0.05356974486 um distance and completed 40 compiled steps with
finite 64-bit output, ending at -64.66136353307346 mV. This component has no
soma labels, so its recording point was explicitly the source root. See
`h01-annotations-wheel-2026-09-04.json`.

## Limits and follow-up validation

The public source lacks per-item verification, per-synapse E/I type and partner
IDs in this CSV. Cell properties and CSV counts differ and remain separate.
Coincident source coordinates cannot be disambiguated by the anatomy mapper.
The simulator resolves continuous regions onto its selected CV discretization;
subcompartment physiology requires discretization and solver qualification.
No fitted human channels, missing arbor reconstruction, full circuit wiring,
GPU qualification, whole-repository regression run, or PP-Prop gradient/training
qualification was performed. For those uses, test mesh/time-step convergence,
passive equilibrium and transfer responses, and a finite-window learning oracle.

Source data retain H01's CC BY 4.0 attribution in every evidence record.
