# Historical human channel inputs: resolve missing QC and protocol provenance

Approved parent: all-six-term human-only donor qualification. The previous turn
completed a rejected second-connection experiment; that result closes strength-only
synaptic transfer and does not improve a population term. Return to the donor gate.

The current public Dataverse history for DOI 10.34894/L5J0SD exposes version 1.0
with 126 files and versions 2.0/3.0 with 65 each. Version 1.0 includes a different
Figure 3 workbook (318532, 180676 bytes), and historical fitting scripts remain
public. The currently retained version 3.0 lacks the exclude field referenced by
the sodium plotting code. Check whether the original version contains the author
QC or protocol information, without presuming that older values are preferable.

Acquire at most the old Figure 3 workbook and the old channel_tools.py,
channel_kinetics.py, and human Na/K fitting scripts: IDs 318532, 318547, 318550,
318548, 318546. Cap total file bytes at 500000 and per-request time at 30 seconds.
Pin the version manifests, exact published checksums and local SHA256. Inspect
source as text only. Never execute downloaded scripts or compiled objects.

Compare workbook sheets/columns and record identities with retained version 3.0.
Any recovered QC must join uniquely by original recording identity; disagreement
remains explicit and cannot silently exclude failed predictions. Compare historical
protocol code with the current source to determine what changed. Do not change
the existing calibration/validation roles, model parameters, scores or failed
decisions. Original currents remain missing unless actual source arrays are found.

The purpose is to resolve a concrete missing input for human Na/K qualification
before spending compute on a donor replacement. If history supplies no usable
missing information, preserve that negative finding and do not repeat this search.

## Follow-up after recovering the missing fields

The first acquisition recovered all ten removed Figure 3 fields, including
exclude and source sweep ranges. All 34 recordings join uniquely and every one
of the 16830 shared cells agrees exactly across versions. Before further model
fitting, check the historical Figure 2 sodium workbook (318529, 544224 bytes)
for analogous removed AP-clamp command/sweep/QC metadata. Raise this audit's total
file-byte cap to 1000000; retain the same 30-second request cap. This is one
additional source file, not an optimization or a change to failed decisions.
Record every changed value; do not favor old values or silently drop observations.
