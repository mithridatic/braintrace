# Missing human sodium QC recovered from the original release

The version 1.0 Figure 3 workbook contains the `exclude` field and nine other
columns absent from version 3.0. All 34 recordings join uniquely by filename;
all 16,830 cells in their 495 shared columns agree exactly. The nineteen human
rows all have a source `exclude` value of zero. This replaces a missing-input
assumption with original author metadata. It does not exclude any human record,
change a fit or qualify a donor.

Recovered columns: `Cp`, `Vhalf_ia`, `artefact_subtraction`, `exclude`, `gof_ia`,
`swpnr_start`, `swpnr_end`, `swpnr_sylguard`, `swpnr_sylguard_end`, and `wholeCell`.
[Comparison and all nineteen human joins](comparison.json) retain original
Excel row numbers and every recovered value, including missing values. These
are source fields, not independently verified clamp quality or delivered voltage.
Other fit-quality, compensation and voltage restrictions in the author's plotting
code still apply; `exclude=0` alone does not establish usable kinetics.

The recovered [180,676-byte workbook](318532-data_fig3.xlsx) is Dataverse file
318532, SHA1 b76026cf9e5584d4b216413f2047cd7e77abc8ad and SHA256
58512456c32c5ec29e1c403484714f07d15918dc709bc088f58127caaec9f4ca.
The current workbook remains untouched. See the [frozen version history](versions.json)
and [acquisition receipts](acquisition-r2.json). The entire six-file acquisition
is 770,746 bytes, below the registered 1 MB cap; each request completed within
30 seconds. The [registration](../../../specs/2026-09-14-h01-human-channel-history.md)
identifies the initial scope and subsequent one-file AP-workbook check.

The historical sodium AP-clamp workbook has the same 49 identities, columns and
cell values as the current one ([comparison](ap-workbook-comparison.json)). It
does not recover missing stimulus/sweep mapping or raw currents. The nineteen
kinetic-record QC flags cannot be assigned to the different AP-clamp records.
The previously rejected thermal-scaling result therefore remains rejected.

The historical human Na/K fitting scripts and channel_kinetics.py are byte-identical
to the current scripts. Changes in channel_tools.py are limited to K_IV, Na_IV
and Na_recovery: paths/species-column conventions and the displayed potassium
inactivation voltage mask changed. K_recovery and APclamp are unchanged. Historical
code does not resolve the potassium conditioning-command disagreement. All source
scripts were read as text and parsed as syntax; none were imported or executed.

The complete public history has versions 1.0 (126 files), 2.0 (65) and 3.0 (65).
None lists NWB/ABF/HDF5/MAT traces, archive files or the referenced Experimental
means directory. The extra original-release entries are largely compiled/generated
model files. This is a specific inventory result, not proof that original recordings
do not exist elsewhere. Do not repeat this release-history search unchanged.

## Consequence for donor qualification

The missing Figure 3 exclusion field is no longer a blocker. Future human sodium
kinetic comparisons can join its original QC and sweep ranges to unchanged
current measurements. Original currents, their extraction/QC details and the
34 C AP-clamp correspondence remain unresolved. The [unsent data request](../human-channel-data-request.md)
has been corrected to ask only for remaining inputs.

The process error was treating the latest release inventory as the whole public
source history before pursuing more fits. Checking original versioned records
recovered an actual dependency. Preserve this correction and use the recovered
fields; do not report another absent-exclude blocker or silently set missing QC
to false. No six-term score changes, new model runs or holdout accesses occurred.
