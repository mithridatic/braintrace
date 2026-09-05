# Source quality audit before physiological tolerance selection

Use the Allen October 2017 v5 electrophysiology white paper, page 8,
and the public EphysSweep records for specimen 541563728. Evaluate only
calibration sweeps 43 and 50 and repeated-input sweeps 56 through 62.
Do not inspect reserved response features for parameter selection.

Check source-reported pre and post short-window noise below 0.07 mV,
slow noise below 0.5 mV, absolute voltage delta below 1 mV, absolute
bias below 100 pA, and bridge balance below 20 Mohm. Missing metrics
remain unresolved. Retain each value and each decision for each sweep.

The reported input-resistance feature can provide a labeled bridge-ratio
cross-check. It is not a demonstrated substitute for the break-in input
resistance required by the source protocol. Do not claim final source
QC approval from metric checks alone: manual test-pulse review, electrode
zeroing, final drift, and original approval status require separate evidence.
Keep source counts that are null as null; use raw traces for event absence.
