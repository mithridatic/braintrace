# Human PAX6 tail-current constraint

Approved scope: the human-only donor qualification plan approved with
"Execute the plan." This extends calibration voltage constraints after the
initial-state-only candidate failed. It does not reopen that fitting family.

Use only already acquired human specimen 840043506, session 840043481,
NWB SHA256 30abbb3cca63b0629242c7ad96f595d6e6ceea2ce7e522fdb5ebf8a3bc2ac944.
Read calibration sweeps 99-103 with the existing pinned exporter. Their commands
were previously inspected; their responses have not yet been analyzed. The
external validation session 835648738 and all whole-cell holdouts stay sealed.

The missing constraint is the current following a common 1000 ms +70 mV command
at five commands spanning approximately -90 to -150 mV. Preserve complete raw
current, DAC, reconstructed command, original clock, metadata and instrument
settings. Verify every command segment against the prior command-only inventory
before computing observations. The command is not measured patch voltage and
total amplifier current is not isolated potassium current.

Use the fixed 1000-1090 ms pre-depolarization baseline, retaining its original
samples and offset. Show the early control at 35-70 ms, the complete +70 mV
phase, and the common first 500 ms of tail. Plot full-scale onset separately
from 5-500 ms detail, preserving every plotted sample. Also retain the complete
post-2600 ms phase, identifying that sweep 99 has no command change there.
Record exact samples at tail phases 5, 20, 60, 200 and 490 ms. Report raw current,
initial-baseline centered current, and a distinct descriptive difference from
the tail's 450-490 ms mean. That last subtraction is not an ionic leak correction.

Inspect whether a common conditioning response is consistent across sweeps,
whether tail current changes sign within the observed commands, and whether
late-reference differences decay or develop over time. Do not infer reversal
potential from extrapolation or assign molecular channel identities. If currents
do not support a stationary common tail family, do not fit a deactivation curve
to manufacture one. The result must specify the resulting model-development
decision and retain alternative explanations, including acquisition order and
clamp/compensation effects. No new parameters or population scores are promoted.

Use local CPU extraction and array arithmetic, capped at 120 seconds; no membrane
simulation or remote compute. Reuse the tested source exporter, independently
check all 450000 response samples and DAC conversions against original HDF5 data,
verify source times, segment equality and every observation index, and run its
sibling tests. Fail on a source mismatch, unexpected shape, missing window,
nonfinite sample, off-grid landmark or changed command. Open generated plots
before interpreting them. Write only to a new evidence prefix and seal artifacts.
