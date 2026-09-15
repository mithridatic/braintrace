# Human paired synaptic-response input

Approved scope: the all-104 human-only qualification plan, including synaptic
parameters and human response validation. This work supplies a recorded paired
response for that gate; it does not promote a population score.

The fresh TONOHA version 1.0 inventory contains raw human paired recordings but
no Figure3 directory for the published nucleated-patch channel measurements.
Do not call these files new sodium kinetic data or import the published mixed
species model. Use the available synaptic responses on their actual scope.

Acquire the pinned release metadata, article XML, Figure2 summary CSV (359666),
and the first listed human pair recording 2015_11_04_0076.abf (359652). The CSV
explicitly identifies Human, EXC, presynaptic IN0 and postsynaptic IN2, date
20151104, slice 3, cluster 6. This choice precedes response inspection. The file
is 36,007,936 bytes. Cap total downloaded payload at 40 MB, local acquisition
at 120 seconds; no simulation, remote compute or other raw recordings.

Verify release checksums and exact byte counts, retaining original bytes and
SHA256. Read the ABF using the installed pyabf parser. First inspect its channel
names, units, sweep count, clock and protocol. Require the explicit CSV channels
and voltage units before using them. Preserve protocol ambiguities; no channel
selection based on visually convenient responses.

Inspect full traces before narrowing to events. For the first, second and last
sweeps, retain every original sample around each presynaptic upward 0 mV crossing
from -10 to +50 ms if fully covered. Preserve both recorded voltages, absolute
within-sweep time, source sample indices, and applied commands when the parser
can reconstruct them. Reconstructed commands are not measured synaptic currents.
Do not filter, average, resample, baseline-replace or derive a conductance from
the voltage response alone. Keep overlapping windows and failed/no-response
events visible. Produce full-sweep and paired-event plots, and inspect both.

No fitting or held-out validation in this stage. Published EPSP amplitudes and
decay constants remain summary targets with an unreproduced analysis operator;
do not use an EPSP decay constant as a synaptic conductance time constant. Verify
the retained event arrays independently against a second read of source ABF
samples. Additional raw pairs, synaptic fitting and model-response comparison
require a prospectively defined calibration/validation split and observation
operator. Existing unopened L1 holdouts remain untouched.
