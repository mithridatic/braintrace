# Original human excitatory paired response recovered

The prespecified human pyramidal-to-fast-spiking-interneuron pair is now available
as an original 36,007,936-byte ABF recording, with the published SHA1 verified and
a retained SHA256. It supplies simultaneous human pre/post voltage observations
for synaptic-response qualification. No model parameters or population scores
changed. The raw source is retained in the worktree cache at the exact path in
[acquisition.json](acquisition.json); the selected full traces and event arrays
are versioned here.

## Source and identity

The [Wilbers fast-spiking study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10569701/)
and [TONOHA version 1.0 release](https://doi.org/10.34894/TONOHA) are distinct from
the pyramidal channel study. The full [release metadata](release.json) has 190
files and no Figure3 directory. Its raw paired recordings do not fill the missing
sodium kinetic-current input. The released model uses mixed-species fitting
targets and artificial morphology; it is not installed as a qualified donor.

The [original CSV](Data_fig2_Pyr_FS.csv) identifies 2015_11_04_0076.abf as Human,
EXC, date 20151104, slice 3, cluster 6, presynaptic IN0 and postsynaptic IN2.
ABF labels are IN 0, IN 2 and IN 3, all in mV. Matching ignores whitespace only;
the postsynaptic channel is index 1, not index 2. Recorded date agrees with the
selected table row. This is one paired recording, not 30 independent donors.
Other table rows have apparent date/filename differences and were not imported
or silently repaired. Their identity requires separate verification.

The [recording metadata](recording.json) reports 30 sweeps, each four seconds and
200,000 samples at 50 kHz, protocol Stim5Hz_0to123. Raw voltage samples retain
their source reference. Source-author EPSP features for this pair include amplitude
1.010925764 mV and decay time 14.91300182 ms, but their original feature extraction
has not been reproduced. EPSP decay is not a measured synaptic conductance time
constant. Parser-reconstructed commands are retained with an explicit unverified
electrode-mapping flag; they are not measured synaptic currents.

## Direct observations

[Full sweeps 0, 1 and 29](full-sweeps.png) were opened before event extraction.
Each shows five presynaptic spikes separated by about 200 ms, then a sixth after
about 503 ms. The postsynaptic voltage contains spontaneous deflections outside
those events and a negative excursion near 3.1 s. These observations remain in
[full-sweeps.npz](full-sweeps.npz); no cause is assigned to the negative excursion.

All 18 complete -10 to +50 ms paired windows are retained at original resolution
in [events.npz](events.npz), with crossing identities in [events.json](events.json).
No boundary event was omitted. The [paired views](paired-events.png) show the
first, second and last event in each selected sweep and were visually inspected.
Postsynaptic depolarization follows the presynaptic spike, with different sizes
across events and sweeps. In sweep 1's first event, a large postsynaptic deflection
begins before the presynaptic spike; attributing that entire waveform to the
evoked synapse would be incorrect. Sweep 0's first event also has an additional
deflection about 20 ms later. Sweep 29's displayed responses are smaller than
several earlier responses. These examples do not establish monotonic rundown,
short-term depression or a release mechanism from three sweeps.

[Direct source samples](direct-samples.json) retain offsets -5, 0, 2, 5 and 20 ms
for the nine plotted events. For example, sweep 0/event 0 is -70.153809 mV at the
presynaptic crossing and -69.110107 mV at +5 ms. Sweep 29/event 0 is -70.672607
and -70.007324 mV at those offsets. These are individual voltages, not fitted
amplitudes, peak estimates, averaged EPSPs or isolated ionic-current measurements.

## Verification and next use

Acquisition finished in 21.31 s, 36,357,941 bytes total, below the registered
120-second/40-MB caps. No model rollout or remote job ran. The channel-identity and
window helper passes 17 sibling tests with 100% line coverage, including reordered,
duplicate and wrong-unit channels; exact boundary inclusion; overlapping windows;
no-spike recordings; and invalid clocks/arrays. A separate source reread matches
every retained full pre/post sweep and all event samples exactly.

The first independent clock check incorrectly regenerated timestamps by dividing
sample index by rate. pyabf constructs them by multiplying index by its recorded
sample period; floating-point evaluation differs at some samples. The verification
now uses that source clock operation and retains exact equality. No recorded
timestamps or tolerances changed. Future checks should distinguish source-byte
preservation from numerical equivalence of independently generated time grids.

This input makes a synaptic comparison possible without substituting an animal
waveform. It is not yet a calibrated synaptic model. Before fitting, define a
held-out pair/sweep policy and an observation model that accounts for spontaneous
activity, baseline variation and the postsynaptic cell's electrical response.
No unknown conductance or receptor identity is filled in from an EPSP alone.
The existing L1 holdouts remain untouched. The unchanged all-104 completion
contract still requires human-qualified donors, anatomy transfer and full driven
controls.
