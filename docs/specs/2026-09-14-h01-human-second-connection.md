# Transfer to a second input onto the recorded human cell

Approved parent: human-only qualification of all 104 cells and synaptic response.
Test whether the frozen cell response and synaptic waveform transfer to a second
source-identified connection before deploying either as a general human model.

## New source and fixed identities

Acquire only Dataverse TONOHA v1.0 file 359655, 2015_11_04_0077.abf,
36,007,936 bytes, published SHA1 ddabe5a487523726c42237dbf2fea0fac5a519e8.
The already retained source CSV identifies Human, EXC, date 20151104, slice 3,
cluster 6, presynaptic IN3 and postsynaptic IN2. Those join fields identify the
same postsynaptic cell as record 0076, with a different presynaptic input.
This is the authors' table linkage, not a separately verified universal cell ID.
Verify ABF channel names, physical ADC numbers, units, date and command protocol.
Cap acquisition at 40 MB and 120 seconds; no other source downloads.

Freeze the original cell candidate and synaptic candidate from human-paired-passive
and human-synaptic-prediction-r2. Do not refit R, membrane tau, the effective series
term, synaptic rise/decay or delay using the new record. Before a synaptic transfer
claim, test the frozen electrical model against this record's postsynaptic -10 pA
Step pulse using the same source windows and baseline operator as before. Require
all 30 complete sweeps and verify every command; stop on protocol disagreement.

## Calibration and validation

For the second connection, calibrate only its one positive peak-current strength
using sweeps 0-14. Hold sweeps 15-29 for validation and freeze strength before
extracting their EPSP responses. The original source summary for this connection
is already known; this is not a blind study or independent-donor validation.

Reuse the corrected source-epoch command adapter and identify six positive DAC3
pulses, with exactly one recorded IN3 zero crossing in each. Preserve all extra
crossings, if any, and inspect them without discarding original voltage samples.
Use the same -10 to +50 ms windows, [-10,-2) ms pre-event baseline and inclusive
0-50 ms scoring window as the first connection. Retain original pre/post voltage,
source indices and clocks for all 180 events. No event removal, temporal warp,
kinetic adjustment, per-event gain or response-fitted baseline.

The frozen membrane-filtered template is linear in peak current. Compute its
least-squares strength using every calibration sample, bounded to [0.01,2000] pA.
Record the unconstrained optimum and reject a bound-active result. This changes
one connection strength, not the frozen waveform. Compare the original 130.03165
pA strength, the new calibrated strength and the same no-evoked-response baseline.
Do not inspect validation EPSP scores until the new strength is frozen.

## Registered decision

Electrical transfer: on validation sweeps 15-29, the frozen electrical response
must improve combined RMSE by at least 25% over the pre-pulse-baseline reference,
with no sweep worsening by more than 5%. Report all 30 sweep errors. This gate
does not change the electrical parameters or establish their unique physical
interpretation.

Synaptic transfer: the amplitude-only candidate must have an interior optimum,
improve combined validation EPSP RMSE by at least 25% over the baseline reference,
and worsen no validation sweep by more than 5%. Both electrical and synaptic
gates must pass for a transfer claim. Retain all 180 event and 30 sweep errors,
including regressions masked by group averages. No population ledger term is
promoted by this local transfer test alone.

Use local CPU only, with a separate 120-second cap for analysis. Inspect matched
raw/centered response plots for sweeps 0, 1, 14, 15, 16 and 29, first/second/last
events, and every residual. Verify source samples, the analytical strength solution
and gate calculations independently. If a gate fails, retain it and do not refit
kinetics or weaken criteria after validation.
