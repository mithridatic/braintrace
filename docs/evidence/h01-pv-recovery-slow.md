# Opposite-direction recovery test

This test changes recovery factor from 1 to 2 in the combined candidate.
Soma NaTg factor stays at 1.10 and closure factor stays at 0.15.
Other model parameters, initial state, and stimulus settings remain fixed.
Both inputs use mesh factor 9 and CVode tolerance 1e-10.

The predeclared high-input rescue requires at least two complete
positive-peak spikes after 500 ms. The result is false: there are none.
The faster-recovery rejection remains unchanged in its separate report.

At 0.19 nA, the slow-recovery run has 39 complete events, with 28 positive
events after 500 ms. The control has 39 complete events, with 27 positive
events after 500 ms. Equal total counts do not mean equal event times.
The first-spike errors remain about +8.591027 ms in onset, -0.759895 mV
in peak, and -0.005168 ms in duration above -20 mV.

At 0.27 nA, the run has four complete positive-peak events. Their sampled
peak times are 281.440912, 286.901252, 291.181910, and 294.721435 ms.
Their peaks are 19.132595, 17.144053, 12.270464, and 4.464650 mV.
There is no late spike train. The first-spike errors remain about
-0.520249 ms in onset, -0.304905 mV in peak, and +0.001743 ms in duration.

Neither recovery factor 0.5 nor 2 restores late high-input firing in this
candidate. This excludes those two proposed fixes, not all recovery laws.
Do not infer a unique cause of the failure. No model is promoted.
The conditional finer-mesh rescue check was not triggered because the
rescue criterion failed. Numerical robustness of this failure remains open.

The [audit](h01-pv-recovery-slow-audit.json) retains each event and interval.
Raw traces use the prefix `h01-pv-recovery-slow-`. Reproduce the audit with
`python -m docs.evidence.h01_pv_recovery_rescue --slow`. The default command
still reproduces the factor-0.5 rejection. Four existing datum and reference
tests passed. Missing events and nonpositive peaks cannot pass the rescue
decision. The reserved input was not read.
