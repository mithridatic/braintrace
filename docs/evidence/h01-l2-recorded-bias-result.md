# Recorded bias is not a sufficient correction

The recorded-bias intervention rejects the predeclared sufficiency claim
at the tested source setup. The model still produces seven positive-peak
events during the pulse, while the recording contains five. Its first
onset is 2.338391 ms early and its fifth onset is 280.301332 ms early.
Both the count condition and the 1 ms every-onset condition fail.

The intervention adds -0.003711858945210089 nA to the complete command.
The source fit, section properties, time step, initial-state rule,
temperature, and channel parameters match the control. Recorded input
plateaus match the intended command-plus-bias values. Each of the four
recorded current transitions follows its source command edge by one
model step, within floating-point tolerance. Initial current readback
at time zero is excluded from the plateau check; the stored current
trace retains this boundary sample.

The first peak is 1.301979 mV above the human sampled peak. Its duration
above -20 mV is 0.020937 ms shorter. The model baseline is -84.515114 mV,
compared with -84.232297 mV in the recording. Baseline closeness and a
delayed first spike do not establish agreement of the complete response.

The [complete result](h01-l2-recorded-bias-result.json) retains human
residuals, control events, intervention events, and each intervention
minus control difference. This split supports a conditional effect of
the known input change. It does not identify which channel law causes
the remaining human mismatch. Intervention-specific temporal and spatial
robustness remain unqualified; no physiological model is promoted.
