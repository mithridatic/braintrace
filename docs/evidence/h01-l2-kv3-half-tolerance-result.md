# Rejected half-time response persists at tighter tolerance

CVode tolerance 1e-11 preserves the same one positive-peak spike as
1e-10. The declared numerical comparison passes. Maximum changes are
0.000000293 ms in onset, 0.000001878 mV in peak, 0.000000390 ms in
duration, and 0.000014471 ms in a separate phase. These are below the
respective 0.1 ms, 0.1 mV, 0.01 ms, and 0.01 ms limits.

All nine predefined post-spike sample voltages change by less than
0.000000830 mV, below the 0.01 mV limit. The sustained depolarized
response and absence of later spikes persist at tighter tolerance.
The recovery-improvement hypothesis remains rejected; this passing
numerical check does not turn it into a physiological success.

Only tolerance, sample count, and integration time differ. Raw mappings
are exact, input plateau errors are zero, arrays are finite, and both
traces end at 2100 ms. The [full result](h01-l2-kv3-half-tolerance-result.json)
retains every event, phase, and sample difference and source hashes.
Spatial sensitivity and the specific current path that sustains the
response remain untested here.
