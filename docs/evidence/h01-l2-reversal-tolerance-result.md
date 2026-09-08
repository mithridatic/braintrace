# Reversal candidate passes the subthreshold tolerance check

Reducing CVode tolerance from 1e-10 to 1e-11 changes the 12 specified
voltage samples by at most 2.780401e-7 mV. This is below the predefined
0.01 mV numerical limit. Both runs retain a declining late pulse response,
a return below the starting voltage, and no pulse spikes.

Metadata differ only in tolerance, sample count, and execution time.
Both recordings match their indexed raw arrays exactly. Both inputs
match the source command plus bias with zero plateau error. The
plateau check excludes points within 1e-7 ms of command transitions.
Both traces are finite and end exactly at 2100 ms.

The [full result](h01-l2-reversal-tolerance-result.json) retains each
sample difference and source artifact hash. This check supports the
specified subthreshold observations under tolerance refinement. It does
not establish convergence at every time, spatial convergence, numerical
accuracy of spikes, or agreement with the human waveform.
