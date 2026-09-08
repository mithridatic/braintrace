# Fixed reversal candidate retains five spikes with timing errors

The fixed candidate produces five complete positive-peak events during
the calibration pulse. It passes this necessary condition. No parameter
changed between the subthreshold and active checks.

| Spike | Onset error (ms) | Peak error (mV) | Duration error (ms) |
| --- | --- | --- | --- |
| 1 | +4.980777 | +0.770545 | -0.026141 |
| 2 | +8.335163 | +1.067468 | -0.230447 |
| 3 | +30.037582 | +0.970322 | -0.076755 |
| 4 | +53.877448 | +1.051042 | -0.073867 |
| 5 | +95.614524 | +1.257623 | -0.066272 |

Errors are model minus human. Duration is time above -20 mV, not half
width. Each interval is too long: errors are +3.354386, +21.702419,
+23.839866, and +41.737076 ms. The increasing onset error follows from
these interval errors. It does not identify a channel mechanism.

Setup metadata differ only in waveform, sweep number, sample count,
and execution time. All final arrays match the indexed raw samples.
Current matches the source command plus bias with zero plateau error;
samples within 1e-7 ms of command transitions are excluded from that
check. The trace is finite, covers the full pulse, and ends at 2100 ms.
Integration took 643.821193 seconds.

The [full result](h01-l2-reversal-active-result.json) retains every event,
interval, residual, audit outcome, and source artifact hash. This result
does not qualify the cell. Numerical checks, reserved-input validation,
and correction of the direct waveform errors remain open.
