# Smaller Kv3 closing change improves minima but worsens firing times

The predefined joint recovery prediction passes. Five complete positive-peak
spikes remain, and each absolute minimum-voltage error decreases.

| Recovery minimum | Human voltage error (mV) | Absolute error reduction (mV) |
| --- | --- | --- |
| 1 | +0.058672 | 2.035666 |
| 2 | -2.863275 | 2.103213 |
| 3 | +0.156047 | 1.758046 |
| 4 | +0.182761 | 1.689569 |

Three minima are now slightly too shallow; the second remains too deep.
Their delays after the preceding falling crossing remain about 2.28-2.40 ms,
compared with human delays about 1.23-1.44 ms. Matching minimum voltage
does not establish matching recovery timing.

| Spike | Onset error (ms) | Peak error (mV) | Duration error (ms) |
| --- | --- | --- | --- |
| 1 | +0.901594 | -0.071490 | +0.027486 |
| 2 | -7.409672 | +0.192271 | -0.176925 |
| 3 | +51.460395 | +0.099751 | -0.023000 |
| 4 | +60.259438 | +0.180940 | -0.020034 |
| 5 | +89.444094 | +0.389200 | -0.012392 |

Duration is time above -20 mV. Interval errors are -8.311265, +58.870067,
+8.799043, and +29.184656 ms. The first interval is too short; the later
intervals are too long. Peak errors remain below 0.40 mV, but rising phases
remain too short. These errors prevent full physiological acceptance.

Only the Kv3 closing factor changes among physical setup fields, from
one to 0.9. Source and library hashes match. Raw mappings are exact,
input plateau errors are zero, arrays are finite, and both traces end
at 2100 ms. The [full result](h01-l2-kv3-closing-ninety-result.json)
retains every event, phase, interval, minimum, and residual.

The smaller change retains repeated spiking while the half-time change
does not. This establishes two distinct tested outcomes, not a smooth
dose-response law or the exact transition between them. Closing time
affects both recovery depth and the coupled later firing response. The
intervention does not isolate the separate pathways responsible for
those timing changes. Candidate numerical and subthreshold checks remain
open; no production promotion or reserved-input validation has occurred.
