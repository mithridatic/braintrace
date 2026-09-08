# Opposing phase errors partly cancel in spike duration

Split each complete event at its sampled voltage peak. The first phase
runs from the interpolated -20 mV upward crossing to that peak. The
second runs from the peak to the interpolated downward crossing.

| Spike | Rise-to-peak error (ms) | Peak-to-fall error (ms) |
| --- | --- | --- |
| 1 | -0.171088 | +0.144946 |
| 2 | -0.198776 | -0.031671 |
| 3 | -0.194149 | +0.117394 |
| 4 | -0.188202 | +0.114335 |
| 5 | -0.195680 | +0.129408 |

Errors are model minus human. The model reaches each peak too quickly
after the upward crossing. Four of the five model events then take too
long to return to -20 mV. These opposing errors partly cancel in the
total duration. For the second event both phases are too short.

The [measurements](h01-l2-reversal-spike-phases.json) retain both source
durations and each residual. Their sums reproduce the total duration
errors to within 1e-9 ms. No waveform alignment or fitting was applied.
Human peaks use the 0.02 ms recording grid; model peaks use the adaptive
output grid. The pending spike numerical check is still required.

This localizes waveform errors. It does not show which channel produces
them. A uniform duration correction cannot be assumed to repair both
phases. A future intervention must retain separate rise and fall measures.
