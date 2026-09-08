# Recovery minima change little in the existing numerical comparisons

Recompute the four sampled interspike minima from the nominal, tighter
tolerance, and coarser spatial traces. Use the same event boundaries and
earliest-sample tie rule as the original minima report.

| Comparison with nominal | Maximum voltage change (mV) | Maximum change in delay after fall (ms) |
| --- | --- | --- |
| Tolerance 1e-11 | 0.000000142404 | 0.000305380 |
| Mesh factor 3 | 0.000945123 | 0.00121202 |

The largest absolute minimum-time change on the stimulus clock is
0.000305574 ms for tolerance and 0.0195694 ms for space. Absolute time
includes shifts in the preceding spike; delay after its fall isolates
the timing of this part of the recovery observation.

These changes are small compared with the reported human mismatches.
The minima remain too negative and too late relative to the preceding
fall in both comparisons. The [full review](h01-l2-reversal-minima-numerical-review.json)
retains each minimum and difference with source hashes.

No prospective extrema-specific acceptance threshold was set. This is
a retrospective sensitivity review, not another pass/fail gate. It does
not identify the responsible current or establish full-trace convergence.
