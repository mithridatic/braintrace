# Slower calcium removal separates later recovery from the first return

Doubling only somatic calcium-removal time supports the declared
diagnostic prediction on the factor-nine mesh at dt 0.000625 ms.
The setup and current-prefix checks pass. No other fitted parameter
changes. The first three individual interval errors all become smaller.

| Interval | Human, ms | Control, ms | Doubled removal time, ms |
| --- | ---: | ---: | ---: |
| 1 | 33.851739 | 26.364223 | 26.574181 |
| 2 | 220.403496 | 117.067517 | 197.739810 |
| 3 | 307.406282 | 207.433505 | 377.446422 |

The first minimum changes by -0.000232590 mV and the first sampled
peak by -0.000109188 mV. Both satisfy the declared shape constraints.
Thus, the tested removal-time change substantially alters later
intervals without materially changing these first-return observations.

This is not a validated fit. The first interval remains 7.277557 ms too
short, and the third overshoots by 70.040140 ms. The model has four
events while the human trace has five. The
[complete result](h01-l2-calcium-removal-result.json) retains the direct
events, all three residuals, and each separate acceptance condition.

The result is conditional on the tested model and input. Calcium also
affects its reversal potential, so this does not isolate SK as the only
path. The doubled parameter is inferred, not measured human physiology.
The independent factor-nine solver comparison passes for this changed
four-event regime. Maximum onset difference is 0.093174154 ms, peak
difference 0.002755975 mV, and duration difference 0.000038315 ms.
Counts and peak-sign classes match. Setup, input, and raw-index checks
also pass. The [solver comparison](h01-l2-calcium-removal-solver-comparison.json)
supports this numerical boundary; it does not establish all-mesh
convergence or improve the remaining physiological errors.

The [factor-1.5 midpoint](h01-l2-calcium-midpoint-result.md) supports its
separately declared third-interval prediction, but retains substantial
first- and second-interval errors. It does not use the reserved input.
