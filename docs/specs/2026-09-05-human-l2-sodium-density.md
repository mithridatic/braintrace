# Somatic sodium density split

Use the calcium midpoint (factor 1.5) with source sodium kinetics as
control. Its first onset is 4.534849 ms early and its first peak is
1.237200 mV high. The slower-recovery split had a small first-interval
benefit but worsened the next two intervals. Do not promote that factor.

Reduce only somatic NaTs conductance density to 0.9 of source. This is
an inferred intervention. Lower available inward conductance can slow
the approach to threshold and reduce the spike peak. Whether both human
errors improve in this coupled cell is the prospective prediction.

Keep the isolated default-one sodium build, recovery factor 1, calcium
factor 1.5, mesh factor 9, dt 0.000625 ms, complete command-only input,
initial state, temperature, and 2100 ms endpoint. Scale exactly one
genome row: soma / NaTs / gbar_NaTs. Preserve the source fit and hashes.
The helper must reject invalid factors and missing or duplicate targets.

Before observing the result, require smaller absolute first-onset and
first-peak errors than control, five complete positive-peak events, and
no greater absolute error for any of the four interspike intervals.
Report each condition separately. Missing first-spike data invalidate
the waveform prediction. Lost later events reject the full prediction.
Retain all event onsets, peaks, durations, minima, and human residuals.
Do not use the reserved sweep. A supported prediction still requires
candidate numerical and cross-input validation before cell promotion.
