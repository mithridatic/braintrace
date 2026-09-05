# Calcium midpoint improves one timing target but leaves recovery errors

The factor-1.5 midpoint supports its predeclared bracket prediction.
Its third interval is 293.203654 ms, with error -14.202627 ms. This
absolute error is smaller than at either endpoint. The first interval
error does not exceed the source error. The first minimum changes by
-0.000152459 mV and the first peak by -0.000034814 mV relative to source.
Both preservation conditions pass.

The model has five events, as does the human trace. It is not a validated
fit. The first three interval errors are -7.349581, -61.550693, and
-14.202627 ms. The second interval is worse than in the factor-two
candidate. All five model events occur earlier than their corresponding
human events. The [complete result](h01-l2-calcium-midpoint-result.json)
retains source, doubled, and midpoint observations without replacing
them with a count score.

Across these tested removal times, the first interval changes little
while later intervals change substantially. This observation directs
the next diagnostic toward a mechanism with an earlier effect. It does
not prove that no other removal-time value could alter that interval.
The midpoint parameter remains inferred and the reserved input remains
outside fitting. Candidate-specific numerical checks remain necessary
before physiological qualification.
