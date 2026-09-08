# SK candidate at two calibration currents

Both simulations completed. Closure factor remains 0.18 and recovery factor 1.
SK density doubles in soma and axon. All other parameters remain unchanged.

| Observation | 0.19 nA, source SK | 0.19 nA, doubled SK | 0.27 nA, source SK | 0.27 nA, doubled SK |
| --- | ---: | ---: | ---: | ---: |
| Last complete interspike interval, ms | 43.404 | 64.274 | 12.636 | 14.454 |
| First peak, mV | 22.934 | 22.870 | 23.254 | 23.221 |
| First duration above -20 mV, ms | 0.273111 | 0.272808 | 0.273679 | 0.273621 |
| First rising -20 mV crossing, ms | 303.0095 | 305.7197 | 281.8025 | 282.0154 |
| Complete events | 29 | 22 | 86 | 77 |

The later intervals lengthen while first-event shape stays close to the prior
candidate. The human counts are 12 and 43. The candidate remains too active.
The low-current onset error becomes worse. Do not promote this parameter set.
The last interval compares each trace's own last complete interval; these are
not matched event indices or equal-time measurements. Every interval is retained.

At each upward -60 mV crossing between complete spikes, the audit records
somatic calcium, SK gate, and SK current. This fixes the driving voltage for
comparison without changing the original event times or voltage traces.
The current remains outward at these observations.
The recorded current agrees with density times gate times driving voltage
within 1e-10 mA/cm2 over the full trace.

The calcium and gate states differ after the density change. Thus, a doubled
density does not imply a doubled current along the resulting trajectory.
For example, at the final -60 mV recovery crossing of the low-current runs,
somatic SK current changes from 5.83145e-6 to 6.34674e-6 mA/cm2.
The crossing times differ. This is not an isolated current-gain measurement.

The conductance intervention causes a response change in the model, but these
observations do not identify soma versus axon mediation. Axonal SK current was
not recorded. Do not claim that the local somatic observations explain the
whole-cell effect. The earlier SK-necessity rejection remains in force for
its specified second-spike timing claim.

The [JSON audit](h01-pv-sk-candidate-audit.json) retains all event intervals and
matched-voltage recovery observations. Run
`python -m docs.evidence.h01_pv_sk_candidate_audit` to reproduce the checks.
The reserved 0.23 nA trace remains unused.
