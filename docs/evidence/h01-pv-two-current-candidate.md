# Intermediate waveform candidate at two currents

The inferred sodium-inactivation factor is 0.18. Recovery remains unchanged.
No other physical parameters change. Both runs use mesh factor 9 and CVode
atol 1e-10, with zero bias and initial voltage -80 mV.

| Direct first-event error: model minus human | 0.19 nA | 0.27 nA |
| --- | ---: | ---: |
| Peak voltage, mV | +3.37185 | +3.81614 |
| Time above -20 mV, ms | -0.000283 | +0.006506 |
| Rising -20 mV crossing time, ms | +12.21184 | -0.06338 |

The first-event duration is close at both calibration inputs. The peak error
is much smaller than in the original source model. These are useful candidate
results, not an acceptance decision. Numerical refinement of this candidate
and physiological error tolerances remain pending.

The low-current onset is late, although the high-current onset is close.
The model has 29 events versus 12 recorded at 0.19 nA, and 86 versus 43 at
0.27 nA. Every detected model event reaches a positive peak.
The first interspike interval at 0.19 nA is 12.345996 ms. Later firing is too
sustained. These different errors must not be hidden in one mean firing rate.

Do not promote the candidate. A waveform-duration match does not establish
correct excitation threshold, recovery, or adaptation. Use the distinct onset
and later-event errors to constrain the next change. The existing SK split
excludes SK as a necessary path for the second-event delay; it does not exclude
SK as a constraint on later adaptation.

The [JSON comparison](h01-pv-two-current-candidate.json) retains all model and
human events. Residual NPZ files preserve the original clock without alignment.
The reserved 0.23 nA trace was not used. Run
`python -m docs.evidence.h01_pv_two_current_candidate` to repeat the comparison.
