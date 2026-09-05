# Fine axonal mesh qualification

The selected PV calcium candidate at 0.27 nA satisfies the declared
successive-mesh check from axon factor 729 to 2187. Other regions remain
at factor 9. Both traces contain 40 complete events with equal peak-sign
classes. Every ordinal onset changes by at most 0.012573610 ms, below
the 0.100 ms limit. The selected late interval changes by 0.000338269 ms,
below the 0.010 ms limit. Section geometry and fixed model parameters
pass the audit invariants. Total segment counts are 2853 and 5769.

The [complete comparison](h01-pv-focused-mesh-2187.json) retains every
event and difference. Prior failed comparisons remain in the evidence.
This is qualification of one successive axonal refinement at one input.
It does not establish global convergence across all regions or inputs.

Before the final run completed, the squared-compartment-length model
predicted the 40th onset at 1245.927473867055 ms. The observed onset is
1245.9274627168472 ms. Absolute error is 0.000011150208 ms, within the
predeclared 0.005 ms limit. The
[prediction result](h01-pv-axon2187-prediction-result.json) supports this
conditional numerical prediction. It is not biological validation.

The physiological waveform and early-interval errors remain. The new
numerical result permits their diagnosis with less timing uncertainty
from this axonal refinement. It does not promote the diagnostic candidate
to a validated BrainCell cell or a validated inhibitory circuit model.
