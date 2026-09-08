# Subthreshold mismatch persists under tighter tolerance

The bias-included midpoint passes the specified CVode tolerance check
from 1e-10 to 1e-11. The largest absolute difference across the 12
predefined voltage samples is 2.330168343e-7 mV, below the 0.01 mV
numerical limit. Both runs have no complete pulse spikes. Both still
rise between 1120 and 2019 ms and remain above their starting voltage
at 2099 ms, unlike the human observations.

The [complete audit](h01-l2-subthreshold-tolerance-comparison.json)
retains every sample and signed difference. Physical setup and source
identity agree. Core raw mapping, input plateaus, and endpoints pass.
The added default-one Ih helper leaves the applied genome unchanged.

This result supports stability of these direct voltage observations
under this tolerance refinement. It does not establish spatial or global
waveform convergence. The several-millivolt mismatch and wrong response
directions do not disappear with the tighter tolerance. Further diagnosis
must address model behavior, rather than claiming this tolerance change
solves the human-response error. No parameters or acceptance limits were
relaxed, and the reserved input remains outside fitting.
