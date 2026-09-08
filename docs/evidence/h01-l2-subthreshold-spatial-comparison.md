# Subthreshold diagnosis passes one spatial refinement

The bias-included midpoint passes the factor-3 to factor-9 spatial check
at CVode tolerance 1e-10. Segment counts increase from 1239 to 3717.
Every section count triples while section geometry, parent connections,
and physical parameters remain consistent.

The largest voltage difference across the 12 predefined observations is
0.000283362491 mV, below the 0.01 mV numerical limit. Neither run has
a complete pulse spike. Both still rise between 1120 and 2019 ms and
remain above their starting voltage at 2099 ms. The
[full audit](h01-l2-subthreshold-spatial-comparison.json) retains every
signed voltage difference, setup result, and response-direction check.
Raw mapping, input plateaus, and endpoints pass.

Together with the tighter-tolerance result, this supports numerical
stability of the selected subthreshold observations. It does not prove
global waveform convergence or human physiological validity. The late
voltage error of about 4.12 mV remains much larger than the changes seen
in these checks. Continue diagnosis of model currents and electrical
load with the full direct response retained. Do not change physiological
acceptance limits or use the reserved input to select parameters.
