# Validate the increased-density candidate across solver tolerance and input

Freeze the candidate at sodium opening factor 2, sodium density factor
1.3, recovery factor 1, calcium removal factor 1.375, uniform distributed
Ih factor 75, and passive reversal shift -4 mV. Keep original leak,
other source kinetics, temperature, geometry, initial voltage, and recorded
bias. Use the checked sodium-activation-source library, mesh 9, and
endpoint 2100 ms. The candidate remains diagnostic, not a qualified cell.

## Solver tolerance

Repeat sweep 50 with CVode tolerance 1e-11. Compare against the completed
1e-10 density-1.3 run. Require equal event counts and peak signs, maximum
individual onset change <=0.1 ms, peak change <=0.1 mV, duration above
-20 mV change <=0.01 ms, and each separate rising or falling phase change
<=0.01 ms. Retain all event and phase differences. Only tolerance,
sample count, and integration time may differ in metadata. This does
not replace a spatial check.

## Subthreshold response

Run the frozen candidate on sweep 43 at CVode tolerance 1e-10. This is an
already-used calibration input, not a held-out test. Predict retention
of all three established conditions: no complete spike during the main
pulse, voltage at 2019 ms below voltage at 1120 ms, and voltage at 2099 ms
below voltage at 1019 ms. Failure of any condition rejects this joint
prediction. Retain the twelve predefined human sample times, individual
voltages, changes from 1019 ms, and each human residual. Do not introduce
a new amplitude acceptance tolerance from the candidate output.

Input and its hash, sweep identifier, and run-dependent fields may change
between active and subthreshold runs. All physical settings must stay
fixed. The input must be the original sweep-43 command plus its recorded
bias, including its early test pulse. This input comparison does not
isolate the separate effects of sodium amount, opening, and calcium.

For both checks, require finite arrays, exact right-limit raw mapping,
correct input plateaus outside 1e-7 ms of command edges, and exact 2100 ms
endpoint. Malformed data or unintended physical changes invalidate a
comparison. Valid prediction failures are rejected, not invalidated.
Do not access reserved sweep-53 response. No production transfer follows
from these diagnostic checks alone.
