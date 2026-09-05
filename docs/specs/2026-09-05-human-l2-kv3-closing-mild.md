# Test a smaller Kv3 closing change and qualify the failed response

## Smaller intervention

The half-time intervention removed four later spikes and left a
sustained depolarized response. Test closing factor 0.9 against the
source-equivalent factor-one control. This is a smaller inferred change,
not a measured human kinetic value. Do not infer a smooth response from
the failed factor-0.5 endpoint.

Keep sodium opening factor 2, density factor 1.3, recovery factor 1,
calcium factor 1.375, distributed Ih factor 75, passive reversal shift
-4 mV, original leak, temperature, geometry, and initial voltage fixed.
Use the checked kv3-closing-source build, sweep 50 plus recorded bias,
mesh 9, CVode tolerance 1e-10, and endpoint 2100 ms.

Apply the original joint prediction: five complete positive-peak spikes
and smaller absolute human error at each of the four between-spike
minimum voltages than the factor-one control. Use the earliest sampled
minimum strictly between the preceding fall and next rise through
-20 mV. Missing or extra spikes reject the joint prediction. Retain each
event, phase, interval, minimum delay, residual, and unmatched human event.
A valid prediction failure is rejected, not relabeled invalid.

## Tolerance of the rejected half-time response

Repeat factor 0.5 with tolerance 1e-11 and all physical settings fixed.
Require the same one complete positive-peak spike, onset change <=0.1 ms,
peak change <=0.1 mV, duration change <=0.01 ms, and each separate phase
change <=0.01 ms. At the already selected times 1082, 1090, 1100, 1110,
1120, 1200, 1500, 2019, and 2099 ms, require each absolute voltage
change <=0.01 mV. Retain each difference. These times were retrospective
in the first review and are prospective for this tolerance comparison.
Passing supports tolerance stability, not a unique sustaining mechanism
or spatial convergence. Failing a valid comparison rejects tolerance
qualification of the corresponding observation.

Both runs must preserve source/library hashes, finite arrays, exact raw
mapping, input plateaus within 1e-12 nA outside 1e-7 ms of command edges,
and the 2100 ms endpoint. Wrong setup or malformed data invalidates the
comparison. The new 0.9 candidate requires its own numerical checks
before qualification. Do not access the reserved human response.
