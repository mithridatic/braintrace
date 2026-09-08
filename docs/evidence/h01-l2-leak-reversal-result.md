# Passive reversal restores the tested response directions

The predefined -4 mV passive-reversal shift passes the starting-voltage
prediction. Absolute error at 1019 ms falls from 3.192793 to 0.510911 mV.
All three separate shape conditions also pass at the tested settings.
Voltage falls by 0.215632 mV from 1120 to 2019 ms. At 2099 ms it is
0.397043 mV below its own starting value. The pulse remains below -20 mV.

The human response falls by 1.5 mV between the same pulse samples and
returns 1.3125 mV below its starting value. The candidate's absolute
voltage error at 2019 ms remains +1.672111 mV. Passing the direction
checks is not a full waveform fit.

Metadata comparison found only the intended new passive settings and
output sample count and timing differences. Applied passive settings
match the pinned source fit with only e_pas reduced by 4 mV. The legacy
control lacks the new passive metadata; its source settings are recovered
from that fit. All shared model metadata match, including the genome,
geometry, mechanism hashes, input, and solver settings.

All final arrays match their indexed raw samples exactly. The recorded
current matches the source command plus bias on plateaus, with zero
maximum error. Points within 1e-7 ms of command transitions are excluded
from this plateau check. The endpoint is exactly 2100 ms.

The [result](h01-l2-leak-reversal-result.json) retains all 12 direct
observations, errors, audit outcomes, and input artifact hashes.
Candidate numerical checks, active-input behavior, and reserved-input
validation remain open. No production model is promoted.
