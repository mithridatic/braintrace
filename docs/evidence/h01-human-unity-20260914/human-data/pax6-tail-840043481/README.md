# Human PAX6 tail observations constrain the next current model

Five previously unopened calibration tail responses are now preserved and checked
against the original human recording. Their common conditioning currents are
similar, while the tails remain inward at every command from -90 to -150 mV.
These observations support a joint amplifier-current model across more voltages;
they do not supply an isolated potassium deactivation rate or a measured reversal
potential. No donor candidate, physiological qualification or six-term score changed.

## Observed response and model-development decision

Opened and inspected both [complete response views](direct-tails.png) and
[commands with exact landmarks](commands-and-landmarks.png). The five +70 mV
conditioning responses have the same broad rise and decline. At 10 ms their
original samples span 705-759 pA; at 990 ms they span 287.5-301.9 pA. There are
visible fluctuations and differences; this is not an equality or stationarity
test. The early +10 mV control traces also overlap broadly. The strong monotonic
first-pulse change seen in the later recovery block is not visually apparent here.
Do not transfer that conclusion between recording blocks.

| Sweep | Tail command, mV | Raw current at 5 ms, pA | Raw current at 200 ms, pA | Tail late mean, pA |
| --- | ---: | ---: | ---: | ---: |
| 99 | -89.987 | -55.000 | -18.125 | -17.050 |
| 100 | -104.987 | -60.000 | -21.875 | -21.141 |
| 101 | -119.987 | -59.375 | -29.375 | -25.906 |
| 102 | -134.987 | -73.750 | -35.625 | -31.857 |
| 103 | -149.987 | -79.375 | -34.375 | -33.487 |

The large negative onset transient is shown in full. Beyond 5 ms, current rises
toward a negative late level over tens to hundreds of milliseconds. Subtracting
each tail's own 450-490 ms mean leaves a negative early difference in all five
sweeps: -33.47 to -45.89 pA at 5 ms. These curves approach zero later, with noise
and crossing order. There is no observed sign bracket for that early difference;
extrapolation must not be reported as a measured reversal. Initial baselines are
approximately -15.15 to -15.56 pA, distinct from the voltage-dependent tail levels.
After return to -90 mV, the final currents are close to the initial level; sweep
99 continues at -90 mV and has no extra transition at 2600 ms.

The next candidate must jointly predict the +70 mV conditioning response, the
negative tail, and the return, using an explicit observation model for amplifier
and membrane currents. A sum of positive availability terms fitted only to the
depolarized pulse is insufficient as a description of these raw observations.
No particular ionic identity, number of channel populations, reversal potential,
or recording artifact has been established. Capacitive/compensation effects,
voltage-dependent leak, biological gating and acquisition variation remain
possible contributors. Fitting a standalone exponential and calling its constant
potassium deactivation would discard these unresolved alternatives. Keep this
tail family as a joint raw-current constraint rather than adding another
availability population to repair the already rejected recovery fit.

## Source and verification

The [registration](../../../../specs/2026-09-14-h01-pax6-tail-constraint.md)
was written before response access. Source: human PAX6 CDH12 specimen 840043506,
session 840043481, SHA256
`30abbb3cca63b0629242c7ad96f595d6e6ceea2ce7e522fdb5ebf8a3bc2ac944`.
All five commands match the previously saved command-only inventory exactly.
[Original arrays](observations.npz) contain every current, DAC and command sample,
the original sweep clock, and exact baseline/late-window indices. The
[observations](observations.json) retain absolute start times, instrument settings,
unit conversions, source hashes, all 25 landmarks and both separate centerings.
No interpolation, filtering, transient clipping or ionic leak subtraction was used.
Reconstructed command remains distinct from measured patch voltage.

[Independent verification](verification.json) checked all 450000 response samples,
all baseline/late indices and 25 tail landmarks against source HDF5. Extraction
also checked all 450000 DAC samples. The existing exporter passes
[28 sibling tests](tests.xml), including vector-shaped starting time, source/clock
mismatch, malformed commands and rejection of external-validation access.
The first verifier attempt used a scalar conversion on a one-element source
array and failed with `TypeError: only 0-dimensional arrays can be converted to
Python scalars`. It was corrected to explicit single-element extraction, matching
the exporter; independent verification then passed. No data or analysis output
changed. Future source scalar reads must check shape rather than assume scalar
storage. Additional model tests will need complete signed-current predictions,
command-history carry and predictions on reserved recordings before deployment.

The local CPU extraction completed in [4.575 seconds](terminal.json), below its
120-second cap. No membrane simulation ran. The external PAX6 validation currents
and original whole-cell holdouts remain unopened; these newly exposed observations
are calibration data, not an independent validation result.
