# Direct voltage minima after the first spikes

The combined candidate has soma NaTg factor 1.10, closure factor 0.15,
and recovery factor 1. The observations use its existing mesh-9 traces.
Each minimum is the lowest sampled voltage from a spike peak to the
next upward -20 mV crossing. No time shift is applied. This is not a
threshold-relative AHP amplitude or a fit to a population average.

| Input, nA | Interval after peak | Human minimum, mV | Model minimum, mV | Human time from peak, ms | Model time from peak, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.19 | 1 | -78.843750 | -71.862964 | 0.760000 | 2.297476 |
| 0.19 | 2 | -78.718750 | -71.597338 | 0.780000 | 2.258834 |
| 0.19 | 3 | -78.906258 | -72.408337 | 0.760000 | 2.305399 |
| 0.27 | 1 | -78.906258 | -68.118965 | 0.700000 | 2.045295 |
| 0.27 | 2 | -78.125000 | -63.764814 | 0.740000 | 1.927612 |
| 0.27 | 3 | -78.031250 | -53.078101 | 0.740000 | 1.386200 |

At high input, the fourth model excursion has a negative peak. It remains
in the record and supplies the end of the third interval; it must not be
called a positive spike. All other listed adjacent peaks are positive.

The candidate's early minima are less negative and occur later than the
recorded minima. At high input, successive minima become less negative
before positive spiking stops. This localizes a waveform mismatch below
the -20 mV level that the earlier duration measurement did not describe.
It does not prove that insufficient potassium current is the cause.
Sodium, calcium, other currents, and axial exchange can also affect this
part of the trajectory. A selective intervention is still needed.

The next channel diagnosis should include these return-voltage and timing
targets, together with first-spike shape and later positive events.
The [JSON](h01-pv-postspike-minima.json) retains exact interval endpoints,
sample times, and peak-sign flags. Run
`python -m docs.evidence.h01_pv_postspike_minima` to reproduce the result.
Only the two calibration traces are read. If fewer than four complete
events exist, fewer intervals are reported. Empty sample intervals fail
explicitly. Tied minima select the first sample.
