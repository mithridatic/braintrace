# Repeated threshold input does not give one fixed response

Seven 200 pA long-square trials have exactly equal complete command
waveforms and the same -3.711859 pA bias. Each includes the 50 pA test
pulse at 5--15.02 ms and the main pulse at 1020--2020 ms. All are sampled
at 50 kHz. Source identity and voltage conventions are unchanged.

| Sweep | Complete events | First onset, ms | Baseline mean, mV |
|---|---:|---:|---:|
| 56 | 1 | 1225.459 | -83.792900 |
| 57 | 0 | Absent | -83.894821 |
| 58 | 0 | Absent | -83.907219 |
| 59 | 1 | 1228.401 | -83.966904 |
| 60 | 1 | 1172.239 | -83.697548 |
| 61 | 1 | 1218.741 | -83.647758 |
| 62 | 1 | 1223.564 | -83.769417 |

Every trial passes the existing partial baseline and bias screen. This
does not establish full acquisition quality. The subsequent
[source-quality audit](h01-l2-source-quality.md) also finds that all seven
pass the available source-reported numeric checks. Final source approval
and manual review remain unverified. The
[direct record](h01-l2-repeatability200.json) retains each event's peak
and duration, baseline checks, pulse maximum, and missing-event status.
The NPZ file retains each reported and corrected voltage trace, plus the
verified common command and time arrays. Reserved sweep 53 voltage was
not accessed. These trials were not used to fit a candidate.

The observed command and bias do not determine one unique recorded
threshold response across these trials. The internal initial states are
not measured completely. Noise, state differences, or recording effects
remain possible; this observation does not separate their causes.
The baseline means also differ. A deterministic simulation with one
fixed initial state is not seven independent biological trials.

There is no repeated 250 pA long-square trial in this file. Therefore,
these observations do not justify a numerical tolerance around the
250 pA calibration spike times. Nor do they establish a population
distribution. Final qualification must distinguish direct calibration
error, same-input repeatability, and reserved-input prediction. Do not
use the observed threshold-time range to excuse errors at other inputs.
