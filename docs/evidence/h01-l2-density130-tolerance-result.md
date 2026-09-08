# Increased-density candidate passes the declared tolerance check

Tightening CVode tolerance from 1e-10 to 1e-11 preserves all five
positive-peak events. All declared direct-response limits pass.

| Observation | Maximum absolute change | Limit |
| --- | --- | --- |
| Onset | 0.000023019 ms | 0.1 ms |
| Peak | 0.000042418 mV | 0.1 mV |
| Duration above -20 mV | 0.000001091 ms | 0.01 ms |
| Separate rising or falling phase | 0.000416262 ms | 0.01 ms |

Only tolerance, sample count, and integration time differ in metadata.
Raw mappings are exact. Input plateau error is zero in both runs; all
arrays are finite and end at 2100 ms. The
[full result](h01-l2-density130-tolerance-result.json) preserves each event
and phase difference, the limits, and hashes.

This result applies to sodium density 1.3 and opening factor two at
calcium factor 1.375, distributed Ih factor 75, and reversal shift -4 mV.
It supports tolerance stability of these direct spike observations.
It does not establish spatial convergence, subthreshold numerical
stability, or physiological agreement. The observed waveform and timing
errors relative to the human recording remain much larger than these
tolerance changes.
