# Smaller Kv3 control passes the declared tolerance check

Tightening CVode tolerance from 1e-10 to 1e-11 preserves five positive-peak
spikes. All declared limits pass.

| Observation | Maximum absolute change | Limit |
| --- | --- | --- |
| Onset | 0.000028477 ms | 0.1 ms |
| Peak | 0.000287689 mV | 0.1 mV |
| Duration above -20 mV | 0.000000540 ms | 0.01 ms |
| Separate rising or falling phase | 0.000627008 ms | 0.01 ms |

All four minimum voltages change by less than 0.000001881 mV. Their
relative delays change by at most 0.001316257 ms. These extrema results
are descriptive sensitivity, not a separate prospective acceptance gate.

Only tolerance, sample count, and integration time differ. Raw mappings
are exact, input plateau errors are zero, arrays are finite, and both
traces end at 2100 ms. The [full result](h01-l2-kv3-ninety-tolerance-result.json)
retains each event, phase, and minimum difference and source hashes.

This qualifies the specified tolerance comparison for Kv3 factor 0.9
with calcium factor 1.375. It does not qualify the newer calcium-1.33
candidate, spatial resolution, or human physiology. The large timing
errors and the recovery-depth improvement persist at tighter tolerance.
