# Slower-opening control passes the declared tolerance check

Tightening CVode tolerance from 1e-10 to 1e-11 preserves five positive-peak
spikes. Maximum absolute changes are:

| Observation | Change | Limit |
| --- | --- | --- |
| Spike onset | 0.000012890 ms | 0.1 ms |
| Spike peak | 0.000083513 mV | 0.1 mV |
| Duration above -20 mV | 0.000000645 ms | 0.01 ms |
| Separate rising or falling phase | 0.000421311 ms | 0.01 ms |

All declared conditions pass. Physical setup, source hashes, and library
hashes are unchanged. Raw mappings are exact. Both input plateau errors
are zero; all arrays are finite and end at 2100 ms. The
[full result](h01-l2-sodium-opening-tolerance-result.json) retains each event
difference and each separate phase difference.

This check applies to opening factor two and density factor one only.
It does not qualify density factor 1.3, spatial resolution, or agreement
with human physiology.
