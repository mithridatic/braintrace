# Density candidate passes the declared spatial comparison

Increasing the mesh from 1239 to 3717 segments preserves five positive-peak
spikes and passes all declared direct-response limits.

| Observation | Maximum absolute change | Limit |
| --- | --- | --- |
| Onset | 0.035694332 ms | 0.1 ms |
| Peak | 0.010272916 mV | 0.1 mV |
| Duration above -20 mV | 0.000073660 ms | 0.01 ms |
| Separate rising or falling phase | 0.000346667 ms | 0.01 ms |

Every section has three times as many segments in the finer run. Topology
and physical parameters agree. The largest section-area difference is
1.13687e-13 square micrometres, within the declared roundoff allowance.
Regional areas and distributed Ih parameters match exactly. Mechanism
and library hashes match.

The new coarse-run metadata include kv3_closing_factor=null; the older
fine-run metadata omit that field. Both use the identical original Kv3
mechanism and library. This is the predeclared schema allowance; no Kv3
closing intervention is present in this comparison.

Raw mappings are exact, input plateau errors are zero, arrays are finite,
and both traces end at 2100 ms. The
[full result](h01-l2-density130-spatial-result.json) retains all individual
spike and phase differences. This check and the separate tolerance result
support the specified spike observations at the candidate settings.
They do not qualify the subthreshold numerical response, the Kv3
intervention, or agreement with human physiology.
