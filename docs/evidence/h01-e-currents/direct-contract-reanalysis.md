# SP11 stage 0: B3 current paths (PASS)

[Direct observations and required charts](direct-contract-reanalysis-direct.md)

Inspect those observations first. The tables below are secondary QC summaries, not causal exclusions.

B3 reproduced; direct observations required; no causal exclusion from current shares

| Input | Band | Predicted | Observed | Held |
| --- | --- | --- | --- | --- |
| 200 pA | count | 4 | 4 | yes |
| 200 pA | peak_time_max_dev_ms | <= 0.1 | 0.0 | yes |
| 200 pA | trough_max_dev_mv | <= 0.1 | 0.0 | yes |
| 200 pA | axon_first | True | True | yes |
| 250 pA | count | 8 | 8 | yes |
| 250 pA | peak_time_max_dev_ms | <= 0.1 | 0.0 | yes |
| 250 pA | trough_max_dev_mv | <= 0.1 | 0.0 | yes |
| 250 pA | axon_first | True | True | yes |
| 310 pA | count | 10 | 10 | yes |
| 310 pA | peak_time_max_dev_ms | <= 0.1 | 0.0 | yes |
| 310 pA | trough_max_dev_mv | <= 0.1 | 0.0 | yes |
| 310 pA | axon_first | True | True | yes |

## 200 pA: interspike mean currents (nA, inward positive)

| Window | ms | dV mV | drift | applied | axial | NaTs | Nap | Im | SK | Ih | pas | K_P | K_T | Kv3_1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after spike 1 | 65.4 | 13.52 | 0.0001 | 0.1963 | -0.1877 | 0.0029 | 0.0021 | -0.0003 | -0.0006 | 0.0000 | -0.0069 | -0.0000 | -0.0000 | -0.0058 |
| after spike 2 | 256.2 | 13.69 | 0.0000 | 0.1963 | -0.1883 | 0.0017 | 0.0016 | -0.0001 | -0.0017 | 0.0000 | -0.0066 | -0.0000 | -0.0000 | -0.0029 |
| after spike 3 | 292.0 | 13.83 | 0.0000 | 0.1963 | -0.1882 | 0.0015 | 0.0015 | -0.0001 | -0.0018 | 0.0000 | -0.0065 | -0.0000 | -0.0000 | -0.0027 |
| after spike 4 (tail) | 244.8 | 8.60 | 0.0000 | 0.1963 | -0.1878 | 0.0012 | 0.0013 | -0.0001 | -0.0018 | 0.0000 | -0.0064 | -0.0000 | -0.0000 | -0.0028 |

## 250 pA: interspike mean currents (nA, inward positive)

| Window | ms | dV mV | drift | applied | axial | NaTs | Nap | Im | SK | Ih | pas | K_P | K_T | Kv3_1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after spike 1 | 17.3 | 13.11 | 0.0005 | 0.2463 | -0.2276 | 0.0033 | 0.0021 | -0.0013 | -0.0011 | 0.0000 | -0.0068 | -0.0001 | -0.0000 | -0.0164 |
| after spike 2 | 42.6 | 13.64 | 0.0002 | 0.2463 | -0.2311 | 0.0031 | 0.0021 | -0.0005 | -0.0053 | 0.0000 | -0.0069 | -0.0000 | -0.0000 | -0.0080 |
| after spike 3 | 180.7 | 13.76 | 0.0001 | 0.2463 | -0.2321 | 0.0014 | 0.0014 | -0.0001 | -0.0074 | 0.0000 | -0.0064 | -0.0000 | -0.0000 | -0.0031 |
| after spike 4 | 167.9 | 13.85 | 0.0001 | 0.2463 | -0.2323 | 0.0015 | 0.0014 | -0.0001 | -0.0072 | 0.0000 | -0.0064 | -0.0000 | -0.0000 | -0.0032 |
| after spike 5 | 161.3 | 13.90 | 0.0001 | 0.2463 | -0.2323 | 0.0015 | 0.0014 | -0.0001 | -0.0071 | 0.0000 | -0.0065 | -0.0000 | -0.0000 | -0.0033 |
| after spike 6 | 156.9 | 13.93 | 0.0001 | 0.2463 | -0.2323 | 0.0016 | 0.0014 | -0.0001 | -0.0070 | 0.0000 | -0.0065 | -0.0000 | -0.0000 | -0.0034 |
| after spike 7 | 153.8 | 13.95 | 0.0001 | 0.2463 | -0.2324 | 0.0016 | 0.0015 | -0.0001 | -0.0070 | 0.0000 | -0.0065 | -0.0000 | -0.0000 | -0.0034 |
| after spike 8 (tail) | 35.6 | 5.51 | 0.0001 | 0.2463 | -0.2251 | 0.0007 | 0.0010 | -0.0005 | -0.0087 | 0.0000 | -0.0061 | -0.0000 | -0.0000 | -0.0080 |

## 310 pA: interspike mean currents (nA, inward positive)

| Window | ms | dV mV | drift | applied | axial | NaTs | Nap | Im | SK | Ih | pas | K_P | K_T | Kv3_1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after spike 1 | 10.0 | 12.63 | 0.0008 | 0.3063 | -0.2762 | 0.0032 | 0.0020 | -0.0021 | -0.0015 | 0.0000 | -0.0067 | -0.0001 | -0.0000 | -0.0291 |
| after spike 2 | 20.7 | 13.72 | 0.0004 | 0.3063 | -0.2812 | 0.0034 | 0.0021 | -0.0011 | -0.0089 | 0.0000 | -0.0069 | -0.0001 | -0.0000 | -0.0153 |
| after spike 3 | 117.0 | 13.64 | 0.0001 | 0.3063 | -0.2856 | 0.0019 | 0.0017 | -0.0002 | -0.0135 | 0.0000 | -0.0066 | -0.0000 | -0.0000 | -0.0040 |
| after spike 4 | 131.1 | 13.71 | 0.0001 | 0.3063 | -0.2856 | 0.0017 | 0.0015 | -0.0002 | -0.0136 | 0.0000 | -0.0065 | -0.0000 | -0.0000 | -0.0037 |
| after spike 5 | 122.9 | 13.77 | 0.0001 | 0.3063 | -0.2856 | 0.0018 | 0.0016 | -0.0002 | -0.0134 | 0.0000 | -0.0066 | -0.0000 | -0.0000 | -0.0039 |
| after spike 6 | 119.2 | 13.81 | 0.0001 | 0.3063 | -0.2857 | 0.0018 | 0.0016 | -0.0002 | -0.0134 | 0.0000 | -0.0066 | -0.0000 | -0.0000 | -0.0039 |
| after spike 7 | 116.2 | 13.83 | 0.0001 | 0.3063 | -0.2857 | 0.0019 | 0.0016 | -0.0002 | -0.0133 | 0.0000 | -0.0066 | -0.0000 | -0.0000 | -0.0040 |
| after spike 8 | 114.0 | 13.86 | 0.0001 | 0.3063 | -0.2857 | 0.0019 | 0.0016 | -0.0002 | -0.0133 | 0.0000 | -0.0066 | -0.0000 | -0.0000 | -0.0041 |
| after spike 9 | 112.3 | 13.87 | 0.0001 | 0.3063 | -0.2857 | 0.0019 | 0.0016 | -0.0002 | -0.0132 | 0.0000 | -0.0066 | -0.0000 | -0.0000 | -0.0041 |
| after spike 10 (tail) | 66.3 | 7.05 | 0.0001 | 0.3063 | -0.2828 | 0.0010 | 0.0012 | -0.0003 | -0.0140 | 0.0000 | -0.0063 | -0.0000 | -0.0000 | -0.0053 |

## Plateau contrast (p50 mV; human | model)

| Input | mid pulse 1300-1500 | late pulse 1800-2000 | post pulse 2100-2300 | model R_in (MOhm) | late offset (mV) | offset (nA) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 200 pA | -67.46875265240669 / -64.05678536765974 | -67.06250229477882 / -65.0891958938951 | -85.5937539935112 / None | unavailable | 1.9733064008837289 | unavailable |
| 250 pA | -64.50000175833702 / -64.77067688439986 | -62.1250025331974 / -64.99940900659013 | -86.12500274181366 / None | unavailable | -2.8744064733927246 | unavailable |
| 310 pA | -62.00000041723251 / -64.35921809967417 | -63.4375005364418 / -64.71588164414672 | -85.78125530481339 / None | unavailable | -1.2783811077049165 | unavailable |

## Lever rule (drift at 200 pA 0.00007 nA; plateau offset None nA)

| Lever | mean 200 pA | mean 310 pA late | share 200 | share 310 late | carries drift | selective | carries offset | admissible |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| Nap | 0.0017 | 0.0016 | 0.008 | 0.005 | True | True | None | None |
| Im | -0.0002 | -0.0002 | 0.001 | 0.001 | True | True | None | None |
| SK | -0.0014 | -0.0133 | 0.007 | 0.042 | True | False | None | None |
| Ih | 0.0000 | 0.0000 | 0.000 | 0.000 | False | True | None | None |
| pas | -0.0067 | -0.0066 | 0.033 | 0.021 | True | True | None | None |
| K_P | -0.0000 | -0.0000 | 0.000 | 0.000 | False | True | None | None |
| K_T | -0.0000 | -0.0000 | 0.000 | 0.000 | False | True | None | None |
| Kv3_1 | -0.0038 | -0.0041 | 0.019 | 0.013 | True | True | None | None |
| NaTs | 0.0020 | 0.0019 | 0.010 | 0.006 | True | True | None | None |
| Ca_HVA | 0.0000 | 0.0000 | 0.000 | 0.000 | False | True | None | None |
| Ca_LVA | 0.0001 | 0.0001 | 0.000 | 0.000 | True | True | None | None |
| axial | -0.1881 | -0.2857 | 0.922 | 0.911 | True | True | None | None |

Causal admissibility: not established (not an exclusion).

Current means and shares describe local observations. They cannot exclude interventions or establish a required missing current; voltage and states feed back.
