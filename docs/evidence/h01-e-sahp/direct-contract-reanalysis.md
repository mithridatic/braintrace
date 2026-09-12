# SP12 stage 0: FAIL

[Direct observations and required charts](direct-contract-reanalysis-direct.md)

Inspect these first. The following verdict uses registered QC bands, not a causal or human-qualification verdict.

Human 200 pA late p50 -67.06 mV, mid p50 -67.47 mV.
Reference: h01-e-currents/m0-b3-currents-sweep56. Survivor: None.

Nap removal broke its registered prediction; inspect its direct response before making a mechanistic claim.

## s0-ksahp-a (dose): 5 of 6 bands

| Row | Predicted | Observed | Held |
| --- | --- | ---: | --- |
| count | 1-2 | 1 | True |
| late_p50_minus_human_mv | |x| <= 1.0 | 1.258 | False |
| mid_p50_minus_human_mv | |x| <= 1.0 | -0.466 | True |
| prespike_max_dev_mv | <= 0.1 | 0.000 | True |
| first_spike_shift_ms | |x| <= 1.0 | 0.000 | True |
| axon_first | True | True | True |

Count 1; peaks [1147.35] ms; levels mid -67.93, late -65.80, post None mV (sample p50; None = unavailable); gate max {'soma_KsAHP_ma_cm2': 0.017603155108441088, 'soma_KsAHP_z': 0.5096363416996305}.

## s0-ksahp-b (dose): 4 of 6 bands

| Row | Predicted | Observed | Held |
| --- | --- | ---: | --- |
| count | 1-2 | 1 | True |
| late_p50_minus_human_mv | |x| <= 1.0 | -3.252 | False |
| mid_p50_minus_human_mv | |x| <= 1.0 | -6.007 | False |
| prespike_max_dev_mv | <= 0.1 | 0.000 | True |
| first_spike_shift_ms | |x| <= 1.0 | 0.000 | True |
| axon_first | True | True | True |

Count 1; peaks [1147.35] ms; levels mid -73.48, late -70.31, post None mV (sample p50; None = unavailable); gate max {'soma_KsAHP_ma_cm2': 0.05182071342507987, 'soma_KsAHP_z': 0.5085512468965286}.

## s0-nap-zero (control): 0 of 2 bands

| Row | Predicted | Observed | Held |
| --- | --- | ---: | --- |
| count | >= 3 | 0 | False |
| late_p50_minus_b3_mv | -1.5 to 0 | 2.036 | False |

Count 0; peaks [] ms; levels mid -63.26, late -63.05, post None mV (sample p50; None = unavailable); gate max {}.

