# SP13 stage 0: PASS (SP12 bands, 5 s tail)

[Direct observations and required charts](stage-0-decision-direct.md)

Inspect these first. The following verdict uses registered QC bands, not a causal or human-qualification verdict.

Human 200 pA late p50 -67.06 mV, mid p50 -67.47 mV.
Reference: h01-e-currents/m0-b3-currents-sweep56. Survivor: s0-ksahp-tail5.

## s0-ksahp-tail5 (dose): 6 of 6 bands

| Row | Predicted | Observed | Held |
| --- | --- | ---: | --- |
| count | 1-2 | 1 | True |
| late_p50_minus_human_mv | |x| <= 1.0 | -0.255 | True |
| mid_p50_minus_human_mv | |x| <= 1.0 | -0.469 | True |
| prespike_max_dev_mv | <= 0.1 | 0.000 | True |
| first_spike_shift_ms | |x| <= 1.0 | 0.000 | True |
| axon_first | True | True | True |

Count 1; peaks [1147.35] ms; levels mid -67.94, late -67.32, post None mV (sample p50; None = unavailable); gate max {'soma_KsAHP_ma_cm2': 0.015330914056383598, 'soma_KsAHP_z': 0.4828959321668311}.

