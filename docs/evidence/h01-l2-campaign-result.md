# Layer-2 family campaign result

Specification: [2026-09-05-h01-l2-family-campaign](../specs/2026-09-05-h01-l2-family-campaign.md).
Manifest: [h01-l2-campaign-fine-manifest.json](h01-l2-campaign-fine-manifest.json). Ranking: [stage-a-ranking.json](h01-l2-campaign-fine/stage-a-ranking.json).
Every residual is model minus human. No physiological allowance is agreed; this page issues no pass.

## Stage 0

Coarse settings (nseg 3) fail the preservation rule on five phase measurements, so every family evaluation ran at nseg 9, CVode 1e-10, through 2140 ms. See [stage0-preservation.json](h01-l2-campaign-fine/stage0-preservation.json).

## Stage A: family dissection

Ranked observations: 50 keys whose source-to-candidate contrast exceeds five numerical decision limits. Pooled normalized RSS order: F3 1.432e+04, F5 1.345e+04, F4 7059, F1 5873, F2 5567. Pooled Steep X: none (no family exceeds the others in quadrature).

| Group | Keys | F1 | F2 | F3 | F4 | F5 | Steep X |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| event count change (into S / out of C) | 1 | -1 / +0 | -1 / +0 | -1 / +1 | -1 / +0 | -1 / +0 | S 7, C 5 |
| intervals | 4 | 362.9 | 1009 | 3832 | 803.8 | 2727 | F3 |
| minima | 5 | 5415 | 4869 | 13.27 | 6003 | 277.1 | none |
| phases | 11 | 575.4 | 111 | 31.51 | 2010 | 126.9 | F4 |
| peaks_and_onsets | 20 | 2169 | 2500 | 1.379e+04 | 3016 | 1.195e+04 | F3 |
| subthreshold | 10 | 0.04956 | 0.8741 | 0.4889 | 35.66 | 5515 | F5 |

## Stage B: paired swap

{
  "evaluated": true,
  "checks": {
    "into_has_five_events": true,
    "into_interval_2_and_4_positive": false,
    "into_subthreshold_near_candidate": true,
    "out_has_extra_events": true,
    "out_late_intervals_negative": true,
    "out_subthreshold_near_source": true
  },
  "holds": false,
  "prediction": "Stated before running: no single family reverses the source-to-candidate change (pooled steep_x null; per group: F3 carries event count and intervals, F5 carries every subthreshold sample, F4 carries phases). Prediction: ab-into-F3F5 shows 5 events, positive interval-2 and interval-4 errors like the candidate, and subthreshold residual near 0.39 mV at 1120 ms; ab-out-F3F5 shows 6 or 7 events, large negative late-interval errors like the source, and subthreshold residual near 1.76 mV. If both hold, F3 and F5 jointly reverse the change and the causal page records an interdependency of two families acting on separate observation groups, not one Steep X."
}

## Stage C: tolerance recheck of the paired swap

{
  "out": {
    "max_change": 0.0004431023614870355,
    "event_count_unchanged": true,
    "within_limits": false,
    "exceeding": {
      "e3_rise_to_peak": {
        "change": -0.0004431023614870355,
        "limit": 0.00041626122720117564,
        "claimed_contrast": 0.00014464928040069935
      },
      "e3_peak_to_fall": {
        "change": 0.00044282545013629715,
        "limit": 0.00041626122720117564,
        "claimed_contrast": 0.001278420986409401
      }
    },
    "max_change_over_claimed_contrast": 3.063287700150171
  }
}

## Residuals

| Run | event_count_model | i1 | i2 | i3 | i4 | m1_v | m2_v | e1_rise_to_peak | e1_peak_v | sub_1120_v | sub_2120_v |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source | 7 | -6.586 | -96.7 | -97.28 | -77.31 | -1.503 | -4.404 | -0.1718 | 1.177 | 1.761 | 2.065 |
| candidate | 5 | -8.324 | 51.57 | -0.2235 | 20.55 | 0.05868 | -2.863 | -0.1216 | -0.07149 | 0.3938 | 1.202 |
| corner | 6 | -7.807 | -70.52 | -94.25 | -71.64 | -2.022 | -4.998 | -0.1225 | 0.2979 | 1.771 | 2.067 |
| into-F1 | 6 | -10.58 | -87.2 | -93.6 | -75.19 | 1.158 | -1.845 | -0.1107 | -5.765 | 1.761 | 2.065 |
| out-F1 | 5 | -4.373 | 46.32 | -0.8912 | 20.25 | -1.852 | -4.73 | -0.1782 | 5.398 | 0.3939 | 1.202 |
| into-F2 | 6 | -12.55 | -76.77 | -94.23 | -74.38 | 0.5681 | -2.389 | -0.1718 | 1.177 | 1.761 | 2.065 |
| out-F2 | 5 | 0.4012 | 24.75 | -5.421 | 15.69 | -2.094 | -4.966 | -0.1216 | -0.07149 | 0.3938 | 1.202 |
| into-F3 | 6 | -6.473 | -66.81 | -39.78 | -22.45 | -1.503 | -4.406 | -0.1715 | 1.177 | 1.761 | 2.065 |
| out-F3 | 6 | -8.45 | -2.919 | -66.91 | -43.33 | 0.05881 | -2.86 | -0.1214 | -0.07155 | 0.3939 | 1.203 |
| into-F4 | 6 | -5.398 | -69.34 | -94.27 | -70.77 | -3.829 | -6.757 | -0.1784 | 5.68 | 1.772 | 2.067 |
| out-F4 | 5 | -12.33 | 49.94 | 1.435 | 20.53 | 3.331 | -0.01228 | -0.1098 | -6.255 | 0.3877 | 1.203 |
| into-F5 | 6 | 3.003 | -43.82 | -73.06 | -50.93 | -1.558 | -4.353 | -0.1707 | 0.7705 | 0.3878 | 1.203 |
| out-F5 | 5 | -13.71 | -5.942 | -31.52 | -10.88 | 0.1344 | -2.937 | -0.1225 | 0.2979 | 1.771 | 2.067 |
| ab-into-F3F5 | 5 | 3.262 | -0.06957 | -8.961 | 10.48 | -1.558 | -4.355 | -0.171 | 0.7706 | 0.3878 | 1.202 |
| ab-out-F3F5 | 6 | -13.78 | -48.88 | -91.42 | -68.42 | 0.1345 | -2.935 | -0.1225 | 0.2979 | 1.771 | 2.067 |
| c-out-F3F5-atol11 | 6 | -13.78 | -48.88 | -91.42 | -68.42 | 0.1345 | -2.935 | -0.1225 | 0.2979 | 1.771 | 2.067 |
