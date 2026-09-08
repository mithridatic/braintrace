# Contract scorecard

Every residual is model minus human. Allowances are the approved contract values;
interval rows are derived from two crossing allowances and are diagnostic only.
Tier: target = ratio >= 10, real = ratio >= 2, parked otherwise. A failed row inside its
numerical decision limit is reported as unresolved, not as a pass.

Contract: `docs/specs/2026-09-05-h01-recorded-response-acceptance-proposal.md` (sha256 7965377e5a5b).

## E g2-leak-150 at E 0.11

Trace `h01-e-gain/g2-leak-150-sweep43.npz`. Settings: nseg x9, CVode 1e-10, recorded bias, B3 flags with leak_factor 1.5. Verdicts: {'pass': 3, 'fail': 6, 'unavailable': 1}.

| Row | Human | Model | Residual | Allowance | Ratio | Tier | Numerical limit | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| sub_1040_mv | -78.53 | -79.59 | -1.055 | 1 | 1.055 | parked | 3.209e-07 | fail |
| sub_1120_mv | -74.19 | -77.47 | -3.278 | 1 | 3.278 | real | 7.415e-07 | fail |
| sub_1520_mv | -75.69 | -77.75 | -2.061 | 1 | 2.061 | real | 4.174e-08 | fail |
| sub_2019_mv | -75.69 | -77.75 | -2.061 | 1 | 2.061 | real | 7.09e-08 | fail |
| sub_2021_mv | -76.47 | -78.83 | -2.362 | 1 | 2.362 | real | 8.507e-08 | fail |
| sub_2040_mv | -81.81 | -83.19 | -1.378 | 1 | 1.378 | parked | 3.739e-07 | fail |
| sub_2120_mv | -85.75 | - | - | 1 | - | - | 7.511e-07 | unavailable |

## Targets (ratio >= 10) by cell

- E: none
