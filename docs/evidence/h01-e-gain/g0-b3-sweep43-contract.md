# Contract scorecard

Every residual is model minus human. Allowances are the approved contract values;
interval rows are derived from two crossing allowances and are diagnostic only.
Tier: target = ratio >= 10, real = ratio >= 2, parked otherwise. A failed row inside its
numerical decision limit is reported as unresolved, not as a pass.

Contract: `docs/specs/2026-09-05-h01-recorded-response-acceptance-proposal.md` (sha256 7965377e5a5b).

## E g0-b3 at E 0.11

Trace `h01-e-gain/g0-b3-sweep43.npz`. Settings: nseg x9, CVode 1e-10, recorded bias, B3 flags (sha256 e4825c83). Verdicts: {'pass': 5, 'fail': 4, 'unavailable': 1}.

| Row | Human | Model | Residual | Allowance | Ratio | Tier | Numerical limit | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| sub_1520_mv | -75.69 | -73.97 | 1.716 | 1 | 1.716 | parked | 4.174e-08 | fail |
| sub_2019_mv | -75.69 | -73.97 | 1.714 | 1 | 1.714 | parked | 7.09e-08 | fail |
| sub_2021_mv | -76.47 | -75.07 | 1.396 | 1 | 1.396 | parked | 8.507e-08 | fail |
| sub_2040_mv | -81.81 | -80.13 | 1.686 | 1 | 1.686 | parked | 3.739e-07 | fail |
| sub_2120_mv | -85.75 | - | - | 1 | - | - | 7.511e-07 | unavailable |

## Targets (ratio >= 10) by cell

- E: none
