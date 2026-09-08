# Contract scorecard

Every residual is model minus human. Allowances are the approved contract values;
interval rows are derived from two crossing allowances and are diagnostic only.
Tier: target = ratio >= 10, real = ratio >= 2, parked otherwise. A failed row inside its
numerical decision limit is reported as unresolved, not as a pass.

Contract: `docs/specs/2026-09-05-h01-recorded-response-acceptance-proposal.md` (sha256 7965377e5a5b).

## E g1-ih-half at E 0.11

Trace `h01-e-gain/g1-ih-half-sweep43.npz`. Settings: nseg x9, CVode 1e-10, recorded bias, B3 flags with ih_density_factor 37.5. Verdicts: {'fail': 4, 'pass': 5, 'unavailable': 1}.

| Row | Human | Model | Residual | Allowance | Ratio | Tier | Numerical limit | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| sub_1019_mv | -84.5 | -85.7 | -1.201 | 1 | 1.201 | parked | 2.521e-09 | fail |
| sub_1040_mv | -78.53 | -79.56 | -1.029 | 1 | 1.029 | parked | 3.209e-07 | fail |
| sub_1520_mv | -75.69 | -74.65 | 1.041 | 1 | 1.041 | parked | 4.174e-08 | fail |
| sub_2019_mv | -75.69 | -74.65 | 1.039 | 1 | 1.039 | parked | 7.09e-08 | fail |
| sub_2120_mv | -85.75 | - | - | 1 | - | - | 7.511e-07 | unavailable |

## Targets (ratio >= 10) by cell

- E: none
