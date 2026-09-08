# h01-e-campaign2 result

Second bounded excitatory campaign: the 2^3 matrix over F1, F2, F4 on the S+F3+F5 base, judged per group against the approved contract. Specification: docs/specs/2026-09-06-h01-progressive-search-plan.md.

Cap 24 evaluations; abort 900 s per run; inputs sweep50, sweep43.

**Verdict.** FAIL at 6 of 24 evaluations: no cell of the F1 x F2 x F4 matrix on the S+F3+F5 base passes any contract group; the late subthreshold samples (1520-2120 ms, +1.2 to +1.7 mV) are identical in all eight cells, so no dose of F1, F2, or F4 can pass. Stage B (dose split) was not run because it cannot change the verdict.

Container runs: 12, total 6951.7 s, longest 739.5 s.

## stage-a-decision.json

Decision: no cell passes every group; dose member F1 (largest minima change per interval change)

| Cell | Count | Groups | i2 (ms) | Minima (mV) |
| --- | --- | --- | ---: | --- |
| 000 | {"sweep50:event_count": 5} | {"count": "pass", "onsets": "fail", "peaks": "fail", "phases": "fail", "intervals": "fail", "minima": "fail", "subthreshold": "fail"} | -0.07 | [-1.558, -4.355, -1.289, -1.251] |
| 001 | {"sweep50:event_count": 5} | {"count": "pass", "onsets": "fail", "peaks": "fail", "phases": "fail", "intervals": "fail", "minima": "fail", "subthreshold": "fail"} | 24.29 | [-3.913, -6.755, -3.707, -3.665] |
| 010 | {"sweep50:event_count": 5} | {"count": "pass", "onsets": "fail", "peaks": "fail", "phases": "fail", "intervals": "fail", "minima": "fail", "subthreshold": "fail"} | 25.89 | [0.511, -2.319, 0.708, 0.731] |
| 011 | {"sweep50:event_count": 5} | {"count": "pass", "onsets": "fail", "peaks": "fail", "phases": "fail", "intervals": "fail", "minima": "fail", "subthreshold": "fail"} | 46.32 | [-1.852, -4.73, -1.719, -1.69] |
| 100 | {"sweep50:event_count": 5} | {"count": "pass", "onsets": "fail", "peaks": "fail", "phases": "fail", "intervals": "fail", "minima": "fail", "subthreshold": "fail"} | 16.77 | [1.131, -1.726, 1.349, 1.388] |
| 101 | {"sweep50:event_count": 5} | {"count": "pass", "onsets": "fail", "peaks": "pass", "phases": "fail", "intervals": "fail", "minima": "fail", "subthreshold": "fail"} | 24.75 | [-2.094, -4.966, -1.913, -1.872] |
| 110 | {"sweep50:event_count": 5} | {"count": "pass", "onsets": "fail", "peaks": "fail", "phases": "fail", "intervals": "fail", "minima": "fail", "subthreshold": "fail"} | 49.94 | [3.331, -0.012, 1.675, 2.311] |
| 111 | {"sweep50:event_count": 5} | {"count": "pass", "onsets": "fail", "peaks": "pass", "phases": "fail", "intervals": "fail", "minima": "fail", "subthreshold": "fail"} | 51.57 | [0.059, -2.863, 0.157, 0.184] |

| Member | Minima improvement (mV) | Interval worsening (ms) | Ratio |
| --- | ---: | ---: | ---: |
| F1 | 0.8576332651848162 | 11.614991309240793 | 0.07383847670230198 |
| F2 | 1.1302609175731941 | 26.960504378678195 | 0.04192284022946784 |
| F4 | -1.030850770317155 | 13.567050620683005 | None |

## stage-b-decision.json

Decision: FAIL at 6 of 24 evaluations: no cell of the F1 x F2 x F4 matrix on the S+F3+F5 base passes any contract group; the late subthreshold samples (1520-2120 ms, +1.2 to +1.7 mV) are identical in all eight cells, so no dose of F1, F2, or F4 can pass. Stage B (dose split) was not run because it cannot change the verdict.

| Member | Minima improvement (mV) | Interval worsening (ms) | Ratio |
| --- | ---: | ---: | ---: |
| F1 | 0.8576332651848162 | 11.614991309240793 | 0.07383847670230198 |
| F2 | 1.1302609175731941 | 26.960504378678195 | 0.04192284022946784 |
| F4 | -1.030850770317155 | 13.567050620683005 | None |
