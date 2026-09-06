# E cell holdout prediction, sweep 53 (310 pA)

The prediction was written to `h01-prediction-e.json` before sweep 53 was exported
(`h01_l2_sweep_export.py`, verified to reproduce sweeps 43 and 50 exactly) and
run once as `h01-e-energetic/p-candidate-sweep53` (827 s). No E explanation
passed Stage 3, so this tests the candidate as it stands.

"Limit" is the Stage 1 combined repeatability limit; "Contract" is the approved
engineering requirement (`docs/specs/2026-09-05-h01-recorded-response-acceptance-proposal.md`:
exact count, 1 mV on peak and recovery-minimum voltages; no rate, threshold or
cycle rows).

| Row | Predicted | Human | Model | Limit | Band | Result (limit) | Contract |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spikes | 7 to 9 | 10 | 6 | exact | **outside, by 1** | fail as predicted | FAIL (exact) |
| spike 1 max rise (V/s) | 640 ± 15 | 348 | 670 | 5.7 | **outside, by 15** | fail as predicted | no row |
| spike 1 peak (mV) | 36 ± 1, PASS | 35.3 | 36.9 | 0.55 | inside | **fail; prediction wrong** | FAIL (1.56 of 1 mV) |
| spike 1 threshold (mV) | −52 ± 0.5 | −56.4 | −52.3 | 0.97 | inside | fail as predicted | no row |
| spike 1 minimum (mV) | −70 ± 1, PASS | −69.6 | −69.3 | 1.01 | inside | pass as predicted | pass (0.2 of 1 mV) |
| spike 1 max fall (V/s) | −96 ± 3 | −104 | −96 | 5.7 | inside | fail as predicted | no row |
| late peak (mV) | not claimed | 32.7 | 34.9 | 0.55 | none | fail | FAIL (2.25 of 1 mV) |
| late minimum (mV) | not claimed | −69.9 | −70.1 | 1.01 | none | pass | pass (0.2 of 1 mV) |
| late cycles (ms) | not claimed | 110 to 124 | 196 to 264 | model-limited | none | fail, not predicted | no row |

**Bands.** Four of seven predicted values fell inside their bands. The count
band missed by one, the rise band by 15 V/s, and the peak was predicted to pass
against a 0.55 mV limit and failed by 1.6 mV. These are recorded against the
prediction, not the model.

**Verdict.** The structural verdict of Stage Y holds on the holdout: rise 1.9
times the human's, threshold 4 mV high, fall too slow, and a train that thins to
half the human's rate late in the pulse. Under the contract the holdout FAILS on
count and on both peak voltages; the recovery minima pass. The candidate is not
an accepted E cell.

**Holdout status.** Sweep 53 is now opened and spent. Any later E fit that reads
it is a fit; no other unread long-square sweep has been identified as a
replacement.
