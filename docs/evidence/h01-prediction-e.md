# E cell holdout prediction, sweep 53 (310 pA)

The prediction was written to `h01-prediction-e.json` before sweep 53 was exported
(`h01_l2_sweep_export.py`, verified to reproduce sweeps 43 and 50 exactly) and
run once as `h01-e-energetic/p-candidate-sweep53` (827 s). No E explanation
passed Stage 3, so this tests the candidate as it stands. Limits are the Stage 1
combined limits.

| Row | Predicted | Human | Model | Limit | Result |
| --- | --- | --- | --- | --- | --- |
| spikes | 7 to 9, FAIL | 10 | 6 | exact | fail as predicted; band missed by 1 |
| spike 1 max rise (V/s) | 640 ± 15, FAIL | 348 | 670 | 5.7 | fail as predicted; band missed by 15 |
| spike 1 peak (mV) | 36 ± 1, PASS | 35.3 | 36.9 | 0.55 | fail; prediction wrong |
| spike 1 threshold (mV) | −52 ± 0.5, FAIL | −56.4 | −52.3 | 0.97 | fail as predicted |
| spike 1 minimum (mV) | −70 ± 1, PASS | −69.6 | −69.3 | 1.01 | PASS as predicted |
| spike 1 max fall (V/s) | −96 ± 3, FAIL | −104 | −96 | 5.7 | fail as predicted |
| late cycles (ms) | not claimed | 110 to 124 | 196 to 264 | model-limited | fail, not predicted |

The structural verdict of Stage Y holds on the holdout: rise 1.9 times the
human's, threshold 4 mV high, fall too slow, and a train that thins to half the
human's rate late in the pulse. Two of seven bands were wrong (rise by 15 V/s,
peak by 1.6 mV against a 0.55 mV limit), which is recorded against the
prediction, not the model.
