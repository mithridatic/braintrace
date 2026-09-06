# I cell holdout prediction, 0.23 nA (sweep active-1)

The prediction was written to `h01-prediction-i.json` before the holdout was
opened; the run is `h01-i-energetic/p-close2-023` (42 s), scored in
`h01-i-energetic/stage-p-scores.json`. Finalist: the candidate with somatic Kv3
close factor 2.0 (opening 0.5×, closing 1× the source), Ca_LVA halved, axonal SK
as in the candidate. Limits are the Stage 1 combined limits.

| Row | Predicted | Human | Model | Limit | Result |
| --- | --- | --- | --- | --- | --- |
| spikes | 24 to 28, FAIL | 31 | 26 | exact | fail, as predicted |
| spike 1 minimum (mV) | −79.5 ± 0.8 | −78.9 | −79.4 | 1.24 | PASS |
| spike 1 max fall (V/s) | −345 ± 5 | −327 | −345 | 19.5 | PASS |
| spike 1 max rise (V/s) | 640 ± 10 | 604 | 643 | 245 (mesh) | PASS |
| spike 1 peak (mV) | 17.7 ± 0.3 | 18.9 | 17.8 | 2.07 | PASS |
| spike 1 threshold (mV) | −60.9 ± 0.3 | −61.0 | −60.9 | 1.29 | PASS |
| cycle 2 (ms) | 22 to 27, FAIL | 6.3 | 20.7 | 12.8 | fail, as predicted |
| late minimum (mV) | −80.4 ± 0.5 | −79.6 | −80.4 | 1.24 | PASS |
| late threshold (mV) | −60.5, FAIL | −55.2 | −60.4 | 1.29 | fail, as predicted |
| late max fall (V/s) | not claimed | −294 | −334 | 19.5 | fail, not predicted |
| late cycle (ms) | 45 to 55 | 40.7 | 43.6 | 12.8 | PASS |

Every predicted value fell inside its stated band. The explanation is sufficient
for the rows it claims (the elemental loop and the trough at every spike) and
insufficient, as stated in advance, for the early burst, the count and the
accommodation along the train. The late fall rate belongs to the accommodation
family (the human's late spike is both higher-threshold and slower) and was not
predicted; it is added to the returned question.
