# I cell holdout prediction, 0.23 nA (sweep active-1)

The prediction was written to `h01-prediction-i.json` before the holdout was
opened; the run is `h01-i-energetic/p-close2-023` (42 s), scored in
`h01-i-energetic/stage-p-scores.json`. Finalist: the candidate with somatic Kv3
close factor 2.0 (opening 0.5×, closing 1× the source), Ca_LVA halved, axonal SK
as in the candidate.

Two limits are reported. "Limit" is the Stage 1 combined repeatability limit
(human repeats and model numerical repeats, Dixon factor), which is what the
search can resolve. "Contract" is the approved engineering requirement from
`docs/specs/2026-09-05-h01-recorded-response-acceptance-proposal.md`: exact
count, 1 mV on every peak and recovery-minimum voltage; it has no rate,
threshold or cycle-length rows. A repeatability pass is not a contract pass.

| Row | Predicted | Human | Model | Limit | Band | Result (limit) | Contract |
| --- | --- | --- | --- | --- | --- | --- | --- |
| spikes | 24 to 28 | 31 | 26 | exact | inside | fail, as predicted | FAIL (exact) |
| spike 1 minimum (mV) | −79.5 ± 0.8 | −78.9 | −79.4 | 1.24 | inside | pass | pass (0.5 of 1 mV) |
| spike 1 max fall (V/s) | −345 ± 5 | −327 | −345 | 19.5 | inside | pass | no row |
| spike 1 max rise (V/s) | 640 ± 10 | 604 | 643 | 245 (mesh) | inside | pass | no row |
| spike 1 peak (mV) | 17.7 ± 0.3 | 18.9 | 17.8 | 2.07 | inside | pass | **FAIL (1.15 of 1 mV)** |
| spike 1 threshold (mV) | −60.9 ± 0.3 | −61.0 | −60.9 | 1.29 | inside | pass | no row |
| cycle 2 (ms) | 22 to 27 | 6.3 | 20.7 | 12.8 | **outside, by 1.3** | fail, as predicted | no row |
| late minimum (mV) | −80.4 ± 0.5 | −79.6 | −80.4 | 1.24 | inside | pass | pass (0.8 of 1 mV) |
| late peak (mV) | not claimed | 16.8 | 17.0 | 2.07 | none | pass | pass (0.2 of 1 mV) |
| late threshold (mV) | −60.5 | −55.2 | −60.4 | 1.29 | inside | fail, as predicted | no row |
| late max fall (V/s) | not claimed | −294 | −334 | 19.5 | none | fail, not predicted | no row |
| late cycle (ms) | 45 to 55 | 40.7 | 43.6 | 12.8 | **outside, by 1.4** | pass | no row |

**Bands.** Nine of the eleven predicted values fell inside their bands; the two
cycle-length bands (cycle 2 and late cycle) were both set 1 to 2 ms too high.
An earlier version of this page said every value fell inside its band; that
sentence was wrong and is withdrawn.

**Explanation.** Sufficient, at repeatability resolution, for the rows it claims
(the elemental loop and the trough at every spike) and insufficient, as stated
in advance, for the early burst, the count and the accommodation along the
train. The late fall rate belongs to the accommodation family and was not
predicted; it is added to the returned question.

**Contract.** The holdout FAILS the approved contract: the count is 26 against
31 and the first peak misses 1 mV by 0.15 mV. The recovery minima pass. The
finalist is therefore an experimental candidate, not an accepted I cell, and
its transfer gate has not been run.

**Holdout status.** Every earlier PV page records this trace as reserved and
unread; the contract names it the prospective fitting holdout. It was opened
once here and is now spent: any later I fit that uses 0.23 nA is a fit, and the
I cell has no closed holdout left. Only sweeps not yet read (none are
identified) could serve as a new one.
