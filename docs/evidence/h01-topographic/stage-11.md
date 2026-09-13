# SP15 stage 11: the rise dose, and what a spike leaves behind

## (a) Rise against the somatic NaTs density, x2 density-held geometry, 310 pA

Recorded spike-1 rise (sweep 53): 348 V/s.

| density | status | count | rise V/s | take-off mV | approach mV/ms | span mV | move raw mV | move corrected mV |
|---|---|---|---|---|---|---|---|---|
| 0.9 | read | 9 | 655 | -55.46 | 0.63 | 4.1 | ref | ref |
| 0.7 | read | 9 | 586 | -55.36 | 0.61 | 5.0 | +0.11 | +0.07 |
| 0.5 | read | 9 | 505 | -55.32 | 0.59 | 5.4 | +0.14 | +0.06 |

## (b) Short squares 27 to 31 at 1260 pA: primed against unprimed (retained traces)

| sweep | s since last spike | approach mV/ms | take-off mV | minus unprimed mV |
|---|---|---|---|---|
| 27 | 30.7 | 7.18 | -61.88 | +0.00 |
| 28 | 4.2 | 7.20 | -62.03 | -0.16 |
| 29 | 4.2 | 7.30 | -61.31 | +0.56 |
| 30 | 2.7 | 7.25 | -61.56 | +0.31 |
| 31 | 15.8 | 7.23 | -61.66 | +0.22 |

Sigma of the five: 0.28 mV; primed median +0.27 mV at 2.7 to 15.8 s.

## (b) Later-spike residual by spike index, long-square trains (stage-7 tables)

| band | n | median residual mV |
|---|---|---|
| spike_2 | 6 | +1.83 |
| spikes_3_to_5 | 16 | +1.63 |
| spikes_6_plus | 19 | +2.23 |

Slope +0.091 mV per spike over 41 later spikes; spikes 6+ minus spike 2: +0.40 mV.

## Decision

- (a) pass: spike-1 rise falls monotonically over 0.9, 0.7, 0.5 (+150.01)
- (a) pass: take-off at 0.7, corrected for the approach, moves less than 0.5 mV from 0.9 (+0.07; raw +0.11 mV over -0.01 decades of approach)
- (a) pass: count at 0.7 within 8 to 12 (+9.00)
- (a) pass: take-off at 0.5, corrected for the approach, moves less than 0.5 mV from 0.9 (+0.06; raw +0.14 mV over -0.03 decades of approach)
- (a) pass: count at 0.5 within 8 to 12 (+9.00)
- (b) pass: primed short-square take-offs sit within 3 sigma (0.75 mV, the long-square repeat sigma) of the unprimed one (the step does not outlast seconds) (+0.27; the five short squares' own 3 sigma is 0.84 mV; a persisting step would read about +1.9)
- (b) pass: later-spike residual does not grow with the count (spikes 6+ against spike 2 within 3 sigma, 0.75 mV) (+0.40)

Verdict: PASS

## Reading

- (a) On the x2 density-held geometry at 310 pA the spike-1 rise follows the somatic NaTs density monotonically: 655, 586 and 505 V/s at 0.9, 0.7 and 0.5, about 375 V/s per unit of the factor over the range read, and the somatic take-off holds: -55.46, -55.36 and -55.32 mV at 0.63, 0.61 and 0.59 mV/ms (corrected for the approach +0.07 and +0.06 mV; limit 0.5). The count is 9 at every dose, so the somatic density is a rise lever that touches neither the take-off nor the count here. The onset span widens with the dose, 4.1, 5.0 and 5.4 mV, still inside the recorded 3.5 to 5.7. Every registered prediction holds and no rejection fired.
- (a) qualified: the slope is read between 0.5 and 0.9 only. The recorded 348 V/s lies 157 V/s below the lowest dose; carried outside the range at the same slope it would need a factor near 0.1, which is not a reading, only a statement that the somatic density alone does not reach the recorded rise inside the range read while the span stays in band.
- (b) The step does not outlast seconds. Sweep 27 fires unprimed (30.7 s after the last spike, five silent sweeps between); sweeps 28, 29 and 30 each fire 4.2, 4.2 and 2.7 s after one spike and sweep 31 15.8 s after one, all at 7.2 to 7.3 mV/ms; their take-offs sit -0.16, +0.56, +0.31 and +0.22 mV from the unprimed one, median +0.27, where a persisting step would read +1.9. The yardstick is the independent long-square repeat sigma, 0.25 mV (3 sigma 0.75): every primed deviation is inside it; the spread of the five short squares themselves (0.28 mV) only agrees with it and is not the test, since a step present in the primed four would widen it. Inside a train the step is full-sized after one spike (spike 2 residual +1.83 mV, n 6) and does not grow with the count (spikes 3 to 5 +1.63, spikes 6 and later +2.23; 6+ minus 2 is +0.40 mV, limit 0.75; +0.09 mV per spike over 41). So it is a per-spike step, present in full after one spike, not recovering measurably within 0.73 s (stage 9) and gone within 2.7 s.
- (b) limits: the read does not reach inside the first 2.7 s after a spike, and it assumes the step is as visible at 7 mV/ms as at 0.2 to 0.75 (it is read here at -62 mV, where the recorded relation has slid 6 mV from the long-square regime). The registered next-long-square read was not taken: the first spiking long square starts 173 s after sweep 31.
