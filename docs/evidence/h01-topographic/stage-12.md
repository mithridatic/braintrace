# SP15 stage 12: the recording's measurement chain, measured and applied

## E chain

Chain applied: pipette pole from the edges, acquisition corner at an assumed conservative 10 kHz (the noise floor bounds the corner only from below; its 20 kHz reading beside it).
Noise floor (16 unstimulated seconds, rms 0.078 mV), dB against 1 to 2 kHz: 1 kHz +1.0, 2 kHz -0.8, 5 kHz -2.8, 10 kHz -3.5, 15 kHz -3.8, 20 kHz -4.1, 24 kHz -4.0; slope 10 to 20 kHz -0.7 dB per octave (a 4-pole corner below 10 kHz would give -24); corner applied 20 kHz.
Onset edges of sweeps 15, 16, 17, 18, 22, 23, 24, 25, 26, joint fit at 10 kHz: tau_p 4.7 us, command pole 120 us, t0 -20 us, R_s - R_b +0.24 MOhm, rms 0.097 mV.
Profile: best tau_p 5 us, bound (rms within 20 percent of the minimum) 30 us.

| tau_p us | command pole us | rms mV |
|---|---|---|
| 0 | 97 | 0.100 |
| 1 | 99 | 0.100 |
| 2 | 85 | 0.098 |
| 5 | 127 | 0.097 |
| 10 | 249 | 0.098 |
| 20 | 500 | 0.100 |
| 30 | 500 | 0.105 |
| 50 | 500 | 0.138 |
| 75 | 500 | 0.207 |
| 100 | 500 | 0.291 |

## I chain

Chain applied: pipette pole from the edges, acquisition corner at an assumed conservative 10 kHz (the noise floor bounds the corner only from below; its 20 kHz reading beside it).
Noise floor (17 unstimulated seconds, rms 0.203 mV), dB against 1 to 2 kHz: 1 kHz +1.4, 2 kHz -0.9, 5 kHz -2.6, 10 kHz -3.9, 15 kHz -3.4, 20 kHz -3.8, 24 kHz -4.5; slope 10 to 20 kHz +0.1 dB per octave (a 4-pole corner below 10 kHz would give -24); corner applied 20 kHz.
Onset edges of sweeps 7, 8, 9, 11, 12, 13, joint fit at 10 kHz: tau_p 0.4 us, command pole 36 us, t0 +15 us, R_s - R_b -0.96 MOhm, rms 0.050 mV.
Profile: best tau_p 10 us, bound (rms within 20 percent of the minimum) 30 us.

| tau_p us | command pole us | rms mV |
|---|---|---|
| 0 | 0 | 0.053 |
| 1 | 0 | 0.053 |
| 2 | 0 | 0.051 |
| 5 | 48 | 0.054 |
| 10 | 157 | 0.050 |
| 20 | 203 | 0.052 |
| 30 | 240 | 0.055 |
| 50 | 410 | 0.083 |
| 75 | 500 | 0.110 |
| 100 | 500 | 0.145 |

## E spike 1 at 310 pA, raw and through the chain

| trace | count | rise V/s | peak mV | fall V/s | span mV | rapidness /ms | take-off mV | approach mV/ms |
|---|---|---|---|---|---|---|---|---|
| x2 density held, raw | 9 | 655 | 39.2 | -91 | 4.10 | 16 | -55.46 | 0.63 |
| x2 density held, chain at 10 kHz | 9 | 609 | 39.0 | -91 | 5.45 | 17 | -55.38 | 0.63 |
| x2 density held, chain at the noise floor 20 kHz | 9 | 638 | 39.1 | -91 | 5.74 | 15 | -55.53 | 0.63 |
| B3 control 1 um, raw | 10 | 639 | 38.5 | -95 | 3.39 | 41 | -57.20 | 0.58 |
| B3 control 1 um, chain at 10 kHz | 10 | 570 | 38.2 | -94 | 4.80 | 34 | -57.29 | 0.58 |
| B3 control 1 um, chain at the noise floor 20 kHz | 10 | 615 | 38.4 | -94 | 5.24 | 38 | -57.26 | 0.58 |
| x2 density held at the peak bound (tau_p 60 us) | 9 | 486 | 35.2 | -88 | 4.28 | 22 | -55.45 | 0.63 |
| recording sweep 53 | 10 | 348 | 35.3 | -104 | 4.50 | 39 | -56.41 | 0.63 |

## E slow set on the x2 density-held ramps, raw against chain

- raw: soma take-off -55.19, -55.74, -57.27 mV (slide -2.08); approach 0.45, 0.99, 2.50; axon lead 0.20, 0.20, 0.18 ms; span 5.3, 4.1, 6.0
- chain: soma take-off -55.31, -55.86, -57.40 mV (slide -2.08); approach 0.45, 0.99, 2.48; axon lead 0.18, 0.18, 0.16 ms; span 4.8, 5.7, 5.6

## I onset through the I chain

| trace | count | rise V/s | peak mV | fall V/s | span mV | rapidness /ms | take-off mV | approach mV/ms |
|---|---|---|---|---|---|---|---|---|
| I fit, raw | 14 | 592 | 17.2 | -338 | 13.54 | 5 | -60.54 | 0.61 |
| I fit, chain | 14 | 530 | 15.1 | -326 | 12.80 | 5 | -60.64 | 0.60 |
| I recording 0.19 nA | 12 | 597 | 19.6 | -336 | 1.88 | 51 | -60.31 | 1.13 |

## Decision

- E (a) pass: the noise floor bounds the acquisition corner (slope 10 to 20 kHz flatter than half a 4-pole roll-off) (-0.685; noise-floor corner 20 kHz, chain applied at the conservative 10 kHz; baseline rms 0.078 mV over 16 sweeps)
- E (a) pass: the edges bound tau_p to within a factor of two of the acquisition corner (+30; best tau_p 5 us (joint fit 4.7), command pole 120 us, rms 0.097 mV, R_s - R_b +0.24 MOhm)
- I (a) pass: the noise floor bounds the acquisition corner (slope 10 to 20 kHz flatter than half a 4-pole roll-off) (+0.0731; noise-floor corner 20 kHz, chain applied at the conservative 10 kHz; baseline rms 0.203 mV over 17 sweeps)
- I (a) pass: the edges bound tau_p to within a factor of two of the acquisition corner (+30; best tau_p 10 us (joint fit 0.4), command pole 36 us, rms 0.050 mV, R_s - R_b -0.96 MOhm)
- E (b) FAIL: the chain accounts for less than half of the rise gap under the peak bound (+0.549; measured chain accounts for 15 percent at 10 kHz (rise 609) and 5 percent at the noise floor (rise 638) against the recorded 348; peak bound at tau_p 60 us gives rise 486)
- E (b) pass: the residual rise gap exceeds 3 sigma of the within-chain repeat (4.5 V/s) (+262)
- E (d) pass: E fit onset span through the chain within 1 mV of the recording's (+0.953; raw 4.10, chain 5.45, recorded 4.50; rapidness 16 / 17 / 39 per ms)
- E (c) FAIL: slow set held: take-off moves less than 0.1 mV (+0.128)
- E (c) pass: slow set held: approach within 5 percent (+0.00524)
- E (c) pass: slow set held: axon lead within 0.02 ms (+0.02)
- E (c) pass: slow set held: 310 pA count unchanged (+0)
- I (d) pass: I fit onset span through the chain stays above 8 mV (stage-8 reading survives its chain) (+12.8; raw 13.5, recorded 1.9 mV; rapidness 5 / 5 / 51 per ms)

Verdict: FAIL

## Reading

- The chain is measured where it can be and bounded where it cannot. The onset edges show a jump of at most two samples and no transient larger than 0.4 mV at 1.2 nA, where a live 5 pF pipette pole would put 9 mV: the capacitance neutralisation is live in practice whatever the metadata field holds, and the pipette pole reads 5 us (E) and 2 us (I) at best, profile flat to 30 us because the edge measures the round trip and the command's own pole (E 150 us, I 44 us) absorbs it. The unstimulated noise floor is flat to Nyquist in both files (E -0.7 dB per octave between 10 and 20 kHz, I +0.1, against -24 for a 4-pole corner below 10 kHz), which reads as a corner at or above 20 kHz if the floor is pipette noise and says nothing if it is the digitiser's; the chain is therefore applied at an assumed 10 kHz corner as the conservative case, with the 20 kHz reading beside it.
- Through the chain the E fit's spike-1 rise moves from 655 to 610 V/s at 10 kHz and 638 at 20 kHz (the control 639 to 570 and 615): 5 to 15 percent of the gap to the recorded 348, 17 to 45 V/s, which is 11 to 30 sigma of the within-chain repeat and changes no decision. The residual gap, 260 to 290 V/s, is real. The bound independent of the edges, the pole at which the filtered peak meets the recorded 35.3 mV, would cover about half of the gap; the registered 'less than half' fired by the letter at that bound, not at the measurement, which is an order of magnitude smaller. A third leg: the recording repolarises faster than the fit (-104 against -91 and -95 V/s) while it rises slower; a chain that halved the rise would have halved the fall.
- The slow set holds through the chain: the take-off shifts by a constant 0.13 mV at 10 kHz (the registered 0.1 fired by the letter; the slide across the ramps, the quantity stage 10 decided on, holds to 0.01 mV), approach within 1 percent, axon lead within 0.02 ms, count unchanged; stages 7 to 11 stand. The I fit's rise, equal to the recording's when read raw (592 against 597), reads 530 through the 10 kHz chain: that match was a coincidence of observation points, and the I fit rises about a tenth too slowly when both are read through the chain. The E onset span does not hold: it moves by more than 1 mV and non-monotonically across the ramps under a chain that moves the rise by a few percent; the 10 to 100 V/s span is a fragile reading at a 0.6 mV/ms approach and the stage-8 E cell (no discrimination) gains that qualification. The I fit's span survives its chain (13.5 against the recorded 1.9 mV), so the stage-8 I reading stands.
- First execution, recorded: with the acquisition corner free the registered edge fit returned 2.65 kHz (I 3.3), which applied to the fit gave a rise of 384 V/s against the recorded 348 and would have read the whole gap as the chain. The noise floor refutes that corner; the low value was the command's own pole, which a step edge cannot separate from the output filter. The registration is amended: the pipette pole from the transient's area, the corner from the noise floor where it is pipette noise and from an assumed conservative 10 kHz otherwise.
- Rule, independent of the outcome and now in the causal model: a fast model quantity is comparable to a recorded one only through the recording's own chain (here a 3 to 7 percent correction to every rise), and a repeat sigma bounds repeatability, not bias between the two chains.
