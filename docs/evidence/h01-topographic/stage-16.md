# Stage 16: the threshold climb at the initiation site

Part 1 refused the somatic dose (outward current accumulates across the train while the take-off
does not move; at take-off the axon delivers ~0.33 nA against a ~0.012 nA somatic brake).
Part 2 doses the axonal sodium's recovery on the stage-10 thickened-stub geometry.

Verdict: **MIXED**

## The characteristic curve (axonal sodium recovery factor -> threshold climb)

| recovery factor | climb spike 2 (mV) | climb spike 5 (mV) | count | spike-1 rise (V/s) | rise 5/1 |
|---|---|---|---|---|---|
| 1.0 | -0.07 | +0.14 | 9 | 609 | 0.94 |
| 0.5 | -3.28 | - | 3 | 609 | - |
| 0.25 | +0.89 | - | 2 | 609 | - |
| 0.125 | +2.39 | - | 2 | 609 | - |

## Coupling control (same strongest dose, default 1 um stub)

climb spike 2 = -2.56 mV, count 7, rise 570 V/s

Recorded: climb +1.38 / +3.63 mV, rise ratio 0.86, count 10. B3: -0.02 / +0.24. HL23PYR: +1.38 / +1.78.

## Percent accuracy (ten elements, the stage-15 definition)

| element | recorded | B3 | best climb dose | B3 % | climb % |
|---|---|---|---|---|---|
| rise_v_s | 347.66 | 570.42 | 609.39 | 36 | 25 |
| fall_v_s | -103.91 | -94.16 | -91.04 | 91 | 88 |
| peak_mv | 35.34 | 38.17 | 38.96 | 97 | 96 |
| threshold_mv | -56.41 | -57.29 | -55.38 | 99 | 99 |
| take_off_mv | -56.41 | -57.29 | -55.38 | 99 | 99 |
| count | 10.00 | 10.00 | 9.00 | 100 | 90 |
| rest_mv | -83.91 | -83.97 | -84.13 | 100 | 100 |
| climb_2_mv | 1.38 | -0.02 | -0.07 | 0 | 0 |
| climb_5_mv | 3.63 | 0.24 | 0.14 | 7 | 4 |
| rise_5_over_1 | 0.86 | 0.95 | 0.94 | 89 | 91 |
| **mean** | | | | **71.8** | **69.1** |

## Cells

- (c) FAIL: some dose reaches at least half the recorded first-interval step (+0.69 of +1.38 mV), and the climb is monotone as the recovery factor falls -> {'1.0': -0.07, '0.5': -3.28, '0.25': 0.89, '0.125': 2.39}. recorded +1.38; B3 -0.02; HL23PYR +1.38
- (d) FAIL: at the best dose the count stays within 5 to 15 and the spike-1 rise within 15 percent of the arm's control -> {'dose': '0.125', 'count': 2, 'rise_v_s': 609.3884305683155, 'control_rise_v_s': 609.3884305042917}. the climb must not be bought by crippling the cell
- (f) PASS: the same strongest dose on the default 1 um stub gives at most a fraction of the coupled arm's climb -> {'coupled_2um': 2.39, 'stub_1um': -2.56}. if they are equal the coupling is not required and stage 10's account of the take-off is wrong
- (g) --: percent accuracy of the best climb dose beside B3's 71.8 -> {'b3_pct': 71.8, 'best_climb_pct': 69.1}. reported; this arm's baseline rise is further from the recording than plain B3's

## Reading

- The climb IS reachable at the initiation site, and it is NOT reachable from the soma. Falling axonal sodium recovery moves the first-interval threshold step from -0.07 mV (control) through +0.89 to +2.39 mV, past the recorded +1.38: the site can move the threshold by more than the recording asks, where part 1 showed no somatic conductance can move it at all.
- The coupling is required, and that is the cell that passed. The same strongest dose on the default 1 um Allen stub gives -2.56 mV, the opposite sign to the coupled arm's +2.39 with the 2 um stub. So the site's accommodation reaches the soma only through the thickened coupling, which confirms stage 10's account of the take-off rather than assuming it.
- But the climb cannot be separated from the spike count in this model family. Every dose that moves the threshold collapses the train: 9 spikes at recovery 1.0, then 3, 2 and 2. The registered guard (d) fired at every dose, and the series is not monotone (-0.07, -3.28, +0.89, +2.39) because at recovery 0.5 the second spike arrives early and low before accommodation dominates.
- Scored over the same ten elements the best arm is the CONTROL at 69.1 percent, below plain B3's 71.8: this coupled geometry starts with a worse rise (609 against 570 V/s) and accommodation only costs count on top. Dosing the site does not buy accuracy, even though it buys the mechanism.
- What is left for the climb is therefore a process at the site slower than the 13.5 ms interspike interval, so that the threshold accumulates without gating the next spike; the recovery-factor family tested here acts on the same timescale as the interval and cannot do both. That is a narrower and harder target than the campaign assumed, and it is the honest reason to stop.
