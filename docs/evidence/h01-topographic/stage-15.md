# Stage 15: the human-for-rodent sodium swap on the human cell

Arm A: B3's somatic rodent NaTs (gbar 2.641 S/cm2) zeroed and the Toronto human NaTg inserted on the
soma with HL23PYR's human voltage shifts (vshiftm 13, vshifth 15, slopem 7), dosed at 310 pA.
Everything else held: the human anatomy of Allen 541563728, its human-fitted densities, the recorded
command, nseg 9, CVode 1e-10, 34 C. All fast quantities through the stage-12 chain.

Verdict: **MIXED**

## The characteristic curve (spike-1 rise against the inserted human sodium density)

| NaTg soma (S/cm2) | rise through chain (V/s) | count at 310 pA |
|---|---|---|
| 0.068 | silent (no spike) | 0 |
| 0.136 | silent (no spike) | 0 |
| 0.272 | silent (no spike) | 0 |
| 0.544 | silent (no spike) | 0 |
| 1.320 | silent (no spike) | 0 |
| 2.641 | 625 | 6 |

Recorded 348 V/s; B3 with the rodent equations 570 V/s at gbar 2.641, count 10.
Crossing of the recorded rise: {'dose_s_cm2': None, 'inside_series': False}.

## Percent accuracy over the ten registered elements at 310 pA

| element | recorded | B3 (rodent eq.) | best human eq. | B3 % | human % |
|---|---|---|---|---|---|
| rise_v_s | 347.66 | 570.42 | 625.14 | 36 | 20 |
| fall_v_s | -103.91 | -94.16 | -129.06 | 91 | 76 |
| peak_mv | 35.34 | 38.17 | 38.43 | 97 | 97 |
| threshold_mv | -56.41 | -57.29 | -39.10 | 99 | 83 |
| take_off_mv | -56.41 | -57.29 | -39.10 | 99 | 83 |
| count | 10.00 | 10.00 | 6.00 | 100 | 60 |
| rest_mv | -83.91 | -83.97 | -84.52 | 100 | 99 |
| climb_2_mv | 1.38 | -0.02 | 0.08 | 0 | 6 |
| climb_5_mv | 3.63 | 0.24 | 0.40 | 7 | 11 |
| rise_5_over_1 | 0.86 | 0.95 | 0.98 | 89 | 86 |
| **mean** | | | | **71.8** | **62.0** |

Best human-sodium dose: 2.641 S/cm2.

Voltage levels (peak, threshold, take-off, rest) are scored against a stated 100 mV span, so they
score high by construction and lift both means equally; the comparison between the two columns is
the reading, not the absolute figure.

## Arm B (the climb on this cell's anatomy): no reading

the published HL23PYR template cannot construct on the 541563728 morphology: biophys_HL23PYR raises 'section in the object was deleted' at distribute_channels, with the full reconstructed axon and again with the axon removed. Editing the published template would break the split's unchanged condition, so the arm is closed as a no-reading. Three of four evaluations spent; the fourth is unspent.

## Cells

- (a) PASS: the rise rises monotonically with the inserted human sodium density (the dose acts through the somatic sodium) -> {0.068: 'silent', 0.136: 'silent', 0.272: 'silent', 0.544: 'silent', 1.32: 'silent', 2.641: 625.1}. crossing of the recorded 348 V/s: {'dose_s_cm2': None, 'inside_series': False}
- (b) FAIL: some dose puts the rise within 15 percent of the recorded 348 V/s with the 310 pA count still within 5 to 15 -> {'in_rise_band': [], 'and_count_holds': []}. the rise must not be bought with the count
- (c) --: the human sodium equations reach the recorded rise where the rodent equations reach it only by removing nearly all somatic sodium (stage-11 intercept ~320 V/s) -> {'b3_rodent_rise_v_s': 570.4, 'best_human_dose': '2.641'}. reported beside the stage-11 rodent series 655/586/505
- (d) FAIL: percent accuracy over the ten registered elements improves against B3 -> {'b3_pct': 71.8, 'human_pct': 62.0}. lossy summary; the per-element table is the reading

## Reading

- The rise is NOT the sodium equations' lineage. Swapping the somatic rodent NaTs for the Toronto human NaTg with its human voltage shifts, on the human cell, with everything else held, does not bring the rise down: at B3's own somatic conductance the human equations rise at 625 V/s through the chain against B3's 570 and the recorded 348. The registered rejection (d) fired; the residual belongs elsewhere and is reported, not repaired.
- Below B3's own conductance the human sodium does not fire this cell at all: 0.068 to 1.320 S/cm2 are silent at 310 pA, sitting at a subthreshold plateau near -49 mV, which is ABOVE B3's own -57 mV take-off, while an eight-fold density change moves that plateau by 0.8 mV. The human NaTg's +13 mV activation shift is fitted to work against the Toronto potassium set; against Allen's rodent-lineage potassium, fitted to a sodium that activates 13 mV lower, it never activates. Kinetics are fitted as a set, not as interchangeable parts.
- Percent accuracy over the ten registered elements: B3 with the rodent equations 71.8, the best human-sodium arm 62.0. The swap costs 9.8 points, mostly on the threshold (-39.1 against the recorded -56.4), the count (6 against 10) and the fall (-129 against -104).
- The fully human-fitted model is not more accurate on this human cell either: HL23PYR scores 72.3 against B3's 71.8, a difference of half a point. They fail in complementary ways - HL23PYR wins the first-interval threshold step (99.7 against 0.0) and loses the excitability (count 20 against 10, 0.0 against 100.0). So 'human rather than mouse' is worth about half a point as posed; the kinetic lineage is not what separates either model from this recording.
- Where the accuracy actually lives: of the 28.2 points B3 is missing, the two threshold-climb elements carry 19.4 and the rise carries 6.4; every other element is at 90 to 100 percent and carries 2.4 between them. HL23PYR proves the climb is reachable with a human sodium while B3 holds the excitability, so the one remaining split worth running is what gives HL23PYR its climb, put into B3 without breaking the count.
