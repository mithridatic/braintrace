# H01 E cell gain split: Stage G (g1-ih-half, g2-leak-150), 2026-09-08, evaluations 2 and 3 of 4

Derived from `h01-e-gain/stage-g-decision.json`. Spec: `docs/specs/2026-09-07-h01-e-gain-split.md`; manifest
`h01-e-gain-manifest.json`; gate `h01-e-gain/stage-g0-decision.json` (`stage-g-opens-as-registered`); base `g0-b3`
sha256 `e4825c83...` (B3). Image `braintrace-h01-neuron:9.0.2`. Candidates: `g1-ih-half` sha256 `b407cb3d...`
(`ih_density_factor 37.5` from 75), `g2-leak-150` sha256 `555a5e5c...` (`leak_factor 1.5`). Three evaluations spent of four.

## Outcome

**Both arms are rejected on registered clauses.** G1 leaves the 250 pA count at 8 (its clause: "250 count still >= 8");
it moved the rest -1.7 mV and the 310 pA rate -0.23 Hz and nothing else, so halving the distributed Ih translates the
resting potential without rotating the f-I curve. G2 moves the sweep-43 late return by -3.1 to -3.8 mV (clause: "moving
the sweep-43 return by more than 1 mV rejects the arm") and takes the 310 pA count to 6 (< 8); its pure-shift clause is
*not* met (drops -8 at 250 pA, -4 at 310 pA), so the leak rotates the curve, but by silencing 250 pA (soma plateau
-66.8 mV) rather than landing it near 5. The registered dose rule names G2 (G1's ratio is 0) and a linear dose of
`leak_factor 1.1875`, whose interpolated 310 rate change (-1.51 Hz) and sweep-43 move (-1.15 to -1.42 mV) are both outside
their constraints: no admissible dose exists. **G3 selection recorded: the coordinator's registered Dissection-rule
alternative, combined Ih half + leak x1.5 on B3 (Hartshorne p202/p204). G3 is NOT run.** The additive reading of the two
arms predicts the combined arm fails every band; under the manifest fail rule the campaign stands at FAIL with 3 of 4
spent unless the coordinator elects to spend evaluation 4.

## Wall clocks

| Arm | Input | Start (UTC) | End (UTC) | Wall s | Integration s | Host |
| --- | --- | --- | --- | ---: | ---: | --- |
| g1-ih-half | sweep43 | ~03:19:42 | 03:27:02 | 440 | 435.8 | previous agent; host state not recorded by this agent |
| g1-ih-half | sweep50 | ~03:27:57 | 03:40:49 | 772 | 767.8 | as above |
| g1-ih-half | sweep53 | ~03:41:43 | 03:55:13 | 810 | 804.9 | as above |
| g1-ih-half | sweep56 | - | - | - | - | not run: arm rejected at 250 pA (log status written by hand) |
| g2-leak-150 | sweep43 | ~03:56:14 | 04:03:41 | 447 | 443.9 | previous agent; host state not recorded |
| g2-leak-150 | sweep50 | 04:07:35 | 04:15:46 | 491 | 487.9 | **shared**: two SP8 BrainCell build-only processes (`examples.h01_verified_network --build --control ei`) and four P_NP pytest workers; 0 spikes, hence short |
| g2-leak-150 | sweep53 | 04:16:42 | 04:33:11 | 989 | 981.1 | **shared** at launch with the same SP8 jobs; 6 spikes; 1.2x the 828 s g0-b3 10-spike run on an unshared host |
| g2-leak-150 | sweep56 | 04:34:10 | 04:45:29 | 679 | 673.6 | unshared at launch (no SP8/P_NP python process; docker: synapse only); 0 spikes |

Start times marked ~ are file-time end minus the runner's wall seconds. Every run rc 0 inside the 1500 s abort. Evaluation 2
(three runs) 2022 s wall; evaluation 3 (four runs) 2606 s wall (43.4 min). Each run was launched one container at a time
through a cmd-wrapped `python -u` runner (`stageg-<arm>-<input>.log.out/.err`), dry-run first (sweep 43 `skip`, the
target input `resume evaluation k`).

## Sweep 43 (110 pA) rows: change relative to g0-b3

| Row | Human mV | g0-b3 residual | g1 residual | g1 change | g2 residual | g2 change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sub_1019_mv (rest) | -84.500 | +0.531 | -1.201 | -1.732 | -0.465 | -0.995 |
| sub_1021_mv | -83.781 | +0.908 | -0.823 | -1.731 | -0.102 | -1.011 |
| sub_1025_mv | -82.094 | +0.799 | -0.924 | -1.724 | -0.303 | -1.103 |
| sub_1040_mv | -78.531 | +0.618 | -1.029 | -1.647 | -1.055 | -1.673 |
| sub_1120_mv (plateau) | -74.188 | +0.426 | -0.561 | -0.987 | -3.278 | -3.704 |
| sub_1520_mv | -75.688 | +1.716 | +1.041 | -0.675 | -2.061 | -3.776 |
| sub_2019_mv | -75.688 | +1.714 | +1.039 | -0.675 | -2.061 | -3.775 |
| sub_2021_mv | -76.469 | +1.396 | +0.722 | -0.675 | -2.362 | -3.758 |
| sub_2040_mv | -81.812 | +1.686 | +0.970 | -0.715 | -1.378 | -3.064 |
| sub_2120_mv | -85.750 | unavailable | unavailable | - | unavailable | - |

Numerical limits at every row are below 1e-6 mV (resolved). Contract tallies: g0-b3 5 pass / 4 fail; g1 5 pass / 4 fail
(1019, 1040, 1520, 2019); g2 3 pass / 6 fail. G1's late return moved -0.68 to -0.72 mV (allowed) and now sits +0.72 to
+1.04 mV; G1's onset rows 1019/1040 slipped outside 1 mV by 0.20/0.03 mV because the whole rest fell 1.7 mV. G2's late
return crossed the donor and sits 1.4-2.4 mV below it; its pulse plateau is 3.7 mV lower (leak x1.5 cuts the input
resistance).

## Bands per arm

### g1-ih-half (registered predictions)

| Band | Predicted | Observed | Held |
| --- | --- | --- | --- |
| 250 pA count 5-7, cycles 4-8 >= 200 ms | 5-7 | **8**; cycles 4-8 190.4, 174.1, 167.4, 162.7, 159.5 ms (g0-b3 184 -> 157); rate 7.57 Hz (g0-b3 7.73, human 4.79) | no |
| 310 pA count 9-10, rate within 1.5 Hz of 10.0 | 9-10 | 10; 9.82 Hz (-0.23 Hz from g0-b3) | yes |
| pre-pulse baseline -1 to -3 mV from B3 | -1..-3 mV | -1.73 mV (-83.97 -> -85.70) | yes |
| sweep-43 late return moves <= 1 mV | <= 1 mV | -0.68 to -0.72 mV | yes |
| sweep-43 onset rows within 1 mV | 5 of 5 | 3 of 5; 1019 -1.20, 1040 -1.03 mV | no |
| 200 pA count 1 | 1 | not run (arm rejected at 250 pA) | untested |
| axon-first initiation | axon first | 50: leads 0.213 ms; 53: 0.207 ms; no shoulder | yes |
| no width cycle newly failing | same set | 310 pA cycle 2 only, as B3 | yes |

Rejection clause "310 count < 9, or 250 count still >= 8": **met** (250 count 8). Five of seven scored bands held; the
one that mattered did not.

### g2-leak-150 (registered predictions)

| Band | Predicted | Observed | Held |
| --- | --- | --- | --- |
| 250 pA count <= 6 (both counts fall, 250 more) | <= 6 | **0** (drop 8); plateau -66.8 mV; rate 0 (human 4.79) | yes |
| 310 pA count >= 8 | >= 8 | **6** (drop 4); rate 6.02 Hz (-4.03 Hz from g0-b3; human 10.00); cycles 75.8, 40.2, 137.8, 229.8, 215.3, 208.3 ms; adaptation 5.19 vs 9.95 fail | no |
| 310 pA rate within 1.5 Hz (dose-rule constraint) | <= 1.5 Hz | -3.99 Hz | no |
| 200 pA count 0-1 | 0-1 | 0 (human 1, exact band: contract fail; registered band held by silence; plateau -70.9 mV) | yes |
| sweep-43 late return moves <= 1 mV | <= 1 mV | -3.06 to -3.78 mV | no |
| sweep-43 onset rows within 1 mV | 5 of 5 | 3 of 5; 1040 -1.06, 1120 -3.28 mV | no |
| axon-first initiation | axon first | 310 pA: leads 0.217 ms, no shoulder; 250 and 200 pA untestable (no spike) | yes (where testable) |
| no width cycle newly failing | same set | 310 pA cycle 2 only, as B3 | yes |

Rejection clause "250 and 310 fall by the same count (pure shift)": **not met** (drops -8 and -4). Sweep-43 clause "return
moved by more than 1 mV": **met**. Four of eight held.

## Gain tables (spikes per pA; human 1/5/10/13 at 200/250/310/350 pA)

| Arm | 200 -> 250 | 250 -> 310 | 310 -> 350 | Rate slope 250-310 (Hz/pA; human 0.0869) |
| --- | --- | --- | --- | ---: |
| g0-b3 (B3) | 4 -> 8: 0.080 (ratio 1.00) | 8 -> 10: 0.033 (0.40) | 10 -> 12: 0.050 (0.67; 350 from the retained B3 sweep 55) | 0.0386 |
| g1-ih-half | not run at 200 pA | 8 -> 10: 0.033 (0.40) | not run at 350 | 0.0375 |
| g2-leak-150 | 0 -> 0: 0.000 (0.00) | 0 -> 6: 0.100 (1.20) | not run at 350 | 0.1003 (from 0 Hz) |

G1 reproduces B3's curve to the spike: 8/10 with the 310 pA rate 0.23 Hz lower. G2 rotates it past the human slope between
250 and 310 pA (0.100 vs 0.083) but from a silent 250 pA point: the counts read 0/0/6 against 1/5/10.

## G3 selection (recorded before any G3 run; G3 is not run)

- Registered rule: the arm with the larger |d(250 count)| / |d(310 rate)| per unit factor; dose interpolated linearly to
  land 250 count 5 with the 310 rate change within 1.5 Hz. G1: 0 / 0.23 Hz = 0. G2: 8 / 4.03 Hz = 1.99 (per 0.5 factor).
  Better arm G2.
- Linear dose `leak_factor 1.1875`: predicted 310 rate 8.53 Hz (change -1.51 Hz, outside 1.5), sweep-43 late-return move
  -1.15 to -1.42 mV (outside 1 mV); and the 8 -> 0 collapse at 250 pA (plateau 10 mV under threshold) says the count is not
  linear in the leak, so the interpolation is not a measured estimate. **No admissible dose under the rule.**
- Selection therefore falls to the coordinator's registered Dissection-rule alternative: **combined `ih_density_factor
  37.5` + `leak_factor 1.5` on B3** (Hartshorne p202/p204). Predicted if run, from the two arms added: rest about -2.7 mV,
  250 pA plateau below -67 mV (count 0 at 200 and 250 pA), 310 pA count <= 6, sweep-43 return about -4.5 mV from g0-b3:
  every registered band fails. Not run; whether to spend evaluation 4 on it is the coordinator's call. If it is not run,
  the manifest fail rule applies (FAIL at 3 of 4) and the reassessment names a gain set outside the fit's passive family
  (human slow Na inactivation or Kv7/M kinetics absent from the Allen genome, or a second human L2/3 donor with a recorded
  f-I curve, SP6b lead).

## Files

- `h01-e-gain/stage-g-decision.json` (this page's source; per-arm bands, rejection, verdicts per input, wall clocks, hashes, G3 selection)
- `h01-e-gain/g1-ih-half-sweep{43,50,53}.json` (+ `-raw.json`; `.npz` gitignored), `g1-ih-half-sweep{50,53}-initiation.json`,
  `g1-ih-half-sweep43-contract.{json,md}`, `g1-ih-half-usable.{json,md}` (110/250/310; 200 not run)
- `h01-e-gain/g2-leak-150-sweep{43,50,53,56}.json` (+ `-raw.json`), `g2-leak-150-sweep{50,53,56}-initiation.json`,
  `g2-leak-150-sweep43-contract.{json,md}`, `g2-leak-150-usable.{json,md}` (110/200/250/310, retained per-cycle tables)
- `h01-e-gain/campaign-log.json` (evaluations 1-3; g1 sweep56 carries the hand-written status and a `manual_record` block),
  `stageg-*.log.out/.err`
