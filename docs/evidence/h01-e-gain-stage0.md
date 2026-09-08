# H01 E cell gain split: Stage 0 (g0-b3), 2026-09-07/08

Derived from `h01-e-gain/stage-g0-decision.json`. Spec: `docs/specs/2026-09-07-h01-e-gain-split.md`;
manifest `h01-e-gain-manifest.json`; candidate `g0-b3` sha256 `e4825c83bdc7980a469b5501b2a54c68e680f1cb52a2055d36bcf4fc7b70e564` (equal to B3). Image
`braintrace-h01-neuron:9.0.2`. One evaluation spent of four.

## Outcome

Stage 0 is incomplete. Sweep 43 completed; sweeps 50, 53 and 56 were killed by the runner at the 1500 s abort
and are untested. The host was shared with another agent's containers for the whole window. Stage G does not
open: `decision` is left null in the JSON so the runner gate stays closed until a second approval records one.

## Wall clocks

| Input | Start (UTC) | End (UTC) | Runner seconds | Integration seconds | Status |
| --- | --- | --- | ---: | ---: | --- |
| sweep43 | 22:36:07 | 22:49:47 | 820 | 814.9 | completed |
| sweep50 | 22:49:47 | 23:14:50 | 1503 | - | killed at the 1500 s abort: untested |
| sweep53 | 23:14:50 | 23:39:51 | 1501 | - | killed at the 1500 s abort: untested |
| sweep56 | 23:39:51 | 00:04:52 | 1501 | - | killed at the 1500 s abort: untested |

Anchor for 250/310 pA: 649-665 s on an unshared host; exceeded 2.3x here. Sweep 43 (subthreshold) took 820 s.
Lower bound for one Stage G evaluation under these conditions: 820 + 3 x 1500 = 5320 s; no completed duration exists for 50/53/56.

## Sweep 43 (110 pA) contract rows, 1 mV allowance

| Row | Human mV | Model mV | Residual mV | Numerical limit | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| sub_1019_mv | -84.500 | -83.969 | +0.531 | +0.000 | pass |
| sub_1021_mv | -83.781 | -82.873 | +0.908 | +0.000 | pass |
| sub_1025_mv | -82.094 | -81.295 | +0.799 | +0.000 | pass |
| sub_1040_mv | -78.531 | -77.913 | +0.618 | +0.000 | pass |
| sub_1120_mv | -74.188 | -73.761 | +0.426 | +0.000 | pass |
| sub_1520_mv | -75.688 | -73.972 | +1.716 | +0.000 | fail |
| sub_2019_mv | -75.688 | -73.973 | +1.714 | +0.000 | fail |
| sub_2021_mv | -76.469 | -75.072 | +1.396 | +0.000 | fail |
| sub_2040_mv | -81.812 | -80.127 | +1.686 | +0.000 | fail |
| sub_2120_mv | -85.750 | - | - | +0.000 | unavailable |

Summary: {'pass': 5, 'fail': 4, 'unavailable': 1}. Onset rows (1019-1120 ms) held; late-return rows (1520-2040 ms) missed by +1.40 to +1.72 mV.
Registered rejection clause met: B3's Stage B levers (leak reversal -4 mV, distributed Ih 75) moved F5's late return.

## Bands

| Band | Predicted | Observed | Held |
| --- | --- | --- | --- |
| sweep-43 onset rows (1019-1120 ms) within 1 mV | within 1 mV | 5 of 5 pass; residuals +0.43 to +0.91 mV | yes |
| sweep-43 late-return rows (1520-2120 ms) within 1 mV | within 1 mV | 4 of 4 available rows fail: +1.72 (1520), +1.71 (2019), +1.40 (2021), +1.69 (2040) mV; numerical limit 0.000 mV at every row (resolved fails); 2120 ms unavailable (trace stops at 2100 ms) | no |
| sweep-56 count 1 (repeat band 56/59/60/61/62, range 0, exact) | 1 | untested: run killed at 1500 s | untested |
| sweep-56 first-spike latency within repeat range x 1.47 | within 82.6 ms of the repeat mean | untested: run killed at 1500 s | untested |
| sweep 50 reproduces 8 spikes within solver noise | 8 | untested: run killed at 1500 s | untested |
| sweep 53 reproduces 10 spikes within solver noise | 10 | untested: run killed at 1500 s | untested |
| axon-first initiation at every active input | axon first | no active input completed; sweep 43 is subthreshold (0 spikes, h01_initiation_score: no complete spike inside the pulse) | untested |

## Gain (spikes per pA)

| Segment | Human | Model | Ratio | Model source |
| --- | ---: | ---: | ---: | --- |
| 200-250 | 0.08 | - | - | untested: sweep 56 killed, 250 from the retained B3 trace |
| 250-310 | 0.0833 | 0.0333 | 0.4 | retained B3 traces (h01-e-usable/b3-sk035-ca-decay-sweep50/53/55), same flags and sha256, not this evaluation |
| 310-350 | 0.075 | 0.05 | 0.67 | retained B3 traces (h01-e-usable/b3-sk035-ca-decay-sweep50/53/55), same flags and sha256, not this evaluation |

Mean-full-cycle rate slope 250-310 pA: human 0.0869 Hz/pA, model (retained B3) 0.0386 Hz/pA.

## 200 pA repeat band (human sweeps 56/59/60/61/62)

Counts [1, 1, 1, 1, 1] (range 0: the count row is exact). First-spike latency [205.8, 208.7, 152.5, 199.0, 203.9] ms,
limit range x 1.47 = 82.6 ms. Model: untested (run killed).

## Files

- `h01-e-gain/stage-g0-decision.json` (this page's source)
- `h01-e-gain/g0-b3-sweep43.json`, `g0-b3-sweep43-contract.md`, `g0-b3-sweep43-initiation.json`
- `h01-e-gain/g0-b3-usable.md` (110 pA measured; 250/310/350 columns are the retained B3 traces, labelled)
- `h01-e-gain/campaign-log.json`
