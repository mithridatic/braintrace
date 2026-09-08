# H01 E cell gain split: Stage 0 (g0-b3), 2026-09-07/08, evaluation 1 complete

Derived from `h01-e-gain/stage-g0-decision.json`. Spec: `docs/specs/2026-09-07-h01-e-gain-split.md` (with the
2026-09-08 per-input resume addendum); manifest `h01-e-gain-manifest.json`; candidate `g0-b3` sha256
`e4825c83bdc7980a469b5501b2a54c68e680f1cb52a2055d36bcf4fc7b70e564` (equal to B3). Image `braintrace-h01-neuron:9.0.2`. One evaluation spent of four.

## Outcome

Evaluation 1 is complete. Sweep 43 completed on 2026-09-07 (shared host); sweeps 50, 53 and 56, killed at the 1500 s abort
on that first launch, were completed on 2026-09-08 one container at a time on an unshared host, every run rc 0 inside the abort.
Counts 0/4/8/10 at 110/200/250/310 pA (human 0/1/5/10). The campaign-stop clause (a 250 or 310 count different from B3)
is not triggered. The registered rejection clause is met (sweep-43 late return +1.4 to +1.7 mV). The 200 pA count prediction
(1, exact band) missed at 4. **Decision: Stage G opens as registered**, with the sweep-43 late-return failure carried as the
condition its arms are judged on; this page does not launch Stage G, and each launch needs per-job approval.

## Wall clocks

| Input | Start (UTC) | End (UTC) | Wall s | Integration s | Basis | Host |
| --- | --- | --- | ---: | ---: | --- | --- |
| sweep43 | 2026-09-07 22:36:07 | 2026-09-07 22:49:47 | 820 | 814.9 | runner (first launch, host shared with another agent's containers) | shared (h01-i-reserve-*, h01-l4-reproduction-*, h01-i-transfer-* containers beside it) |
| sweep50 | 2026-09-08 01:38:40 | 2026-09-08 01:52:45 | 845 | 813.2 | container launch to the first 30 s docker-ps poll with the container gone; the runner process died at 01:42:08Z (empty stdout/stderr, no traceback) while the container ran on and wrote the report; row recorded through the runner's record_rows with a manual_record block | unshared: docker ps showed only synapse; no python process above 300 MB |
| sweep53 | 2026-09-08 01:53:17 | 2026-09-08 02:07:09 | 828 | 823.0 | runner (cmd wrapper, python -u); container 01:53:17Z to 02:07:09Z | unshared |
| sweep56 | 2026-09-08 02:08:25 | 2026-09-08 02:21:43 | 781 | 777.3 | runner (cmd wrapper, python -u); container 02:08:25Z to 02:21:43Z | unshared |

Measured cost of one evaluation (four serial runs): **3274 s wall (54.6 min)**, 3228.4 s integration.
Anchor for 250/310 pA was 649-665 s (unshared host, earlier); measured here 845/828 s (1.25-1.3x). Every run finished with at
least 655 s to spare under the 1500 s abort. The first launch's 1500 s kills on a shared host are recorded in `campaign-log.json`.

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

Summary: {'pass': 5, 'fail': 4, 'unavailable': 1} (re-scored 2026-09-08, unchanged). Onset rows held; late-return rows missed by
+1.40 to +1.72 mV. Registered rejection clause met: Stage G arms are judged on the *change* of these rows.

## Bands

| Band | Predicted | Observed | Held |
| --- | --- | --- | --- |
| sweep-43 onset rows (1019-1120 ms) within 1 mV | within 1 mV | 5 of 5 pass; residuals +0.43 to +0.91 mV | yes |
| sweep-43 late-return rows (1520-2120 ms) within 1 mV | within 1 mV | 4 of 4 available rows fail: +1.72 (1520), +1.71 (2019), +1.40 (2021), +1.69 (2040) mV; numerical limit 0.000 mV at every row (resolved fails); 2120 ms unavailable (trace stops at 2100 ms) | no |
| sweep 50 reproduces the retained B3 count 8 exactly (different count = campaign stops) | 8 | 8; cycle table identical to b3-all-inputs-usable.md to 0.01 ms; width and AHP rows identical | yes |
| sweep 53 reproduces the retained B3 count 10 exactly (different count = campaign stops) | 10 | 10; cycle table identical to b3-all-inputs-usable.md to 0.01 ms; width and AHP rows identical | yes |
| sweep-56 count 1 (repeat band 56/59/60/61/62, range 0, exact) | 1 | 4 | no |
| sweep-56 first-spike latency within repeat range x 1.47 (82.6 ms) | |model - human| <= 82.6 ms | model 127.4 ms vs human sweep-56 205.8 ms: residual -78.4 ms (3.6 ms inside the limit); repeat latencies 205.8/208.7/152.5/199.0/203.9 ms | yes |
| axon-first initiation at every active input | axon first | axon first at 50/53/56: axon leads by 0.213, 0.207, 0.219 ms; no shoulder | yes |

Five of seven bands held. The two misses: the sweep-43 late return (registered rejection, met) and the 200 pA count
(unregistered as a stop; carried into Stage G as a scored row).

## Gain (spikes per pA), all four inputs measured

| Segment | Human counts | Model counts | Human | Model | Ratio | Model source |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 200-250 | 1 -> 5 | 4 -> 8 | 0.0800 | 0.0800 | 1.00 | this evaluation: g0-b3-sweep56 and g0-b3-sweep50 |
| 250-310 | 5 -> 10 | 8 -> 10 | 0.0833 | 0.0333 | 0.40 | this evaluation: g0-b3-sweep50 and g0-b3-sweep53 |
| 310-350 | 10 -> 13 | 10 -> 12 | 0.0750 | 0.0500 | 0.67 | 310 from this evaluation (g0-b3-sweep53); 350 from the retained B3 trace h01-e-usable/b3-sk035-ca-decay-sweep55 (same flags, same sha256) |

Mean-full-cycle rate slope 250-310 pA: human 0.0869 Hz/pA, model 0.0386 Hz/pA (this evaluation).

Human counts 1/5/10/13 at 200/250/310/350 pA; model 4/8/10/12. The model matches the human slope between 200 and 250 pA (0.080 vs 0.080 spikes/pA) but sits three spikes above the donor at both inputs, then flattens between 250 and 310 pA (0.033 vs 0.083) to meet the pinned 310 pA point, and stays flat above it (0.050 vs 0.075). The error is an excess of firing at low drive (200-250 pA) that the 310 pA pin absorbs, not a uniformly compressed slope; the first spike at 200 pA comes 78 ms early and is followed by three spikes the donor never fires.

## 200 pA (sweep 56) against the repeat band (human sweeps 56/59/60/61/62)

Human counts [1, 1, 1, 1, 1] (range 0: the count row is exact); first-spike latency [205.8, 208.7, 152.5, 199.0, 203.9] ms,
limit range x 1.47 = 82.6 ms. Model: count **4** (fail, both tiers); first spike 127.4 ms vs sweep-56 205.8 ms,
residual -78.4 ms (pass, 3.6 ms inside the limit).

| Cycle | Human cycle_ms | Model cycle_ms | Model ahp_mv | Model width_ms |
| --- | ---: | ---: | ---: | ---: |
| 1 | 207.76 | 130.59 | -70.72 | 0.988 |
| 2 | - | 69.10 | -70.90 | 0.986 |
| 3 | - | 259.82 | -71.04 | 0.985 |
| 4 | - | 295.65 | -71.11 | 0.984 |

## 250 / 310 pA same-image repeat

Counts 8 and 10, cycle tables identical to `h01-e-usable/b3-all-inputs-usable.md` to 0.01 ms; the width_ms and ahp_mv verdict
rows diff clean against that page (no cycle newly failing). Rate: 250 pA 7.73 vs 4.79 Hz (fail, limit 0.72 Hz); 310 pA 10.05 vs
10.00 Hz (pass). Adaptation passes at both. Axon-first at 50/53/56 (axon leads 0.213/0.207/0.219 ms).

## Stage G judging conditions (recorded, not launched)

- sweep-43 late-return rows: judged on the change relative to g0-b3's +1.72/+1.71/+1.40/+1.69 mV (1520/2019/2021/2040 ms); an arm moving them by more than 1 mV is rejected as registered
- sweep-43 onset rows must stay within 1 mV (g0-b3: +0.43 to +0.91 mV)
- 200 pA count: g0-b3 gives 4 against the exact band 1; G1 registered 1, G2 registered 0-1
- 250/310 counts and rates: registered predictions unchanged (G1: 250 count 5-7, 310 count 9-10 and rate within 1.5 Hz of 10.0; G2: 250 <= 6, 310 >= 8, not a pure shift)
- axon-first initiation at 50/53/56 (g0-b3 leads 0.21-0.22 ms)
- no width cycle newly failing relative to b3-all-inputs-usable.md

## Files

- `h01-e-gain/stage-g0-decision.json` (this page's source; `decision` = `stage-g-opens-as-registered`)
- `h01-e-gain/g0-b3-sweep{43,50,53,56}.json` (+ `-raw.json`; `.npz` traces are gitignored data), `g0-b3-sweep{50,53,56}-initiation.json`
- `h01-e-gain/g0-b3-sweep43-contract.{json,md}`, `g0-b3-contract-records.json`
- `h01-e-gain/g0-b3-usable.{json,md}` (110/200/250/310 measured; 350 column is the retained B3 sweep-55 trace, labelled)
- `h01-e-gain/campaign-log.json` (one evaluation entry; sweep 50 row carries a `manual_record` block), `stage0-resume-sweep{50,53,56}.log*`
