# SP11 result: B3 current paths measured; the low-drive lever is not in the soma

Spec: [2026-09-11-h01-e-current-measurement](../specs/2026-09-11-h01-e-current-measurement.md).
Manifest: [h01-e-currents-manifest.json](h01-e-currents-manifest.json). Executor: Vast box,
NEURON 9.0.2, local executor. One evaluation spent of three; stage 1 registered, not spent.
[Stage 0 decision](h01-e-currents/stage-0-decision.json),
[current tables](h01-e-currents/stage-0-current-paths.md),
[close decision](h01-e-currents/stage-close-decision.json).

## Stage 0: unchanged B3 with soma currents (PASS, 12 of 12 bands)

Counts 4, 8, 10 at 200, 250, 310 pA; peak times and troughs identical to `g0-b3`;
axon-first initiation at every input. Runs took 885, 981 and 1015 s.

## What the currents show

| Boundary at the soma, interspike mean (nA, inward positive) | 200 pA | 250 pA late | 310 pA late |
| --- | ---: | ---: | ---: |
| applied | 0.196 | 0.246 | 0.306 |
| axial into the cable | -0.188 | -0.232 | -0.286 |
| leak | -0.0066 | -0.0065 | -0.0066 |
| Kv3_1 | -0.003 | -0.0034 | -0.004 |
| SK | -0.0017 | -0.007 | -0.0133 |
| NaTs | +0.0015 | +0.0016 | +0.0019 |
| Nap | +0.0015 | +0.0015 | +0.0016 |
| Im | -0.0001 | -0.0001 | -0.0002 |
| net drift | 0.00003 | 0.00006 | 0.0001 |

The soma is a pass-through: 96 percent of the input leaves axially; the soma ionic
terms sum to about -9 pA at every input; only SK changes with the input.

**Erratum (2026-09-11).** The table is the balance of the soma(0.5) segment, one of
nine soma segments (65.8 of 592 um2). Whole-soma currents are nine times larger: at
200 pA leak -0.058, Kv3 -0.017, SK -0.016, NaTs +0.010, Nap +0.012, Im -0.0002 nA,
summing to -0.069 nA (35 percent of the applied current). Of the segment's 0.188 nA
axial outflow, 0.061 nA enters the neighbouring soma segments and 0.128 nA the cable.
The "pass-through" sentence above is withdrawn; the drift and share figures stand.
Whole-soma Nap (+0.012 nA) reaches 59 percent of the 0.020 nA p50 offset, so the
stage-1 "not spent by construction" reasoning below is withdrawn for Nap; that
evaluation is spent as the SP12 comparison arm (`h01-e-sahp-manifest.json`). Details in
the `erratum_2026-09-11` block of the close decision.

## The human contrast

| 200 pA, mean voltage (mV) | human | model |
| --- | ---: | ---: |
| mid pulse 1300-1500 ms | -67.7 | -61.9 |
| late pulse 1800-2000 ms | -67.1 | -64.8 |
| post pulse 2100-2300 ms | -85.8 | -84.4 |

The human fires once at 206 ms and then sits at -67 mV; the model fires four times and
ramps between -70 and -57 mV. At 110 pA without a spike the same model matched the
human's onset rows within 0.4-0.9 mV. The offset therefore appears after a spike.
By p50 it is 2.0 mV, 0.020 nA at the model's 98 MOhm; by mean it is 2.3 mV late and
5.8 mV mid pulse.

## Stage 1 not spent (fail-fast)

Nap can supply at most 0.0017 nA and Im 0.0002 nA at the plateau; neither reaches the
offset. An Im dose that does (about 100x) acts more strongly at the 310 pA threshold
and removes that train, which is the registered rejection. The scorer's admissible
boundary is the axial current alone: the lever lives in the cable or in a mechanism
the fit lacks.

## Reading, both tiers

Contract: 200 and 250 pA counts fail (4 vs 1, 8 vs 5), 310 passes; unchanged from
`g0-b3`. Usable tier: identical to `h01-e-gain/g0-b3-usable.md` (same trace). B3 stays
frozen and unpromoted. Next: the draft
[SP12 spec](../specs/2026-09-11-h01-e-spike-triggered-outward.md) (spike-triggered slow
outward current; needs a new mechanism and approval). Causal model Y4 updated in the
same commit.
