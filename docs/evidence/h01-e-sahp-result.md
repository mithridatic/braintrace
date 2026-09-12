# SP12 result: a spike-triggered outward gate lands the 200 pA count; its 1 s tail is too short

Spec: [2026-09-11-h01-e-spike-triggered-outward](../specs/2026-09-11-h01-e-spike-triggered-outward.md).
Manifest: [h01-e-sahp-manifest.json](h01-e-sahp-manifest.json). Executor: Vast box,
NEURON 9.0.2, local executor, `kv3-closing-source` recompiled with
[KsAHP.mod](h01-l2-mechanisms/KsAHP.mod). Three evaluations spent of four; stage 1 not
opened. [Stage 0 decision](h01-e-sahp/stage-0-reanalysis.json) (`campaign_close` block),
[dose registration](h01-e-sahp/dose-registration.json).

## Stage 0 at 200 pA (FAIL under the registered bands)

| Arm | Count | Mid p50 (1300-1500) | Late p50 (1800-2000) | Bands |
| --- | ---: | ---: | ---: | --- |
| human | 1 | -67.47 | -67.06 | |
| unchanged B3 | 4 | -64.06 | -65.09 | |
| dose A, 3.50e-4 S/cm2 | 1 | -67.93 | -65.80 | 5 of 6 (late +1.26 mV) |
| dose B, 1.03e-3 S/cm2 | 1 | -73.48 | -70.31 | 4 of 6 |
| Nap soma density 0 | 0 | -63.26 | -63.05 | 0 of 2 (its own prediction) |

Both doses keep B3's first spike (1147.35 ms) and its pre-spike trace to 0.000 mV, so
the recompiled library reproduces B3 and the gate is silent before a spike. Runs took
635 and 656 s.

## Reading

Dose A follows the human within 0.5 mV from 1240 to 1700 ms and then drifts upward by
1.4 mV as the gate decays with its 1 s tail; the human's shift does not decay within the
pulse. The residual is the tail length. Dose B overshoots: the 0.059 nA target was sized
against a window in which B3 spikes.

The comparison arm broke its prediction the other way: removing Nap removes every spike
at 200 pA. Nap is a rheobase lever whose intervention effect is far larger than its
0.012 nA baseline current suggested. Its signature differs from the human's (one spike,
then a lower level).

## Both tiers

Contract: count 1 equals the human for both doses; the first-spike time (1147 vs
1225 ms) fails the 1 ms row and is outside this family. Usable tier:
[dose A](h01-e-sahp/s0-ksahp-a-usable.md), [dose B](h01-e-sahp/s0-ksahp-b-usable.md),
[Nap zero](h01-e-sahp/s0-nap-zero-usable.md). B3 stays frozen and unpromoted.

## Next

[SP13](../specs/2026-09-12-h01-e-sahp-tail.md): the same gate with a 5 s tail at
3.12e-4 S/cm2, one evaluation at sweep 56, then the registered stage 1 at 250, 310 and
110 pA. Causal model Y4 updated in the same commit.
