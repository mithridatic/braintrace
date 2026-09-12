# SP13 result: the 5 s spike-triggered gate lands 200 pA and over-brakes 250 and 310 pA

Spec: [2026-09-12-h01-e-sahp-tail](../specs/2026-09-12-h01-e-sahp-tail.md). Manifest:
[h01-e-sahp-tail-manifest.json](h01-e-sahp-tail-manifest.json). Executor: Vast box,
NEURON 9.0.2, local executor, library at the SP12 recompile. Two evaluations spent of
two. [Stage 0](h01-e-sahp-tail/stage-0-decision.json) PASS 6 of 6;
[stage 1](h01-e-sahp-tail/stage-1-decision.json) FAIL 11 of 16; closed.

## Stage 0 at 200 pA (PASS)

| | Count | Mid p50 (1300-1500) | Late p50 (1800-2000) |
| --- | ---: | ---: | ---: |
| human | 1 | -67.47 | -67.06 |
| unchanged B3 | 4 | -64.06 | -65.09 |
| KsAHP 3.12e-4 S/cm2, tau_off 5 s | 1 | -67.94 | -67.32 |

The derived prediction (-67.43 mid, -67.06 late) landed within 0.5 mV. Every 100 ms
median from 1240 to 1940 ms is within 1 mV of the human's. The pre-spike trace and the
first spike are B3's to 1e-6 mV and 1e-8 ms. Run 632 s.

## Stage 1 (FAIL)

| Input | Human | B3 | KsAHP 5 s | Band |
| --- | ---: | ---: | ---: | --- |
| 110 pA onset rows vs B3 | | | 0.000 mV | held (5 of 5) |
| 110 pA onset rows vs human | | | 0.43-0.91 mV | held (5 of 5) |
| 250 pA count | 5 | 8 | 3 | 5-7: failed |
| 310 pA count | 10 | 10 | 7 | 9-10: failed |
| 310 pA late cycle | 120 ms | 121 ms | 184 ms | 120 +/- 12.8: failed |

At 250 pA the model fires a doublet (1075, 1121 ms) and then, with the gate at 0.83,
one more spike at 1563 ms. At 310 pA the gate reaches 0.96 and stretches the cycles to
about 180 ms.

## Reading

The gate is silent without a spike and sufficient for the 200 pA response, but it
accumulates along a train, and the human's 310 pA cycles equal unchanged B3's. So a
single non-saturating brake cannot serve all three inputs. Two readings remain, not
separated by this test: the human's brake saturates after the first spike and B3 lands
the 310 pA rate only because a second error compensates (the human's steeper gain, the
cross-cell pattern IE11); or the 200 pA shift is not a brake that persists into a train.
The 250 pA doublet followed by 442 ms of silence points to the first: right brake,
missing drive.

## Both tiers

Contract: 200 pA count passes; 250 and 310 pA counts fail; 110 pA rows pass; the
first-spike time at 200 pA (1147 vs 1225 ms) still fails. Usable tier:
[200 pA](h01-e-sahp-tail/s0-ksahp-tail5-usable.md),
[250 and 310 pA](h01-e-sahp-tail/s1-ksahp-tail5-usable.md),
[110 pA](h01-e-sahp-tail/s1-ksahp-tail5-sweep43-usable.md). B3 stays frozen and
unpromoted.

## Next

Draft [SP14](../specs/2026-09-12-h01-e-brake-plus-gain.md): a saturating gate (one
spike opens it fully) paired with one gain lever, registered against 200, 250 and
310 pA together. Needs approval. Causal model Y4 updated in the same commit.
