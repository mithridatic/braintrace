# SP15 stage 0 result: the families and the branch, from retained traces

Spec: [2026-09-12-h01-topographic-strategy](../specs/2026-09-12-h01-topographic-strategy.md).
Script: `h01_topographic_stage0.py` (tests alongside). No simulation. Outputs in
[h01-topographic/](h01-topographic/): [decision](h01-topographic/stage-0-decision.json),
[tables](h01-topographic/stage-0-decision.md), phase planes and load curves for both cells.
Search tree: [h01-search-tree.svg](h01-reasoning/h01-search-tree.svg).

## Decision limits (temporal family, from the human's own repeats)

| cell | repeats | level p50 | threshold | max rise | first spike |
| --- | --- | ---: | ---: | ---: | ---: |
| E | 5 sweeps at 200 pA, one spike each | 0.30 mV | 0.25 mV | 1.5 V/s | 23 ms |
| I | 4 sweeps, one spike each | 0.17 mV | 0.38 mV | 5.3 V/s | 8 ms |

## Contrasts (model minus human, in units of the repeat sigma)

| cell | family | row | human | model | sigma |
| --- | --- | --- | ---: | ---: | ---: |
| E | elemental | max rise of spike 1, every cycle and input | 332 V/s | 598 V/s | 177 |
| E | elemental | post-spike level p50, 200 pA | -67.06 mV | -65.00 mV | 6.9 |
| E | elemental | first spike, 200 pA | 205 ms | 127 ms | 3.3 (marginal) |
| E | cyclical | threshold climb, 310 pA train | +2.1 mV | +0.1 mV | 8.1 |
| E | cyclical | cycle 2, 310 pA | 12.1 ms | 13.5 ms | 0.1 of the 12.8 ms limit |
| I | elemental | post-spike level p50, 0.19 nA | -66.5 mV | -71.2 mV | 28 |
| I | elemental | max rise of spike 1 | 597 V/s | 592 V/s | 1 |
| I | cyclical | threshold climb, 0.27 nA train | +4.8 mV | +0.9 mV | 10 |
| I | cyclical | cycle 2, 0.27 nA (the "early burst") | 6.3 ms | 16.4 ms | 0.8 of the 12.8 ms limit |

Onset capacitance by the same estimator: E human 128 pF, model 125 pF; I human 70 pF,
model 57 pF. The E upstroke contrast is therefore not near-soma charging.

## Load curves (injected current minus capacitive current, against voltage)

After the first spike at 200 pA the E human absorbs the whole 0.196 nA between -70 and
-65 mV and holds there; B3 absorbs 0.09 nA there and reaches 0.19 nA only at -62 mV. After
the first spike at 0.19 and 0.27 nA the I human absorbs 0 to -0.1 nA between -78 and
-65 mV (a net inward current carries it to the next spike in 6 to 8 ms); the finalist
absorbs 0.16 to 0.25 nA. After the last spike the human and model curves overlap in both
cells. The difference is in time after a spike, not in voltage.

## Reading

- Temporal family eliminated; elemental and cyclical carry the contrast; the structural
  pattern (both humans against both fits) stands.
- Branch: function, use-dependent, for the threshold climb (both cells), the falling rise
  along the train (I), and the early post-spike load difference (both cells, opposite sign
  early, converging late). SP13's intervention result agrees: a spike-triggered gate
  reproduced the E 200 pA response and over-braked the trains.
- The E upstroke (177 sigma, every cycle) is a separate elemental contrast not yet split
  between somatic sodium (function) and cable load (inputs).
- The I early burst, the target of two campaigns, is inside the repeat limit and is not a
  steep X. B3's early first spike is marginal.
- No lever is named.

## Next split (registered before it runs)

E upstroke, Q2: hold the function, change the input (the B3 genome on the H01 morphology,
one run). Prediction if function: the 600 V/s upstroke persists; if inputs: it falls toward
332 V/s. Causal model Y3 and Y4 updated in the same commit.
