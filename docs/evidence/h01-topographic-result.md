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

## Stage 1 (executed 2026-09-12): the upstroke's branch, no reading

The B3 genome on the H01 skeleton 955432427 (2460 um of cable, B3 soma held) beside the
donor anatomy at nseg 1 ([decision](h01-e-morphology/stage-1-decision.json)). Mesh
control held: 653 V/s at nseg 1 against 598 at nseg 9 (9 percent), count 4, first spike
127 ms, threshold -57.4 mV. The H01 anatomy did not spike at 200 pA: plateau -78 mV,
onset capacitance 785 pF against 125 pF, input resistance about 38 MOhm against 98. The
registered rejection applies: a count contrast of the input is recorded, the upstroke is
unread. The cable load moves the low-input response by more than any channel lever tested,
but the change was too large to keep Y observable. Next registrations: a graded cable
change on the donor anatomy (membrane area x1.5 and x2) and a check of the skeleton
conversion's membrane area against the H01 surface mesh. Two evaluations spent of two
(the first failed before construction: soma-current probes need nseg 3).

## Stage 2 (executed): the cable load is a steep rheobase lever, still no upstroke reading

Dendritic and apical membrane area (capacitance, leak and distributed Ih together) scaled
1.5, 2 and 3 times at 200 pA, fixed geometry
([decision](h01-e-cable-load/stage-2-decision.json)):

| dose | count | onset capacitance | input resistance | 200 pA plateau |
| --- | ---: | ---: | ---: | ---: |
| x1 | 4 | 125 pF | 95 MOhm | -64.9 mV |
| x1.5 | 0 | 153 pF | 75 MOhm | -68.6 mV |
| x2 | 0 | 177 pF | 60 MOhm | -71.3 mV |
| x3 | 0 | 214 pF | 45 MOhm | -74.1 mV |

The dose acts on the load monotonically and leaves rest unchanged, but B3 sits just above
its rheobase at 200 pA, so a 50 percent cable increase already silences it. The registered
no-reading rejection applies again.

## The human's own cable load (retained traces, no run)

Measured at 110 pA where neither cell spikes
([human-passive-load.json](h01-e-cable-load/human-passive-load.json)):

| | input resistance | onset capacitance | time constant |
| --- | ---: | ---: | ---: |
| human | 80.4 MOhm | 128 pF | 10.3 ms |
| B3 | 90.9 MOhm | 124 pF | 11.3 ms |

The human is 1.13 times B3's conductance and 1.03 times its capacitance. That is the
observable range of this input, so a cable-load explanation of any model-human difference
has to work at 1.13 times, not at the 1.5 times that already silences the cell.

## Stage 3 (executed): control invalid

The same doses at 310 pA on the coarse mesh. The x1 control fired 72 times and then fell
silent (plateau -74 mV), against the fine mesh's ten steady spikes at -64.6 mV; the x2 dose
entered depolarisation block at -28 mV. The spike-1 upstroke band held (9 percent), which
is exactly how a one-number control check passes a control that does not reproduce the
response ([decision](h01-e-cable-load/stage-3-decision.json)). The scorer now requires the
control to reproduce the reference spike count. Stage 4 repeats the series at the fine mesh
and stage 5, registered before stage 4 was read, sets the model's load to the human's
measured 1.13 times and asks whether the upstroke follows.

## Stage 5 (executed): the recorded cell's own load, and what it costs the model

The dendritic and apical area was set so that the model carries the recorded cell's
conductance ([decision](h01-e-cable-load/stage-5-decision.json)). The dose hit that target,
1.137 against the recorded 1.131, and overshot the capacitance, 1.11 against 1.03, because
scaling area moves both together while the recorded cell carries more conductance per unit
capacitance. Its load differs in kind and not only in size.

| at the recorded cell's conductance | 200 pA | 310 pA |
| --- | ---: | ---: |
| model count | 0 | 4, then block at -28 mV |
| model count at its own load | 4 | 10 |
| recorded count | 1 | 10 |

The registered rejection applies, so the upstroke is again unread. The observation that
matters is the cost: carrying the recorded load removes this fit's low-input response
entirely. The fit therefore has less inward drive per unit load than the recorded cell, and
its excess count at its own lighter load is not evidence that it is too excitable. Together
with the stage-4 slope this closes the input branch: the cable is neither sufficient for the
rise difference nor available as an explanation of the count.
