# Stage 1 decision: Isolation split of the human-to-model contrast

Method: Hartshorne 2020, Isolation strategy (p144, p145, p179) with search
dissection decision limits (p199, p200). Inputs branch = the recording and its
datum; function branch = the model. Limits: human repeat sets and model numerical
repeats, combined in quadrature; rows above five times the limit are searched,
rows below are parked as inside repeatability (p033). Full tables:
[I](h01-isolation-i.md), [E](h01-isolation-e.md); machine-readable
`h01-isolation-{i,e}.json`.

## Inputs branch

**I cell.** Stage 0 ([forensics](h01-pv-bias-forensics.md)) moved the 31.4 pA
bias from the function to the inputs branch and closed it: the published fit
absorbed the holding current, so the model input is the command current only. With
that datum the passive rows are inside repeatability at both hyperpolarising inputs
(plateau ratio 0.2 and 0.4, return 0.6 and 1.7) and the post-pulse return of the
spiking inputs is parked (0.2 to 0.7). The "passive family" named by the 2026-09-06
campaign was the double-counted bias, not a model deficit.

Human repeat sets: short square 750 pA (sweeps 15, 16, 17, 19) and long square
120 pA (sweeps 40 to 43). Limits: peak 2.1 mV, threshold 1.3 mV, minimum 1.2 mV,
slopes 19.5 V/s, first-spike latency 25 ms (near rheobase), return 0.9 mV.

**E cell.** Bias is −3.7 pA on both calibration sweeps; no forensics needed.
Human repeat set: long square 200 pA (sweeps 56, 59, 60, 61, 62; 57 and 58 did not
fire). Limits: peak 0.55 mV, threshold 0.97 mV, minimum 1.0 mV, slopes 5.7 V/s,
latency 83 ms, plateau 0.6 mV, return 0.9 mV.

## Model numerical repeats

- I: source 0.19 nA at nseg ×9 and ×27; candidate 0.27 nA at ×9, ×27, ×81. The
  maximum rise rate moves 245 V/s between meshes and the late cycle length 12.8 ms,
  so those two landmarks cannot be searched at ×9; every other landmark is
  resolved to better than 0.15 units.
- E: candidate sweep 50 at atol 1e-10, 1e-11 and nseg ×3. All landmarks resolved
  to better than 1.7 V/s and 0.05 mV.

## Function branch: search rows

**I cell, source (0.19 nA, 11 vs 12 spikes).** Elemental loop wrong from the first
spike: peak +44.2 vs +19.6 mV (12×), maximum fall −141 vs −336 V/s (10×), rise
1388 vs 597 V/s (real; mesh-limited). Late train: the human threshold rises from
−60.5 to −52 mV across the train, the model's stays at −60.3 (6.7×); late cycle
120 vs 197 ms (6×).

**I cell, candidate (0.19 nA 15 vs 12; 0.27 nA 43 vs 43).** The elemental loop is
repaired: peak, rise and fall are inside repeatability at both inputs. What remains:
the post-spike minimum is 5.5 to 8 mV too shallow at every spike (0.19 nA 4.4×
real; 0.27 nA 5.7 to 6.5× search), and the late-train threshold accommodation is
absent (6.5 to 6.8× at 0.19 nA; 1.9 to 2.6× at 0.27 nA). Late cycle: 88 vs 197 ms
at 0.19 nA (8.5×), one row; the other late cycles are within the mesh limit.

**E cell, source (sweep 50, 7 vs 5 spikes).** Maximum rise 1019 vs 351 V/s
(108 to 115×), the one unmistakable row. Threshold +3.4 mV (3.6×), peak +1.2 mV,
minimum −1.5 mV: real. Sweep 43 plateau +4.1 mV (6.9×).

**E cell, candidate (sweep 50, 5 vs 5).** Maximum rise 641 vs 351 V/s (48 to 52×)
at every spike. Threshold +2.7 to +3.8 mV (2.4 to 3.9×), fall, minimum, peak real
or parked. Sweep 43 plateau +1.6 mV (2.8×) and return +1.6 mV (1.9×): real, not
search. The "late subthreshold return" named by the 2026-09-06 E campaign is inside
two times its limit once the human's own repeat spread (0.9 mV) is counted.

Cycle-length rows for E carry a model-only limit (no human repeat at a spiking
amplitude); their ratios are against numerical noise and are listed, not ranked.

## Decision

| Cell | Branch | Search landmarks | Parked |
| --- | --- | --- | --- |
| I | Function | minimum_mv, late threshold_mv (accommodation), late cycle_ms (mesh-limited) | rise rate (mesh), return, plateau, passive inputs |
| E | Function | max_rise_v_s | threshold, peak, minimum, fall, plateau, return |

Both contrasts live in the function. Neither is carried in by the recording: the
human repeat limits are an order of magnitude below the search rows. No
interdependency between inputs and function is needed to explain the rows, and the
I cell's remaining contrast is the same at both current levels (structural family
flat).

Stage 2 opens for both cells. Predictions registered before the multivari:
I contrast is elemental (minimum depth) plus temporal (accommodation); E contrast is
elemental (rise rate) at every spike and every input.
