# PV transfer drift: simulator isolation split

Specification: [2026-09-05-h01-transfer-isolation](../specs/2026-09-05-h01-transfer-isolation.md).
Audit: [h01-pv-transfer-isolation-audit.json](h01-pv-transfer-isolation-audit.json),
written by `h01_pv_transfer_isolation.py` with the sha256 of every input.

## Design

Behavior: rise-crossing drift at events 7 and 8 of the frozen PV candidate in
BrainCell, 0.27 nA, 270-329.5 ms window. Two elements were swapped in a 2x2
half-split with everything else held equal and asserted by the audit:

| Element | Level 0 | Level 1 |
| --- | --- | --- |
| A mesh | BrainCell MaxCVLen 2.5 um (1223 compartments) | CVPerBranchList copied from the NEURON x9 run (1413 compartments, equal per branch) |
| B integration | NEURON CVode atol 1e-10 | NEURON fixed step, dt 0.000625 ms (the BrainCell dt) |

Gates: rise crossing 0.1 ms, peak 0.1 mV, time above -20 mV 0.01 ms, equal
event count. Every trace holds 8 events in the window.

## Reversible isolation

Halving dt at the matched mesh moves event 8 by 0.002 ms in NEURON and
0.002 ms in BrainCell; the largest change over all eight events is 0.022 ms in
each. Both pass the gates, so the 0.1 ms gate is a valid decision limit.
NEURON CVode against NEURON fixed step at the same mesh also passes
(largest change 0.044 ms).

## Result

| Cell | Rise gate | Peak gate | Width gate | Event 7 rise error (ms) | Event 8 rise error (ms) |
| --- | --- | --- | --- | ---: | ---: |
| MaxCVLen x CVode | fail | pass | pass | +0.108 | +1.224 |
| MaxCVLen x fixed | fail | pass | pass | +0.153 | +1.228 |
| matched x CVode | pass | pass | pass | -0.044 | -0.004 |
| matched x fixed | pass | pass | pass | 0.000 | 0.000 |

Decision recorded by the audit: **mesh**. Both matched-mesh cells pass; both
MaxCVLen cells fail on the rise gate only. The integration scheme moves no
event by more than its gate. At matched mesh and matched fixed step the two
simulators agree to below 1e-8 ms at event 8, so the gate time-level
difference between NEURON and the staggered BrainCell solver is not a
measurable contributor at this dt.

The drift sign differs from the earlier full-response comparison
(-1.7 ms at event 8) because that comparison used a NEURON reference with the
axon refined 2187 times; the mesh, not the simulator, set the sign.

## Measured durations

| Run | Seconds |
| --- | ---: |
| NEURON CVode, 330 ms | 10 |
| NEURON fixed dt 0.000625, 330 ms | 72 |
| NEURON fixed dt 0.0003125, 330 ms | 128 |
| BrainCell MaxCVLen 2.5, dt 0.000625, 330 ms | 398 |
| BrainCell matched, dt 0.000625, 330 ms | 556 |
| BrainCell matched, dt 0.0003125, 330 ms | 1038 |

## Limits

This is a numerical transfer result on donor anatomy. It identifies no channel
cause, no human waveform validity, and no H01 result. It qualifies the
transfer only at the matched mesh, this dt, this input, and this window; the
full 1270 ms response and other inputs are not covered.
