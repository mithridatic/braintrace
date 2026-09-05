# First-onset current account

The candidate uses sodium closure factor 0.18 and recovery factor 1.
All other parameters retain their source values. There is no bias current.
The independent NEURON runs use spatial factor 9 and CVode tolerance 1e-10.
The reserved 0.23 nA trace was not read.

## Direct observations

At 0.19 nA, travel from -75 to -65 mV takes 22.80149 ms in the model
and 9.14000 ms in the human trace. At 0.27 nA, it takes 6.34142 ms in
the model and 5.34000 ms in the human trace. The delay is voltage-dependent;
a uniform shift of the trace cannot remove it.

At the first upward -70 mV crossing, the local current account is:

| Current at soma midpoint, nA | Input 0.19 nA | Input 0.27 nA |
| --- | ---: | ---: |
| Applied inward | 0.19000000 | 0.27000000 |
| Axial inward | -0.18625037 | -0.26298061 |
| Net ionic outward | 0.00286254 | 0.00285450 |
| Capacitive outward | 0.00088708 | 0.00416488 |

The local capacitance is 0.00309332 nF. The corresponding instantaneous
voltage slopes are about 0.287 and 1.346 mV/ms. Axial current comes from
the recorded neighbour voltages and resistances, not from a balance residual.

Both runs have exactly the same time and soma voltage arrays as their
unprobed candidate controls. The maximum absolute current-balance errors
are 3.55e-15 and 4.00e-15 nA. The limit is 1e-5 nA. Initialization and
stimulus discontinuities are excluded from this check.

## Interpretation and limit

At this location and voltage, the local ionic totals are nearly equal.
The remaining current that charges the local capacitance is much smaller
at the lower input. Most applied current leaves this small segment through
axial paths. That current can charge or cross other membrane regions.
It is not all dissipated locally, and it is not all dendritic current.

This explains the local voltage slope through charge conservation. It does
not identify the parameter that causes the mismatch with the human trace.
Human channel and axial currents were not measured here. Matching voltage
does not match the full cell state. In particular, a small soma channel
current does not rule out an effect of the same channel in the axon.

The next causal split should test regional active current before changing
measured geometry. A proposed sodium-conductance intervention must report
first-onset time and first-spike shape together. Earlier onset alone cannot
qualify a candidate that loses its waveform agreement.

The full crossing currents are in [the audit](h01-pv-onset-balance-audit.json).
The raw traces are `h01-pv-onset-balance-019.npz` and
`h01-pv-onset-balance-027.npz`. Reproduce the account with
`python -m docs.evidence.h01_pv_onset_balance` after the two reference runs.
The audit rejects missing or repeated model crossings and changed control
traces. No new production model code is included.
