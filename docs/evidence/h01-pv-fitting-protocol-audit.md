# Fitting protocol audit

The pinned optimization files define the same step timing, current levels,
temperature, and recording sites as the reference driver.
The optimizer starts at -81 mV. The released circuit starts at -80 mV.
The existing initial-state intervention tests this difference.
It changes timing but does not resolve the high spike peaks.
The optimizer runs for 2000 ms. The reference runs for 1500 ms.
Both include the full 270-1270 ms current step.
No additional holding pulse is defined in the inspected fitting protocol.
This does not establish the holding current used during human acquisition.

The optimizer includes AP height. Its normalization scale is 4 mV.
The spike-count scale is 0.05. These are fitting weights.
They are not biological standard deviations or acceptance limits.
The optimization combines several feature objectives. Thus, inclusion of
spike height does not prove a close fit to each recorded spike.
The present discrepancy cannot be assigned to omission of a height objective.

The source SWC files have identical Git blob hashes.
Ten of eleven channel and calcium mechanism files also have identical hashes.
Kv3_1 differs in the declaration of its voltage shift and whitespace.
The optimizer declares a local RANGE parameter. The circuit uses a global
parameter. Both set it to zero. The rate equations are unchanged.
This difference does not explain the tested cell's high spike peaks.
Runtime equivalence of the original BluePyOpt replacement axon remains unchecked.

The [machine-readable audit](h01-pv-fitting-protocol-audit.json) records hashes,
protocol values, and limits. Source code is pinned to ModelDB 267587 commit
`82cdd91bc93942ba19315371330a2412e064baf5`.
See the [optimization source](https://github.com/ModelDBRepository/267587/blob/82cdd91bc93942ba19315371330a2412e064baf5/Single%20Cell%20Modelling/Optimizations/L5PV/init_active.py).

Raw acquisition filtering and voltage-correction provenance are still open.
Do not apply an offset or filter to obtain a better fit without that evidence.
Do not treat the published circuit parameters as a validated direct waveform fit.
