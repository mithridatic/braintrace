# Separate sodium inactivation and recovery

The prediction and decision rules were specified before whole-cell simulation.
The test changes the time scale only when h moves toward or away from
availability. Equilibrium curves and maximum conductance remain unchanged.
The source equations remain in a separate cache. Production code is unchanged.

All 16 fixed-voltage checks pass from h=0 and h=1 at -80 and 20 mV.
These include both isolated changes and the unchanged case.
The control satisfies the predeclared 0.01 mV voltage reproduction limit.

| Case | First time above -20 mV, ms | Spike count |
| --- | ---: | ---: |
| Unchanged | 0.77489334 | 11 |
| Inactivation twice as fast | 0.45129186 | 19 |
| Recovery twice as fast | 0.77489854 | 11 |

The inactivation-only intervention shortens the first spike by 0.32360148 ms.
The recovery-only difference is about 0.00000521 ms in the opposite direction.
It is too small to claim a resolved biological effect.
The predeclared directional width prediction is supported in this model.

The increased count also occurs without accelerated recovery.
Thus, the interpretation that faster recovery explains the prior 19-spike
train is not supported under this protocol. Do not retain that explanation.
The unchanged count does not mean unchanged behavior. Recovery-only peak
shifts grow to +5.154551 ms at the last spike on mesh factor 9.
This timing observation is conditional on that mesh and protocol.
The downstream cause of the train change remains open. Shorter spikes can
change calcium entry, potassium activation, and later excitability. This test
does not distinguish those pathways.

This is a conditional simulation result. It does not establish human kinetics
or a qualified fit. The first peak and full train still fail the human target.
Spatial and temporal refinement of the intervention remain pending.

The mechanism uses a state-dependent split at h=hInf. This split is an inferred
diagnostic operation, not a measured molecular mechanism. The derivative is
zero at equality. The fixed-voltage analytic checks test both directions.
The preparation script first expected five identifier occurrences, but the
intended change has four. Its assertion stopped before writing the mechanism.
The corrected count and the compiled relaxation checks verify the actual change.
Future preparation checks should verify the expected changed expressions
rather than an unexplained identifier count.

Run `python -m docs.evidence.h01_pv_recovery_audit` to reproduce the event audit.
The [JSON result](h01-pv-recovery-causal-audit.json) retains every spike and RSS.
The [clamp checks](h01-pv-recovery-clamp.json) retain expected and observed gates.
Use `--sodium-h-tau-factor` for inactivation and
`--sodium-h-recovery-factor` for recovery in the isolated mechanism container.
