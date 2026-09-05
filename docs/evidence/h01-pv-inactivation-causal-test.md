# Sodium timing: first causal test

The [test specification](../specs/2026-09-05-sodium-inactivation-causal-test.md)
states the prediction and rejection criteria before simulation.
The intervention halves NaTg hTau in soma and axon.
It preserves hInf, activation equations, and maximum conductance.
Original source files and production BrainCell channels remain unchanged.

The fixed-voltage check passes at -80, -20, and 20 mV.
Measured gate relaxation agrees with the analytic solution within 1e-8.
The activation gates agree between factors within 1e-10.
The modified-mechanism factor-1 control reproduces the original voltage exactly.

| Observation | Control | Half hTau |
| --- | ---: | ---: |
| First time above -20 mV, ms | 0.774893 | 0.451293 |
| First peak, mV | 44.24820 | 41.19062 |
| Number of spikes | 11 | 19 |
| h at falling -20 mV crossing | 0.069797 | 0.039021 |
| NaTg current at that crossing, mA/cm2 | -2.03023 | -1.18087 |

The spike-duration reduction is 0.323600 ms, above the predeclared 0.02 ms
diagnostic margin. The directional prediction is supported at these conditions.
The measured intervention causes a shorter spike in this model.
This does not establish a unique cause or complete mediation through one phase
of the current. Voltage feedback changes the gate trajectory, and the factor
changes both recovery and inactivation in two regions.

The human first duration is 0.273395 ms and its first peak is 19.5625 mV.
The intervention does not qualify as a human waveform fit.
The increase to 19 spikes also fails to match the 12-spike human train.
Numerical refinement and matched-voltage trajectory checks remain pending.

The first clamp test compared different times because continuerun stopped
before 0.5 ms. Its analytic assertion failed. The test now calls CVode.solve
to evaluate the exact endpoint. Retain exact endpoint checks in later gate tests.

The [clamp results](h01-pv-inactivation-clamp.json),
[source manifest](h01-pv-inactivation-source.json), and
[whole-cell results](h01-pv-inactivation-causal-audit.json) retain the evidence.
Current is outward positive. Event durations use interpolated -20 mV crossings.
No voltage shift or spike alignment is applied to the whole-cell traces.
