# First voltage fall at matched levels

These observations use the existing human, slope-5 control, and half-somatic-
Ca_LVA traces. Each entry is the first downward crossing after the first
sampled peak, before the next upward -20 mV crossing. Times below are
elapsed from that trace's own sampled peak; the JSON retains absolute times.

| Input, nA | Falling level, mV | Human, ms | Control, ms | Half Ca_LVA, ms |
| --- | ---: | ---: | ---: | ---: |
| 0.19 | -40 | 0.215934 | 0.296409 | 0.292511 |
| 0.19 | -60 | 0.304600 | 0.660930 | 0.630985 |
| 0.19 | -65 | 0.339762 | 0.828988 | 0.777672 |
| 0.19 | -70 | 0.387692 | 1.154199 | 1.025146 |
| 0.27 | -40 | 0.216737 | 0.300816 | 0.294371 |
| 0.27 | -60 | 0.303810 | 0.712722 | 0.652866 |
| 0.27 | -65 | 0.337857 | 0.933240 | 0.818208 |
| 0.27 | -70 | 0.384528 | absent | 1.139284 |

Neither model case reaches -75 mV in this interval at either input.
Both human traces do. The missing crossing is retained, not extrapolated.

The half-conductance intervention reaches each shared falling level earlier
relative to its peak than the control. This supports an improved early
voltage return, despite a later minimum. The later minimum reflects an
endpoint at a different voltage and cannot alone diagnose a slower return.
The remaining matched-voltage delay is large at both inputs.

This is a direct observation of the existing conductance split, not a new
isolated mediator test. It does not prove that a particular potassium-channel
parameter is responsible for the remaining delay. First-peak sampling and
crossing interpolation also limit timing precision.

The [JSON](h01-pv-return-crossings.json) retains all requested levels,
absolute times, elapsed times, missing values, and crossing multiplicity.
Run `python -m docs.evidence.h01_pv_return_crossings`. Two helper tests pass:
nonuniform samples with repeated and missing crossings; exact-level plateaus
with an exclusive stop boundary. The reserved human input was not read.
