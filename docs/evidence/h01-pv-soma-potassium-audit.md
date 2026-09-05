# Somatic Kv3 intervention

Only somatic Kv3_1 conductance changes. Sodium and axonal channels retain
their source values. All cases use the same 0.19 nA step and zero bias.
The initial voltage is -80 mV. Mesh factor is 9. CVode tolerance is 1e-10.

| Case | Spike count | First peak, mV | First time above -20 mV, ms |
| --- | ---: | ---: | ---: |
| Human trace | 12 | 19.5625 | 0.273395 |
| Source conductance | 11 | 44.2482 | 0.774893 |
| Twice source conductance | 6 | 41.7474 | 0.651024 |
| Four times source conductance | 0 | No spike | No spike |

Increasing Kv3 alone reduces spike width and height, but removes spikes
before either waveform observation approaches the human datum.
The source model is too broad as well as too high.
Neither changed case is a qualified candidate.
Do not continue this one-parameter increase as a calibration method.
Future fitting must constrain voltage-dependent activation and inactivation
along with conductance and recovery. Density alone has not resolved the error.
This result does not prove that a unique kinetic parameter is wrong.

The two densities, 0.857697 and 1.715394 S/cm2, lie within the original
optimizer's 0-3 S/cm2 bounds. This does not make them measured human values.
The original NPZ files retain direct soma and axon voltage, calcium, and gates.
The [JSON audit](h01-pv-soma-potassium-audit.json) records every somatic event
and response-change RSS on a fixed 200000-sample grid.
RSS describes the controlled intervention, not biological uncertainty.
Time above -20 mV is not AP half-width. No peak alignment is applied.

Run `python -m docs.evidence.h01_pv_soma_potassium_audit` to repeat analysis.
The simulation driver uses `--scale-conductance Kv3_1 --conductance-region soma`
and `--conductance-factor 2` or `4`, with the conditions stated above.
