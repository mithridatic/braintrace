# Doubled somatic Ih is insufficient for the response shape

The prospective sufficiency prediction is rejected. Only somatic Ih
density doubles. All other fitted parameters, geometry, solver settings,
source build, waveform, and bias remain consistent. Raw sample mapping,
endpoint, and applied-current plateau checks pass.

| Time, ms | Human change from 1019 ms, mV | Doubled-Ih model change, mV |
|---|---:|---:|
| 1019 | 0 | 0 |
| 1120 | 10.3125 | 12.050406 |
| 2019 | 8.8125 | 12.874848 |
| 2099 | -1.3125 | 1.392456 |

The model still rises during the selected constant-input interval and
remains above its starting voltage after the pulse. It produces no
complete pulse spikes. The [full audit](h01-l2-sweep43-ih-result.json)
retains all 12 direct voltage samples and signed errors. Compared with
control, late deflection falls only from 12.932369 to 12.874848 mV.
The initial absolute voltage error changes from -0.015036 to +0.056587 mV.

Helper isolation and driver guards pass 31 tests, with 100 percent
statement coverage of the density helper. The retrieved fit independently
confirms that one density changes from 3.474747445e-5 to 6.949494891e-5
S/cm2 and that the default multiplier preserves the source fit.

This result rejects the tested somatic factor as sufficient. It does not
exclude other Ih densities, distributions, or kinetic changes, and it
does not identify the cause in the human cell. Candidate-specific
numerical qualification remains open. Do not promote this candidate.
