# Human-response errors on the completed fine axonal mesh

The same calcium candidate at 0.27 nA still differs from the human
recording after axonal refinement. Values below retain the established
voltage and stimulus clocks. Duration means time above -20 mV, not
half-width. The peak is the largest recorded sample in the first excursion.

| Direct observation | Model | Human | Model minus human |
| --- | ---: | ---: | ---: |
| First upward -20 mV crossing, ms | 281.197858 | 281.865850 | -0.667992 |
| First peak, mV | 17.940147 | 19.437500 | -1.497353 |
| First duration above -20 mV, ms | 0.220733 | 0.267173 | -0.046440 |
| First interval, ms | 4.989954 | 6.229948 | -1.239994 |
| Second interval, ms | 4.259545 | 5.948006 | -1.688461 |
| Third interval, ms | 4.127157 | 6.880304 | -2.753146 |
| Selected late interval, ms | 28.166646 | 25.259988 | 2.906659 |

The first model post-spike minimum is -71.900057 mV, at 1.362963 ms
after the sampled peak. The [residual record](h01-pv-axon729-human-residuals.json)
retains all human and model events and the exact selected late pairs.

These are direct residuals, not newly chosen acceptance limits. The
factor-2187 comparison remains pending. Numerical qualification, if it
passes, will not establish physiological qualification. The first shape
and early-interval errors remain targets for channel diagnosis; a count
or one late interval cannot replace them. The reserved input was not used.
