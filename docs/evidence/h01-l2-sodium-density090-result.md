# Lower sodium density does not satisfy the full prediction

The declared combined prediction is rejected. Reducing somatic NaTs
density by 10 percent improves first-onset and first-peak errors, and
all five positive-peak events remain. However, three interval errors
worsen. The [full audit](h01-l2-sodium-density090-result.json) retains
all events, intervals, interspike minima, and matched human residuals.

| Direct error | Control | Reduced density |
|---|---:|---:|
| First onset, ms | -4.534849 | -3.038188 |
| First peak, mV | +1.237200 | -0.953206 |
| First duration above -20 mV, ms | -0.021040 | -0.040327 |
| First interval, ms | -7.349581 | -8.144695 |
| Second interval, ms | -61.550693 | -75.741722 |
| Third interval, ms | -14.202627 | -14.573861 |
| Fourth interval, ms | +1.566464 | -1.207055 |

Setup isolation passes. The applied genome changes only the declared
soma / NaTs / gbar_NaTs row, from 2.934071278 to 2.640664150 S/cm2.
Time and applied current match exactly. Geometry, temperature, initial
state, mechanism build, calcium factor, and recovery factor match.
The source fit is preserved. Helper and CLI checks pass 27 tests; the
event-decision audit passes 11 tests. Both new helper modules have
100 percent statement coverage. These tests do not qualify cell biology.

The result supports the conditional onset-delay and peak-reduction
effects at this discretization. It rejects their sufficiency for the
combined direct-response targets. Do not promote this candidate. Its
specific numerical checks are not complete. The reserved input remains
outside fitting. Future selection must consider waveform and every
interval together, rather than treating peak correction as sufficient.
