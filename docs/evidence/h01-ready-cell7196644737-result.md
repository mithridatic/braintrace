# Isolated reproduction of the full-population failure

The [paired decision](h01-ready-cell7196644737-dt-decision.json) reproduces the
failure in cell 7196644737 alone. Both builds exactly match this cell's complete
metadata in the full-104 construction reference: source anatomy, electrical
regions, donor parameters, output location, 1 nA input and 1,647 compartments.
The full model has no projections incident on this cell. Both diagnostic runs
use the verified installed wheel and retain their failed trace arrays.

| dt (ms) | First nonfinite voltage (ms) | Maximum finite voltage (mV) | Process wall (s) |
| --- | --- | --- | --- |
| 0.005 | 3.2700 | 383.810459 | 15.928 |
| 0.0025 | 3.2675 | 383.792136 | 15.718 |

Soma and output voltages fail together. The failure persists without other
cells or synapses, and this timestep halving does not repair it. These two
runs do not rule out every numerical cause or identify a channel mechanism.
The retained arrays now permit inspection of the first divergent state.

Next inspect membrane area and current placement, then calcium and channel
state behavior under the unchanged input, before selecting another intervention.
Do not clip states, alter anatomy or reduce the population to obtain a pass.
No physiological fit was changed. The full-104 driven runtime, controls,
refinement and physiological gates remain open.
