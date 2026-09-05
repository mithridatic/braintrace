# Spatial check of the high-input firing failure

The combined candidate at 0.27 nA has no positive spikes after 500 ms on
either spatial factor 9 or 27. Each trace has three positive spikes and
one later complete -20 mV excursion with a negative peak. The absence of
a sustained train persists under this refinement.

Fine-minus-coarse onset changes for the first three events are 0.000410,
0.003340, and 0.007972 ms. The fourth excursion shifts by 0.027594 ms,
and its negative peak changes by 0.441018 mV. Late sampled voltage bounds
are -29.870584 to -29.033319 mV on factor 9 and -30.032657 to -29.223520
mV on factor 27. Do not call the full voltage response converged.

The source parameters, integration tolerance, stimulus, and initial state
match. Section counts and connections match, and each section has three
times as many segments. Lengths, integrated areas, Ra, and Cm agree within
the audit tolerance. Reported section diameters differ slightly in tapered
sections; those differences are retained in the JSON.

The initial audit incorrectly required these discretized diameter values
to be identical. A regression test reproduced the failure before correction.
The corrected audit checks the physical length and integrated area and
retains the diameter differences. Future refinement checks must distinguish
physical source geometry from discretized summaries.

The [audit](h01-pv-failure-mesh-audit.json) preserves all events. Reproduce it
with `python -m docs.evidence.h01_pv_failure_mesh_audit`. Its retained-data
regression passes. This is two-mesh support for a specific observed failure,
not proof of a unique channel cause or full numerical convergence.
