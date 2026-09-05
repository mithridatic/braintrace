# Regional spatial sensitivity

Each intervention refines only the named region from source factor 9 to
27. Other regions stay at factor 9. Cell parameters, input, temperature,
and CVode tolerance are fixed. All section-count and physical-geometry
invariants pass the audit.

| Mesh | Selected late interval, ms | Complete events |
| --- | ---: | ---: |
| All factor 9 | 26.054280 | 43 |
| All factor 27 | 27.898400 | 41 |
| Axon factor 27 only | 27.896467 | 41 |
| Soma factor 27 only | 26.054409 | 43 |
| Dendrites factor 27 only | 26.056052 | 43 |

Axon-only refinement reproduces the full-refinement selected interval
within 0.001933 ms. It passes the predeclared 0.25 ms sufficiency limit.
Soma-only and dendrite-only refinement fail that limit. Their changes from
the coarse interval are 0.000129 and 0.001772 ms, respectively.

This isolates the axonal mesh as sufficient for the tested late-interval
change. It does not establish a unique current pathway, human physiology,
or convergence of every event. The full event records must remain available;
equal counts alone would not establish the result. Regional effects need
not add in this coupled system.

The [audit](h01-pv-regional-mesh.json) retains all events and intervals.
Run `python -m docs.evidence.h01_pv_regional_mesh` to reproduce the decision.
The prospective specification is
[regional mesh diagnosis](../specs/2026-09-05-pv-regional-mesh.md).

The new diagnostic driver option preserves source-factor-1 defaults for
unselected regions. Allocation tests first failed for the missing option,
then all 55 driver tests passed. They cover positive odd counts, invalid
counts, different initial section counts, and both dendrite families.
To avoid repeating the initial baseline mistake, the analysis verifies
each actual section count before it accepts a regional comparison.
Missing late positive pairs invalidate the sufficiency decision.

The next spatial check can concentrate refinement in the axon, but it must
still verify full event times against an independently refined reference.
No cell model or circuit is promoted by this diagnostic.

## Focused mesh against the full reference

Axon factor 81 with other regions at factor 9 passes the prospectively
declared comparison with the full factor-81 reference. Both runs contain
40 complete events with matching positive/negative classifications.
The largest ordinal onset difference is 0.081786 ms, below 0.1 ms.
The selected late interval differs by -0.002151 ms, within 0.01 ms.

The focused mesh contains 1557 compartments; the full mesh contains 12717.
Physical geometry and cell parameters match. The [focused audit](h01-pv-focused-mesh.json)
retains every onset, peak, and duration difference and verifies every
section's expected mesh. Run `python -m docs.evidence.h01_pv_focused_mesh`.

This supports use of the focused mesh for further numerical diagnosis at
this input and parameter set. It does not establish convergence of the
factor-81 reference, qualification at another input, or a measured runtime
speedup. The next check must refine the axon further while retaining the
full-train event criteria. No physiological or circuit claim follows from
agreement between these two numerical models.

## Axon factor 81 to 243

The next focused refinement fails the unchanged full-train limits. Both
runs have 40 complete events, but the largest ordinal onset change is
1.016766 ms, above 0.1 ms. The selected late interval changes by
0.027282 ms, above 0.01 ms. Equal event counts do not establish convergence.

The [factor-243 audit](h01-pv-focused-mesh-243.json) records every paired
event and verifies the fixed non-axonal mesh and tripled axonal mesh.
Run `python -m docs.evidence.h01_pv_focused_mesh --fine-factor 243`.
The factor-729 run is a separate pending comparison. No failed limit
is relaxed or replaced by the smaller compartment count.

Eighteen tests of the audit pass. Synthetic voltage traces exercise
matching responses, excessive onset shifts, excessive late-interval
shifts, unequal counts, missing late pairs, and changed event polarity
for the full-reference and both successive-refinement command modes.
These verify acceptance logic, not biological model behavior.

## Axon factor 243 to 729

This comparison also fails the unchanged onset limit. Both runs retain
40 complete events with matching classifications. The largest onset
change is 0.113263 ms, above 0.1 ms. The selected late-interval change,
0.003038 ms, is below 0.01 ms. The overall decision remains rejected.

The [factor-729 audit](h01-pv-focused-mesh-729.json) preserves all paired
responses. Run `python -m docs.evidence.h01_pv_focused_mesh --fine-factor 729`.
The smaller onset difference supports another refinement test, not a
post-hoc tolerance change or a claim of a converged human neuron.
