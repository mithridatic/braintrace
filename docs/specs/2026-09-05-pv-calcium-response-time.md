# Calcium response time and early-to-late adaptation

The both-fast, gamma-0.002, decay-1000-ms candidate has a third low-input
interval that is too short and a late interval that is too long.
At 0.19 nA, test (gamma, decay_ms) pairs (0.004, 300) and (0.008, 150).
Keep all other candidate parameters, the phase build, mesh factor 9, and
CVode tolerance 1e-10 fixed. These are inferred calibration candidates.

For imposed constant calcium current and fixed pool depth, the stationary
excess calcium in the source equation is proportional to gamma times decay.
Both new pairs have product 1.2, versus 2 in the preceding candidate.
They have equal stationary gain under that restricted condition but
different response times. In the full cell, calcium current and SK feedback
change, so equal product does not establish equal mean calcium or firing.

Compare the first three positive-spike intervals and the first positive
pair whose upward crossings occur at or after 1000 ms. Require the absolute
errors of both the third interval and late interval to decrease against
the gamma-0.002, decay-1000-ms control before selecting a candidate for
the higher-input check. Missing events invalidate the corresponding screen.
Retain the first two errors and first shape separately. This screen does
not establish a correct burst or authorize promotion.

Retain all direct events and calcium/voltage/current observations. Do not
use the reserved input. A selected candidate still needs the 0.27 nA
comparison, spatial checks, and human physiological qualification.

## Spatial check of the selected candidate

At 0.27 nA, refine all sections from spatial factor 9 to 27. Keep the
selected gamma 0.004, removal time 300 ms, other parameters, phase build,
and CVode tolerance 1e-10 fixed. Verify section identity, parent, length,
integrated area, axial resistivity, and capacitance. Sampled diameter need
not be identical when segment centres move along a tapered section.

Before this run, set these diagnostic numerical limits: each of the first
three interval changes must be at most 0.05 ms, and the selected late
interval change must be at most 0.25 ms. Missing positive-spike pairs make
the comparison invalid. These are engineering limits for this diagnostic,
not measured human variability or full model acceptance limits.

Retain every event and ordinal timing difference, including unmatched
events. Compare first onset, peak, and duration separately. A pass supports
these interval diagnostics on two meshes only; it does not establish
convergence of the complete trajectory or permit production promotion.

## Third spatial level

After the factor-9 to factor-27 late-interval check failed, run factor 81
at the same high input and all other settings. Compare factor 27 to 81
with the same 0.05 ms early and 0.25 ms late limits. Retain the earlier
failed result. Require valid positive-spike pairs and geometry invariants.
Report all ordinal differences and unmatched events; a local interval pass
does not establish convergence of all event times.

Do not use the existing regional option for a factor-9 baseline split:
it leaves unselected sections at factor 1. Such a run changes more than
the intended region relative to the current baseline.

## Time integration at factor 27

Compare CVode absolute tolerance 1e-8 with the saved 1e-10 response on
factor 27, at 0.27 nA. All cell and geometry parameters remain fixed.
Retain all complete excursions and ordinal event differences. Require
equal event counts, a maximum ordinal onset change of 0.01 ms, first
three interval changes at most 0.01 ms each, and selected late interval
change at most 0.01 ms. Missing positive-spike pairs invalidate the test.
These prospective engineering limits test tolerance sensitivity; they
are not human variability bounds or proof of exact integration.

A pass would bound this tolerance effect below the observed spatial
changes. It would not establish full spatial convergence or physiological
validity. A failure requires further time-integration diagnosis before
attributing the observed numerical discrepancy mainly to the mesh.
