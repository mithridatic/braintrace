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

## Internal-state observation

The [state decision](h01-ready-cell7196644737-state-decision.json) preserves the
baseline voltage exactly (0 mV maximum finite difference and identical finite
masks), all cell metadata and compartment geometry. The output compartment has
0.165031014 um2 membrane area; the modeled soma intervals total 0.763029420 um2.
The unchanged 1 nA point input divided by that output CV area is 605.946712
mA/cm2. This is an assumed diagnostic input, not measured H01 stimulation.

At 3.265 ms, output voltage is 378.118892 mV, calcium is 1.92819965e-7 mM,
and inward-positive calcium current is -0.215624671 mA/cm2. At 3.270 ms,
calcium is -2.95955282e-7 mM; Nernst reversal, SK state and voltage are NaN.
Calcium itself becomes NaN on the following recorded step. This probe does not
establish which of all 1,647 compartments becomes invalid first.

The installed family update caches step-start calcium current and integrates
calcium removal exponentially with that frozen current. Substituting the
observed current and source donor gamma/decay into that exact scalar update
predicts the negative calcium within 1.06e-22 mM. The source flux has no outward
current clamp; the current-freezing update can cross zero before Nernst feedback
reduces outward flux. The next numerical investigation should retain that
feedback inside the calcium solve, preserving signed flux and source parameters.
This identifies a numerical failure route; it does not qualify the borrowed
physiology or repair the extreme voltage response to the assumed point input.

The first observer attempt stopped before initialization because tuple-valued
live metadata was compared directly with JSON lists. Two regression cases now
accept only representation equivalence and reject changed geometry. The rerun
uses a fresh label and retains both launch/provenance records. No production
model equation or parameter changed in this observation.

## Experimental numerical repair

The [implicit-solver decision](h01-ready-cell7196644737-implicit-decision.json)
records finite 10 ms traces at both .005 and .0025 ms, with positive calcium
throughout and one event in each. Only solver identity changed; all other cell
metadata and compartment geometry match. Both use the source worktree; they are
not installed-wheel or full-population qualification. Maximum voltage remains
about 383.8 mV, so numerical finiteness is not physiological plausibility.

Matched end-step voltages differ by 6.364916 mV between the two timesteps. This
exceeds the 1 mV criterion; numerical refinement remains unqualified. Next inspect
the maximum-error location and convergence under smaller steps before choosing
the full-population timestep or changing the splitting order.

The first assembled-cell test exposed a unit error missed by the initial scalar
arithmetic oracle: mS/cm2 times mV gives uA/cm2, requiring a 1e-3 conversion to
mA/cm2. The corrected kernel and independent roots now include that factor, and
a new BrainUnit derivative oracle independently verifies the conversion. The
historical scalar decision predates this correction and is superseded by these
33 passing checks (kernel 100%, solver 96% line coverage). Unsupported non-ohmic
currents and non-family ordering are rejected. No-calcium traces match the
existing scan exactly; ordinary-regime differences shrink with timestep.

## Finer timestep evidence

The [extended refinement](h01-ready-cell7196644737-refinement-decision.json)
records consecutive voltage errors of 6.364916, 3.471139, 1.819563 and 0.933343
mV as dt halves from .005 to .0003125 ms. The last pair (.000625/.0003125 ms)
passes the 1 mV limit and retains one event each, with .0003125 ms timing
separation. All recorded arrays remain finite and calcium remains positive.
This qualifies only this isolated comparison. Full104 refinement remains open.
