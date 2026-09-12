# Measured glial fragment assembly

This phase reconstructs an explicitly selected H01 glial component as a dynamic
Kir-only BrainCell cable. It follows coupling commit `de1f1e2` on
`feat/h01-braincell-biology`. The full biological H01 model remains incomplete.

`H01GlialSelection` pins the acquired skeleton digest, identity, coordinate
frame, original anchor vertex, maximum selected size, attribution and rationale.
`load_glial_fragment` selects that anchor's entire connected component, preserves
every source edge, coordinate and tapered radius, and records all excluded
components. Source nm are converted to um without translation or rescaling of
anatomy. Branch paths retain original vertex/edge array indices. The cable's
computational `dendrite` branch label does not classify the glia as a neuron.

`build_glial_cable` checks reconstruction identity and requires explicit
electrical settings. It returns an uninitialized cable and a canonical assembly
identity. No soma or inter-fragment bridge is invented. Fragment boundaries are
sealed. EnvironmentPotassium supports later explicit pool binding; until then,
the declared K pools are fixed. No calcium or uptake is implied by construction.

## Measured probe

The acquired candidate is H01 astrocyte `63900941936`. The selected component
contains original vertex `88562`, identified in the previous pinned proximity
audit near neuron `620880207`. Selection is for a local diagnostic: the reported
0.201338 um centerline-vertex distance is not proof of contact or connectivity.
The component's canonical lowest vertex is `81456`; previous audit label `189`
is informational, not the selection identity.

The complete component has **571 vertices and 570 edges**, reconstructed as
**148 branches and 148 compartments** at maximum CV length 10 um. All **256
excluded components and their 103,562 vertices** remain accounted for in
`assembly.json`; no claim is made that this fragment is a whole astrocyte.

`measured-probe.json` records 20 compiled steps at 0.005 ms: **0.1 ms biological
time**. All voltages were finite; a fresh source load and cable reconstruction
produced an exactly matching trajectory. With diagnostic initial voltage -100 mV,
fixed intracellular/extracellular K of 140/10 mM and paper Kir conductance
0.4 mS/cm2, final voltage was approximately -99.314655 mV. This is a short
fixed-pool electrical check, not coupled glial uptake or physiological validation.

Run from the checkout root in the scientific development environment:

```
python -m docs.biology.glial_fragment_probe
```

The default asset is `.cache/h01-biology/astrocyte-63900941936.npz`. The script
does not download. An alternative path may be supplied with `--source`, but its
bytes must match the pinned acquired-source hash. `selection.json` and
`assembly.json` contain the source URL, CC BY 4.0 attribution and reconstruction
identity. The existing `acquire_h01_glia.py` is the source acquisition utility.

## Edge cases and remaining checks

Co-located tests reject changed source bytes, invalid manifest fields, cycles,
duplicate/self/out-of-range/noninteger edges, zero-length segments, nonpositive
or nonfinite radii, malformed coordinates, isolated anchors and oversized
components. Endpoint, junction and degree-two anchors preserve all source
segments. Tests also cover changed reconstruction identity, invalid electrical
settings, exact reset and fresh reconstruction, and identity changes when a
biological setting changes. Coverage and affected CPU tests are recorded here.

The affected CPU gate passed **97 tests with six existing session readout-state
warnings**. Both new production modules have **100% statement coverage**. One
additional axial-current test passed after the broad gate: with Kir disabled,
a local voltage perturbation spreads through the tapered cable without leaving
the initial voltage bounds. `verification-summary.json` pins both gate artifacts
and the final implementation hashes; no GPU or full-repository gate is claimed.

A scalar-coordinate regression initially raised an incidental TypeError. Shape
validation now runs before taking vertex count and returns the intended domain
error. Future source loaders should validate rank before dimension-dependent
operations. The probe initially referenced an unavailable `Cell.morphology`
attribute for reporting; branch counts now come from the preserved assembly
record, avoiding runtime internals.

Next tests must qualify tapered CV volumes and calcium/spatial discretization,
explicit extracellular neighborhoods, coupled time/space refinement and complete
session save/rebuild/restore with these measured glial assets. Proximity must
not silently create release sites. Long calcium waves, myelin electrophysiology,
joint sparse learning and 4/12/40/all-104 execution remain unqualified. All probes
in this phase stayed within the 15-minute local-run limit.
