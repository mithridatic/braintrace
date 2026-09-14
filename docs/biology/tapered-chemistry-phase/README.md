# Tapered cable calcium and potassium geometry

Implemented after glial assembly commit `6fce59c` on
`feat/h01-braincell-biology`. `CableChemicalGeometry` now maps initialized cable
CVs to physical chemical volumes and diffusion. `AstrocyteCalcium(geometry=map)`
uses those volumes and actual membrane areas. The cylindrical API remains
available and is covered by its existing independent reference tests.

## Physical construction

The mapper clips every source linear frustum at its actual electrical CV
boundaries. A piece of length L and radii a,b contributes volume
`pi*L*(a*a+a*b+b*b)/3`, lateral area `pi*(a+b)*sqrt(L*L+(b-a)*(b-a))` and axial
geometric resistance `L/(pi*a*b)`. Piecewise radius changes within a CV are
retained. Length, area and both half-CV resistance integrals must agree with
BrainCell's electrical declarations. Every source branch must be covered once.

Four homothetic annuli use the source outer-to-inner volume fractions
`[11/36, 4/9, 2/9, 1/36]`. Full CV volume is also the intracellular K pool volume.
The calcium pump uses actual lateral membrane area divided by outer-shell
volume. This avoids substituting a midpoint-radius cylinder for a taper.

Each massless shared boundary has incident conductances `g_i`, the inverse of
half-CV geometric resistance. Eliminating the common concentration gives
pairwise conductance `g_i*g_j/sum(g)`. At forks this includes sibling diffusion.
BrainCell's direct midpoint edges retain the roles of collapsed degree-two
boundaries; those are included too. Sealed endpoints add no fictitious volume.
Source radial conductance is integrated per cable length; bound mobile buffer
keeps longitudinal transport only.

This is a **conservative finite-volume taper extension**. Homothetic shells and
separate axial/radial transport are modeling assumptions, not a solved 3-D taper
problem or demonstrated agreement with a tapered NEURON RxD model. The chemical
map hash pins CV order, scalar source dimensions and boundary topology. It does
not pin absolute anatomical coordinates: use it alongside the source assembly
identity, as done in the probe's physical checkpoint manifest.

## Measured probe and replay

`measured-probe.json` uses the previously selected 148-CV fragment of H01 glia
`63900941936`. Its exact integrated intracellular volume is
**21.5226991986 um3**. Calcium shell volumes sum to the K volumes with maximum
floating-point discrepancy **2.22e-16 um3**.

The probe runs **three 0.005 ms steps (0.015 ms)**. Diagnostic extracellular
volumes are ten times each CV's volume, isolated from one another; these are
explicit finite pools, not reconstructed extracellular anatomy. Initial K is
140 mM inside and 10 mM outside. A diagnostic initial calcium pulse sets CV 0's
outer shell to 0.001 mM. There is no neuron or inferred release site in this probe.

All validity checks passed. Recorded total K balance error is zero, and calcium
plus bound-buffer change minus net ER/pump exchange is **1.20e-17 mM um3**.
Intracellular K increased, demonstrating dynamic Kir uptake using the mapped
volumes. A snapshot after step one was restored into a fresh source/cable/
calcium/environment reconstruction; all packed physical state arrays matched
exactly after continuation. Reset plus reapplication of the declared initial
pulse also replayed exactly. This does not demonstrate a sustained calcium wave.

Reproduce from the checkout root in the scientific development environment:

```
python -m docs.biology.tapered_chemistry_probe
```

The source asset remains `.cache/h01-biology/astrocyte-63900941936.npz`; its hash
is checked against the prior source selection. No download is performed.
`geometry.json` contains the complete physical map, and the verification summary
pins source and artifact hashes. Each local probe stayed below 15 minutes.

## Edge cases, corrections and remaining qualification

Tests cover analytic taper integrals, subdivision invariance, piecewise radius
changes, the exact cylindrical limit, shared-fork flux, positive conservative
transport, finite-difference gradients, calcium/buffer mass and exact shell/K
volume agreement. Malformed CV ordering, incomplete partitions, missing/duplicate
boundary ends, unsupported midpoint attachments, altered electrical geometry,
negative diffusivity and mixed constructor geometry are rejected.

The affected CPU gate passed **201 tests with six existing session readout-state
warnings**. Statement coverage is **99.05% for CableChemicalGeometry** and
**100% for the updated AstrocyteCalcium module**. This includes the existing
cylindrical independent-reference tests; no GPU or full-repository gate is claimed.

Initial tests exposed that inverse units must be passed as Unit objects
(`u.um**-1`), and that not every physical boundary has an explicit runtime node.
The implementation now handles direct midpoint-edge provenance as well as
explicit junction nodes. The cylinder-limit and subdivision tests preserve
these corrections. Fixture fixes use a nonreserved branch name and test invalid
raw geometry separately from BrainCell's own constructor validation.

Remaining work includes extracellular spatial neighborhoods, explicit neuronal
release placements, complete measured-glia session construction, coupled temporal
and spatial convergence, long calcium-wave propagation, myelin electrical
dynamics and sparse learning qualification. No full H01 population, GPU,
long-duration or physiological qualification is claimed by this phase.
