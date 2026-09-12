# BRAINCELL biology in the H01 JAX execution path

Status: implementation in progress; tested mechanism and small-cable milestone.
Full spatial assembly and 104-neuron forward/training qualification are unfinished.
Current evidence and remaining gates: [biology report](../biology/README.md).

## Contract

Port the biological experiment families in bioRxiv
10.64898/2026.08.22.746419v3 to the existing JAX/BrainState runtime:
explicit spines, stochastic synaptic release, extracellular potassium and GABA,
astrocyte calcium reaction-diffusion and potassium buffering, and myelinated
axon potassium transport. Complete with the selected 104 H01 neurons and a
bounded matched ARC training comparison. Do not interpret feature presence,
reference agreement, or training as human physiological validation.

Reference: RusakovLab/BRAINCELL at
8acd1c23032a17af59ae1f813d142735fef8689c. Preserve per-file source digests and
license notices for reused source. Paper configurations and repository defaults
are separate: 500 synapses in the first results description conflicts with 316
spines in the methods; calcium buffer/diffusion settings also differ. Never
silently resolve a discrepancy into an exact-reproduction claim.

## Architecture

Reusable mechanisms live in `braintrace/biophysics`; H01-specific anatomy and
placement remain dataset adapters. Geometry and rates are immutable model
configuration. Continuous physical states belong to BrainState and use compiled
transforms for repeated evolution. Use brainstate.random for random draws.
Use sparse finite-volume transport with implicit differentiation, never dense
compartment-squared Jacobians. Explicit source-history diffusion is a separate
reference mode and must not be confused with conserved molecule injection.

Concentration units are mM, lengths um, time ms, voltage mV; finite-volume amount
is mM*um^3. Each membrane exchange is applied once with opposite signs to its
source/destination; clamped reservoirs and boundary exchange are recorded.
Reject nonpositive volumes, malformed topology, invalid rates, nonfinite state
and nonconverged solves. Do not clip invalid physical states into passing runs.

## Anatomy

Prefer verified matching H01 structures, then published human reconstructions,
then the paper's reconstructed source models, then source-constrained synthesis.
H01 c3 segmentation/skeletons, proofread meshes, astrocyte labels and myelin masks
are acquisition candidates, not certified completed structures. Verify identity,
frame, continuity, attachment and truncation. Preserve original neuron geometry;
store additions and inferred transformations separately, with stable IDs and
digests. No duplicated membrane area or contact. Default generation seed: 21.

## Integration

New mechanisms are opt-in. Keep existing H01 manifests and baseline equations.
Add a shared chemical environment to network stepping; close sparse pp-prop
influence over actual environmental coupling. Train only encoder, contact
magnitudes and readout; freeze rates, release probabilities and anatomy.
Stochastic derivatives condition on recorded masks, not on probability gradients.
Keep 441 input features, 360 logits and 0.1 ms event intervals; optional zero-input
physical intervals of 0, 1, 10 and 20 seconds precede answer requests in matched
arms. Never accelerate biological rates. Padding advances nothing.

Reset all physical, random, queue and eligibility state between independent
episodes. Versioned v2 checkpoints additionally support physical-step boundaries
inside long waits. Restore clocks, physical and chemical state, random counters,
eligibility, optimizer, queues and immutable manifests together. Legacy import
is explicit and only valid for the original mechanism set. Rebuild spatial maps
after topology mutation and preserve original/synthetic identity distinctions.

## Gates

1. Pin source evidence and test reusable units independently.
2. Validate spine/release and transport analytic limits, mass/boundary balance,
   receptor responses and recorded random-mask reproducibility.
3. Compare source-equivalent astrocyte and myelin mechanisms with isolated NEURON.
4. Integrate single systems, then 4/12/40/104-neuron populations.
5. Validate derivatives against AD/finite differences and learning through
   finite-window chunked_online_param_gradients. Keep approximate pp-prop
   qualification distinct from exact BPTT regimes.
6. Require restart equivalence, finite nonzero applicable updates, diagnostic
   descent, and the full 104-cell forward/learning gates.

Retain exact spike-count, 1 ms timing, 1 mV voltage and 0.05 ms phase limits.
Chemical peaks, decay and spatial profiles must change by <5% under refinement;
record full trajectories as well. New modules require meaningful sibling
`*_test.py` coverage above 90%. Write regression tests before bug fixes.

Local probes are capped at 15 minutes. Estimate measured time/memory before
requesting longer or paid runs. Stops remain incomplete. No GUI, assistant,
unrelated ARACHNE feature, unsourced channel-noise model or biological-rate fitting.

## Recorded implementation decisions, 2026-09-12

The calcium step solves reactions and diffusion together with matrix-free
Newton-Krylov and local 25-state block preconditioning. Operator splitting did
not meet the bounded source comparison. The selected internal step is .00125 ms
(four solves per .005 ms cable tick), based on the strong three-segment pulse
refinement fixture. This is not a qualification of long-duration waves or all
measured morphologies. Fixed ER and plasma-pump exchanges have explicit ledgers.

The initial v2 physical snapshot supports whole .1 ms event boundaries inside
waits. Checkpoints inside individual .005 ms cable substeps remain unfinished.
Restore requires an already constructed, manifest-matched session. Source v1
snapshots cannot silently omit new biology. Biological topology mutation is
rejected until a complete remapping adapter exists, including the case where
the contact count stays constant but contact identities change.

Only fixed release probabilities are currently accepted by the H01 session
factory. Other primitives can be explicitly assembled in small systems; that
does not make the complete spatial model available through Example 21. Measured
astrocyte acquisition must not be described as a completed anatomical placement.

The finite-volume myelin transport is a distinct physical discretization. The
pinned RxD code and its comments disagree about axial diffusion and radial
geometry. Retain this distinction until independent reference experiments
resolve it; copied repository conversation notes are not scientific authority.
