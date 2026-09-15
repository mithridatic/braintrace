# Human-only qualification of all six population terms

Status: approved by the user: "Execute the plan."
Objective: "Get the values to 1.000. True human only."
Branch: campaign/h01-human-unity-20260914, initially based on dba5ae6;
implementation checkout reverified at 23ed242 after intervening integration.

## Completion contract

Retain all 104 cells and all six terms. Intermediate single-cell diagnostics
are steps toward the full objective, never a replacement deliverable. Preserve
the original metrics, tolerances, raw traces, and failed decisions. A displayed
rounded 1.000 is insufficient if the underlying value fails its requirement.
Do not change denominators, drop failed observations, relax tolerances, or set
missing evidence to a pass. Report each term separately; their product does
not constitute a measured probability of human physiological accuracy.

Human-only means documented human provenance for anatomical and physiological
inputs and human validation of biological mechanisms used for qualification.
Human-fitted parameters alone do not prove human-specific channel kinetics.
Audit channel equations, kinetic data, conductance distributions, soma/axon
geometry, synaptic parameters, and any glial parameters on the executed path.
Shared mathematical/numerical methods do not themselves assert a species.
Unknown or animal-only biological evidence cannot earn human qualification.

There is currently no paired electrophysiology for these particular H01 cells
in the repository evidence. Comparison with another human donor validates a
specified transfer prediction, not the actual physiology of the H01 donor.
Never claim the latter without an appropriate experimental reference.

| Term | Evidence needed for completion |
| --- | --- |
| donor_accuracy | Original ten-element score reaches 1.000 without rounding or altered scoring; preserve direct observations and existing independent-input/holdout requirements. Qualify every donor used across the population rather than assigning B3's score to other donors. |
| type_coverage | 104/104 human donors or human-fitted models match documented layer and class; missing subtype evidence and ambiguous identities remain explicit. No relabeling a polarity-only match as a type match. |
| construction | Exact 104-cell inventory imports, constructs, initializes, and forwards with the final qualified geometry and mechanisms. Existing evidence must be rerun if those inputs change. |
| driven_window | The full 104-cell system completes the prescribed physiological window with finite, complete traces and matched ei/e_only/i_only/disconnected controls. Register duration, input, observation sites and acceptance rules before execution; a shorter diagnostic does not close the term. |
| timestep | Demonstrate convergence for the final population, controls and required observation windows under the existing numerical limits. The isolated-cell 10 ms ladder is only one piece of this evidence. |
| anatomy_transfer | Verify source-to-simulation anatomy and electrical load, then test human-constrained response predictions for each deployed donor/type on the retained H01 anatomy. A single restored spike, another donor's reconstruction, or increased input alone cannot close this requirement. |

## Current evidence inspected on 2026-09-14

The primary checkout has pre-existing edits on main; leave them intact. The new
worktree starts at HEAD and does not contain those edits. Before execution,
record hashes of all inputs and reconcile relevant dirty source changes
explicitly rather than silently running a different revision.

- h01-population-accuracy-ledger.json and its generator: six terms, currently
  0.7178289854, 55/104, 1, 0, 1, 0. The two zero terms are currently hardcoded
  summaries of historical outcomes; eventual promotion needs new validated
  decision records, not a literal edit.
- h01-timestep-ladder.json: cell 7196644737 only, 10 ms, finest adjacent pair
  0.000625/0.0003125 ms differs by 0.933343 mV. No population qualification.
- h01-e-morphology/stage-1-decision.json: donor 4 spikes, H01 0 at 200 pA;
  effective onset capacitance 125 versus 785 pF. Cause of load difference open.
- h01_e_morphology_swap.py: positions scaled by (0.032, 0.032, 0.033) um;
  radii by 0.001; radius floor 0.05 um; soma collapsed and replaced by B3's
  soma radius; longest subtree labeled apical; downstream axon replaced by stub.
  This diagnostic conversion differs from datasets/_h01_swc.py, which preserves
  positive source radii and neutral labels. Audit both paths separately.
- h01-e-cable-load-manifest.json and h01-topographic-result.md already contain
  subsequent graded-load experiments. Stage 1's suggested runs are historical
  next steps and must not be blindly repeated.
- h01-donor-survey.md: remaining unmatched types and rejected donor reproduction
  outcomes require further work; old unsuccessful searches do not establish
  that new human data or fitting are impossible.

## First implementation: anatomy and biological-source audit

1. Reuse retained artifacts first. Inventory human sources, specimen identifiers,
   actual archive members, raw hashes, units, annotation meaning, mechanism
   provenance, and geometry transformations on both the diagnostic and deployed
   paths. Reconstruct the exact stage-1 input before interpreting its failure.
2. Fetch metadata and then only the necessary proofread mesh fragments for
   cell 955432427. Match the exact soma-bearing component and spatial domain
   used by stage 1; whole-cell mesh against partial cable is not a valid ratio.
3. Measure raw and converted tapered cable area, mesh triangle area, soma and
   neurite contributions, omitted components, radius-floor effects, and
   capacitance contributions. Record mesh resolution, seams, open boundaries,
   spine inclusion and compartment correspondence. Do not infer a physical
   correction solely from different area totals or effective capacitance.
4. Produce an audit decision naming any proven conversion defect or missing
   reference. Only an evidence-backed correction changes geometry. Reproduce
   any defect in a failing sibling test before fixing it. Preserve source
   anatomy; no radius, soma, axon or channel tuning just to restore spiking.
5. Apply verified fixes on the branch, then run bounded isolated transfer
   checks with direct observations before population qualification. Benchmark
   the actual executor, inspect live jobs, and register estimates before any
   expensive run. Do not duplicate a live campaign or silently buy compute.

Proposed modules: docs/evidence/h01_anatomy_area_audit.py and sibling
h01_anatomy_area_audit_test.py; provenance evidence in a new dated directory.
Production modules change only when the audit identifies a reproducible defect.
The audit will report lateral frustum area separately from a soma sphere, group
source edges by endpoint annotations, expose zero-length and label-crossing edges,
and never silently interpret all soma skeleton branches as soma membrane. Mesh
area is measured from physical vertices after source-format decoding, with exact
seam deduplication, duplicate-face and boundary-edge counts. A ratio is descriptive
and unavailable until the mesh component and anatomical domain are matched.
The acquisition runner will retain the mesh LOD, library versions, metadata,
fragment count and hashes. Coarse LODs may locate components but cannot qualify
membrane area without finer-level convergence. Initial retrieval is one cell,
with a byte cap checked from its manifest before downloading mesh payloads.
No new repeatedly executed model loop may use bare Python for/while; use the
brainstate transforms required by AGENTS.md and brainstate.random for RNG.

## Subsequent work toward the unchanged objective

### Confirmed conversion defect and repair contract

The first geometry measurement found that the historical diagnostic converter
confuses H01 annotation 1 (dendrite) with standard SWC 1 (soma), and H01 2
(astrocyte) with standard SWC 2 (axon). Its own reported length omits the newly
introduced soma connections. For cell 955432427 it creates about 91,056 um of
cable from a 3,343 um source component. This is an implementation defect, not
a measured human anatomy difference.

Repair the diagnostic export by retaining every source node, parent connection,
coordinate and positive radius, with a reversible node-ID map. Export neutral
SWC types and original H01 annotations in a sidecar, consistent with the
production importer. Do not replace the soma, floor radii, guess apical labels,
or replace the axon. Retire the old donor.json preparation path: this geometry
is not compatible with the donor runner's soma/axon reconstruction assumptions.
The corrected preparation writes anatomy evidence only, with runtime
compatibility stated. The existing BrainCell source-preserving importer is the
subsequent integration path. Preserve all historical inputs and decisions,
and issue a linked correction of the historical anatomy-failure interpretation.

Audit/replace unsupported mechanisms using human measurements, acquire or fit
human models for all unmatched types, and close each donor's recorded-response
errors with sealed independent validation. Transfer those qualified models onto
audited H01 geometry, qualify numerical precision across the final system, and
complete construction plus the full driven controls. Repeat only gates affected
by changed inputs. Update the ledger from traceable decision artifacts once
their scoped requirements pass; never mark completion while any item is missing.

## Verification and edge cases

Meaningful tests must cover anisotropic coordinate conversion, radius/diameter
confusion, tapered area against analytic shapes, repeated seam vertices,
degenerate triangles, fragment omission, disconnected components, open mesh
boundaries, soma collapse, invented apical labels, radius floors, axon replacement,
double-counted spine area, and missing human provenance. Missing correspondence
must return unavailable, not a passing ratio. Target >90% coverage on changed
modules. Runtime tests must reject partial, nonfinite or time-misaligned traces.
Numerical convergence cannot assert physiological correctness. Human holdouts
must not be used to select parameters. Inspect required direct-observation plots
and record their interpretation under docs/evidence/AGENTS.md.

## External primary sources verified

- H01 release: https://h01-release.storage.googleapis.com/data.html lists
  proofread_104 meshes, skeletons, subcompartments and the SWC archive.
- Allen model description:
  https://brain-map.org/our-research/computational-modeling/perisomatic-biophysical-single-neuron-models
  describes models built from individual neuronal recordings and morphology,
  with optimized somatic conductances and other stimuli used for testing.
  This description does not certify human provenance of every kinetic equation.

## Bounded post-import geometry check

Use the existing production H01Archive importer for cell 955432427, component 0,
and compare every directed source edge, coordinate and endpoint radius with the
actual BrainCell branches. Then lower the same morphology with CVPerBranch and
inspect every returned control-volume lateral area. Preserve the importer,
geometry-library and policy source hashes, all branch/CV observations, and the
release hash. This CPU-only static check has a 60-second process cap and executes
no membrane dynamics. Require all edges to match and total lateral area within
1e-8 relative tolerance. Passing this check qualifies only the recorded geometry
and discretization, not membrane completeness, physiology or the final population.

## All-104 production compartment geometry

Extend source-to-simulation verification to the exact 104-cell inventory and
component selection in the retained construction reference. Freeze its SHA256
against the recorded build decision. Use the deployed MaxCVLen(10 um) policy
with the same BoundaryAlignedCV electrical regions, without painting or running
any biological mechanism. Compare each region with the recorded intervals.
For every cell, verify every directed source edge, coordinate and radius against
the production import. Independently compute tapered lateral area and length
from each branch's source segments. Check every branch's CV intervals cover
[0,1] once without gaps/overlap, and its summed CV area and length agree within
1e-8 relative tolerance. Check per-cell and total compartment counts against
the pinned construction reference (808,495 overall). Do not equate aggregate
area agreement with per-branch preservation.

Retain every source/imported edge, branch and CV area/length/interval observation,
source identities, executed source hashes and terminal process receipt. Reject
changed component hashes, missing/duplicate cells, invalid geometry, partial
coverage and nonfinite values. Original acquisition coordinates remain the
datum. This stage measures static geometry on human anatomy only; its success
does not qualify channel kinetics, source-mesh completeness, physiological
transfer, or population timestep. The existing scores remain unchanged until
their full requirements have independent evidence.

Use one local CPU process capped at 600 seconds, retaining completed per-cell
artifacts on timeout without declaring 104-cell success. Historical import of
all 104 took 509 seconds; current optimized construction may be faster. No
multi-hour run or membrane rollout is authorized by this stage. Tests must
exercise per-branch mismatch even when global area cancels, holes/overlaps,
empty branches, nonfinite and nonpositive geometry, and reordered CV rows.

The first attempt terminated after 15 cells on a historical-region comparison
for cell 1830470325 (maximum normalized-boundary discrepancy 1.3103e-12 versus
the runner's 1e-12 comparison). Preserve that attempt and its terminal receipt.
A second prefix keeps the identical tolerance, records every current/reference
region interval and each discrepancy, and continues collecting all cells' geometry.
Report strict historical-region agreement separately from source-edge and per-CV
conservation. Any failed region comparison keeps the overall comparison failed;
successful process termination or matching total compartments does not override it.
