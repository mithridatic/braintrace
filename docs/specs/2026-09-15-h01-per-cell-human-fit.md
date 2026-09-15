# Per-cell human-constrained fits on H01 anatomy (proposed next stage)

Status: proposed, not executed. Follows the measured stage-C result of
[driven window and anatomy transfer](2026-09-15-h01-driven-window-and-anatomy-transfer.md):
donor densities transferred as densities onto retained H01 anatomy produce
depolarisation block (L2, PV) or spontaneous firing (SST) for three of four
deployed donors, and scaling the dendritic passive load 3x and 10x does not
remove the block. Type-matched density transfer is therefore not the route to
all 104 cells; more donors imported for that purpose would fail the same way.

## What changes

Each of the 104 H01 cells gets its own perisomatic fit on its own anatomy
(the production import, unchanged), with the human recording of its type as
the target. Human inputs only: Allen human cells (43 perisomatic fits exist;
their recordings are public) and the Toronto human L2/3 and L5 recordings
already retained. Channel equations stay the existing set until a human kinetic
source replaces one; the fit adjusts densities, not equations. This keeps the
six-term contract: donor_accuracy becomes a per-cell score against the
type-matched human recording, type_coverage counts cells whose target recording
matches layer and class, and anatomy_transfer is retired into the per-cell score
because the fit is made on the deployed anatomy.

## Feasibility, measured

- BrainCell/JAX single-cell evaluation on Vast: 2.1 s of simulation at dt 0.005
  on a 3,923-CV cell took 185 s alone-ish and 390 s under GPU contention
  (`transfer-e310`, `split-e310-dendload-x3`). A fit of ~1,000 evaluations x 4
  inputs is 200+ GPU-hours per cell. Not viable on this path without batching
  parameter sets inside one program (untested).
- NEURON on the box's 255 cores (`/workspace/h01-neuron/venv312`, NEURON 9.0.2),
  measured 2026-09-15: `Import3d_SWC_read` loads the corrected source-preserving
  export `h01-human-unity-20260914/source-preserved/h01-955432427.swc` directly,
  4,307.5 um² and 3,343.1 um exactly as the source, but as 11,449 sections (one
  per sample, because the neutral type 0 gives Import3d no branch merging), and a
  passive 2 s run at dt 0.025 takes 20.2 s on one core. Merging unbranched
  samples into the 2,569 source branches (the same branch set the production
  importer uses; `population-geometry-r2/955432427.npz`) should bring that to a
  few seconds; that merge is the one new importer component, and it must
  reproduce the production `_regions` partition from the anatomy sidecar so the
  fitted densities paint back through the BrainCell builder with no second
  interpretation of the anatomy. At 5 s per evaluation, 104 cells x 4 inputs x
  2,000 evaluations over 255 cores is about 5 hours of wall time; at the
  unmerged 20 s it is about 22 hours. Both are within reach; neither is
  authorised by this note. The Allen mod set is already compiled in
  `/workspace/h01-neuron/cache/human-pyramidal-l2/kv3-closing-source`.

## Registered acceptance before any fit runs

- Features per cell: rheobase, spike count at each recorded input, first-spike
  latency, AP threshold, peak, width, AHP, adaptation ratio, resting level,
  input resistance, from the type-matched human recording; tolerances are the
  recording's own repeat spread where repeats exist, else the existing usable-tier
  limits. Holdout: one recorded input per cell never used in the objective.
- Pass per cell: every feature inside its tolerance on the calibration inputs and
  the count rule held on the holdout input. The per-cell score is the fraction of
  features held; the population value is the fraction of cells passing.
- Falsifier for the whole approach: if fewer than half of a 10-cell pilot (two
  per layer class) reach the count rule on their holdout within the evaluation
  cap, stop and report; do not widen tolerances.
- Caps: pilot 10 cells x 2,000 evaluations, 2 hours wall on the box; full 104
  only after the pilot passes and its cost is measured.

## What this does not do

It does not measure the H01 donor's own physiology (no recordings exist), it
does not make rodent-lineage channel equations human, and it does not touch the
driven-window gates, which stand on the population as built.
