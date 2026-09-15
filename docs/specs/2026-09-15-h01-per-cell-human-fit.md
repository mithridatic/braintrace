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
- NEURON on the box's 255 cores (`/workspace/h01-neuron/venv312`, NEURON 9.0.2):
  an Allen-style perisomatic evaluation of a ~3,000-compartment cell for 2 s at
  dt 0.025 is a few seconds on one core. 104 cells x 4 inputs x 2,000
  evaluations at 4 s over 255 cores is about 4 hours of wall time. This is the
  viable path, and it needs one new component: an H01 -> NEURON importer that
  reproduces the production electrical partition (soma / axon / dend regions from
  `_regions`) so the fitted densities can be painted back through the BrainCell
  builder without a second interpretation of the anatomy. The corrected
  source-preserving SWC export (`h01-human-unity-20260914/source-preserved/`)
  plus the region intervals retained in `population-geometry-r2/*.npz` are the
  inputs; no soma replacement, radius floor or axon stub.

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
