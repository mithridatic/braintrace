# H01 next conversation

## State on 2026-09-15

Objective unchanged: all six population terms at 1.000, true human only, all 104
cells. Branch `campaign/h01-driven-window-20260914` (local worktree
`.worktrees/h01-driven-window`, Vast checkout `/workspace/braintrace`).
Stage record: `docs/evidence/h01-driven-window-20260915/README.md`.

| Term | Value | What it rests on |
| --- | ---: | --- |
| donor_accuracy | 0.718 | B3 ten-element score against its own recording; donor for 28/104 cells. The score has no tolerance plateau, so an unrounded 1.000 needs zero discrepancy; the recording has no 310 pA repeat to measure its own ceiling. Unchanged. |
| type_coverage | 55/104 | layer-and-class donor matches. Unchanged. |
| construction | 1.000 | 104 cells, 808,495 compartments, now also built and gated on Vast. |
| driven_window | DRIVEN_WINDOW_VALUE | four 50 ms controls + refinement pair on Vast through the existing gates. |
| timestep | 1.000 | isolated-cell ladder; population refinement pair adds the 104-cell reading. |
| anatomy_transfer | 0.250 measured | 1 of 4 deployed donors holds its human count on retained H01 anatomy (L4). L2 and PV block, SST fires spontaneously. |

## What was learned that changes the next move

1. **The population runs on Vast.** Every earlier "driven window" failure was
   the Windows laptop. On the 4090: construction 169 s, init 177 s, compile
   ~370 s, 62 ms/step at 808,495 compartments, 1.6 GB GPU. The container's
   cgroup pids limit (7,680, read-only) allows at most two population processes
   at once; `var/h01-driven/queue.sh` enforces it.
2. **Density transfer onto H01 skeleton anatomy fails for three of four donors**
   (causal model Y7). Tripling and tenfold dendritic passive load do not remove
   the L2 block, so the condition is the somatic profile on a 107-315 um² soma
   region with 0.4 um-diameter processes, not the missing dendrites. Importing
   more donors for density transfer (43 Allen human fits exist, inventoried in
   `allen-human-perisomatic-models.json`) would meet the same condition.
3. **Per-cell fits on H01 anatomy are feasible on the box.** NEURON imports the
   corrected source-preserving SWC exactly (4,307.5 um², 3,343.1 um) and a passive
   2 s evaluation takes 20 s unmerged; with sample merging into the 2,569 source
   branches, about 5 h of wall time for 104 cells x 4 inputs x 2,000 evaluations
   on 255 cores. Proposal, with registered acceptance and a 10-cell pilot
   falsifier: `docs/specs/2026-09-15-h01-per-cell-human-fit.md`.

## Next decision

Run the 10-cell per-cell fit pilot under the registered acceptance. Its
deliverable is per-cell scores against type-matched human recordings on the
deployed anatomy; if fewer than half reach the count rule on their holdout, the
approach stops there and the report says so. Do not import more donors for
density transfer and do not tune anatomy to restore spiking.

## Operating rules that held

- Runs on Vast, gates and plots local. Launchers `var/h01-driven/run_h01.sh` and
  `run_transfer.sh` write receipts; every run has launch/terminal JSON.
- Register plans and predictions before launching; report the falsified ones
  (stage C's prediction was falsified and is recorded as such).
- Two population processes at a time; check `ps -eo nlwp | awk '{s+=$1}'` before
  launching anything.
- Gate Vast runs against the Vast build; the output-site tie-break fix
  (`_nearest_cv`) is on this branch and a future rebuild re-pins 17 cells' sites.
