# H01 diagnostic evidence

These instructions apply when producing or interpreting H01 causal evidence here.
Read `../specs/2026-09-11-h01-direct-observation-contract.md` and the response
explanation being investigated in `../h01-causal-model.md` before a diagnostic run.

- Start from the observed response and its spatial/temporal boundary. Use a few
  informative cycles, preserving the direct samples within them. Do not substitute
  a mean, percentile, firing rate, count, or current share for the behavior to explain.
- The E-cell SP11/SP12 scorer CLIs produce direct-observation bundles by default.
  For other retained L2 runs use `python -m docs.evidence.h01_direct_report --run
  <run-prefix> --human <human-sweep.npz> --output <new-output-prefix>`, repeating
  `--run` and `--human` as needed. This adapter uses the L2 1020-2020 ms pulse clock;
  do not apply it to a different clock without an explicit adapter.
- Inspect the generated multivari, trajectory, and conjugate-pair PNGs with an
  image-viewing tool. Record the observed within-cycle and between-cycle patterns,
  missing phases, and alternative explanations in a linked visual-review record.
  Successful rendering or a QC pass does not count as visual interpretation.
- Design multivari views around physical nesting and common scales, following
  Hartshorne's Figures 101 and 104 (local book scans p108 and p111; interpretation
  on p109). Preserve individual points, event identity, and the datum. Reduce
  sampling only after inspecting which patterns repeat; three samples are a
  starting heuristic, not universal statistical sufficiency.
- Pair voltage with signed current at the same boundary and time, and retain
  voltage/charge for storage. Keep axial branches separate. Local current is not
  whole-soma current, command current is not human ionic current, and membrane
  voltage times axial current is not branch dissipation.
- Require full time-window coverage before computing a window result. Never let
  interpolation extend the last sample into missing time. Distinguish adaptive
  sample percentiles from time-weighted summaries. Missing evidence is unavailable,
  never a pass or a fabricated zero.
- A causal claim needs the conditions, mechanism, observed response, discriminating
  intervention, and a contradicting outcome. Baseline current magnitude does not
  bound its intervention effect in a feedback system. Failed doses do not exclude
  every mechanism in a family. Report human qualification separately.
- Preserve historical registered decisions and doses. Write reanalysis to a new
  prefix with source hashes, and identify changed interpretations explicitly. Check
  active Vast processes before launching a run; do not duplicate a running campaign.
  Reanalyze adequate retained traces first, and obtain a bounded new run when a
  required observation is absent. State whether work used CPU or GPU on that host.
