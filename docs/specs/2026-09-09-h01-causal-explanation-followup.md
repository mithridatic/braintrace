# H01 causal explanation follow-up

## Purpose

Connect the Y3 and Y4 response errors to measured states and currents.
Use the book's sequence: response difference, physical path, cause test, and prediction.

## Scope

Analyze existing traces and decisions. Update the causal document and its evidence.
Keep the recorded candidate, input, geometry, solver, and acceptance limits explicit.
Do not change model code, fit parameters, or open new human data.
Do not count observations as interventions.

## Existing-data analysis

For I, use the recorded finalist at 0.19 and 0.23 nA.
The latter input has already been evaluated; it is not a new validation sample.
Inspect the first two troughs and the penultimate trough, each at an offset of 6 ms.
Exclude any sample that reaches the next spike's threshold.
Use the existing landmark and current-conversion functions.
Calculate axial current from each recorded neighbor voltage and resistance.
Retain all current terms, sodium availability, axonal calcium, and the axonal SK gate.
Record file hashes and candidate metadata with the results.

Check finite arrays, increasing time, and the sampled current balance.
The balance residual checks consistency. It does not bound solution error.
Do not apply a local soma budget to the whole cell.

For E, inspect B3 trace fields and candidate metadata.
Check whether the earlier current study used the same parameters.
Use only matched evidence to support a B3 mechanism.

## Missing-link tests

Specify the observed site, fixed conditions, prediction, and rejection condition for each open link.
Keep these tests separate from completed measurements.
A later intervention needs a fixed dose, numerical controls, and a run limit before execution.

## Completion checks

Each causal claim must identify its evidence and conditions.
All local document links must resolve.
Use short sentences and consistent technical terms.
Check whitespace and preserve unrelated worktree changes.
