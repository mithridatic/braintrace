# Reassessment of the H01 validation path

## Required outcome

The original goal remains unchanged: active human-constrained H01 neurons,
a validated excitatory and inhibitory pair in BrainCell, and a small
circuit with verified individual synaptic responses and numerical stability.
A fitted donor reference is not validation of the exact H01 donor cell.

## Current evidence and missing completion evidence

| Requirement | Current evidence | Remaining completion evidence |
| --- | --- | --- |
| Human E reference | Allen cell 541563728; pinned input, voltage conventions, individual events | Full acceptance contract across calibration inputs; frozen reserved response test |
| E candidate | Five spikes; peak and recovery improvements; timing and waveform errors remain | One candidate satisfying all required direct observations together |
| Human I reference | Published putative-PV model and source recordings; BrainCell source transfer exists | Human response qualification across required inputs; latest diagnostic changes are not a validated production model |
| Numerical checks | Several individual candidates pass selected refinements | Correct candidate and input coverage for each promoted model |
| H01 transfer | Selected E/I profiles implemented with explicit region assumptions; I candidate fails positive response at tested input; source closing-time restoration restores it | Qualified intact I dynamics, full transfer and waveform checks, and justified electrical mapping |
| Small circuit | Implemented real H01 pair; E delivery verified; I delivery delays E with an explicit source closing-time override | Numerical robustness, reciprocal behavior, and use of physiologically qualified cell models |
| Causal explanation | Conditional intervention results and numerical limits recorded | Maintain only supported explanations; no implication of a unique human mechanism |

See the original [goal specification](2026-09-04-h01-active-circuit.md),
[E residuals](../evidence/h01-l2-kv3-ninety-ca133-result.json),
[PV implementation](../h01-pv.md), and
[PV residuals](../evidence/h01-pv-axon729-human-residuals.md).
No completed end-to-end physiological milestone follows from these records.

## Process failure and correction

Serial interventions optimized different observations in turn. Some changes
improved peaks or minima while worsening firing times. Repeating numerical
checks on each small improvement consumed time without establishing a
single complete acceptance decision. Local yes/no predictions answer causal
questions, but do not substitute for a cell-validation contract.

Stop new one-parameter tuning launches. Finish and audit the two already
running calcium-1.33 numerical checks. Keep their results even if they fail.
Do not lower the goal, relabel a reference as validated, or use a provisional
circuit as evidence that the validated-circuit requirement is complete.

## Required decision before another fitting campaign

The [event requirements](../evidence/h01-human-event-requirements.json) retain
all five E calibration events and all I events at each released active input
(12, 31, and 43). They explicitly block a physiological pass while allowances
and the remaining measurements are unresolved. The I 0.23 nA response retains
its prospective holdout status.

Complete one explicit acceptance table for each cell and required input.
For each individual response, name the measurement convention, datum,
allowed error, basis for that error, and pass/fail rule. Include missing
and extra spikes, each onset and phase, peaks, recovery voltage and delay,
and specified subthreshold sample voltages. Keep physiological allowances
separate from numerical comparison tolerances.

The E file has no repeated 250 pA trial. Its seven repeated 200 pA trials
cannot supply 250 pA spike-time uncertainty. Published optimizer weights
also cannot be relabeled biological standard deviations. If an allowance
is an engineering choice, label it as such and obtain agreement before
using it to declare success. An exact waveform approximation can be a
calibration target without being a claim of biological universality.

Keep sweep 53 reserved under its existing release condition: freeze the
candidate and calibration decisions before opening its response. A later
failure remains a validation failure; do not fit the reserved trace and
continue to call it held out.

## Bounded fitting design for review

Use simultaneous constraints across all calibration inputs. A shared
candidate must account for every required observation; a lower pooled
score must not conceal a failed event. Retain each raw residual and
unmatched event. Use normalized residuals only to guide search, not to
replace acceptance. Pin parameter bounds to sources or explicit inferred
assumptions. Keep the current candidate and source model as frozen controls.

Before expensive execution, benchmark one existing complete driver call
and estimate total cost. A proposed first campaign is capped at 24 new
full-cell candidate evaluations, including all calibration inputs per
candidate, with at most two concurrent simulations. Present total expected
runtime before approval. Do not begin this campaign under the old serial
split-test plan. If no candidate passes at the cap, report failure and
reassess the model or data; do not append another unbounded tuning series.

Use coarse numerical settings for candidate search only after showing
they preserve decisions over the relevant parameter region. Recheck
selected finalists at the declared fine settings. A comparison on one
candidate does not validate a coarse solver for all parameter choices.

After a candidate passes the complete calibration contract, freeze it,
perform its numerical and reserved-input checks, and transfer it to
BrainCell with independent reference comparisons. Repeat the cell gate
on the transferred implementation and H01 morphology. Then implement
and verify the small E/I circuit under the original specification.
These are dependency milestones, not a guarantee that the current model
family can meet the full physiological target.

## Immediate next work

Audit the two existing runs. Assemble the acceptance table from current
source records and explicitly flag unsupported tolerances. Present the
bounded fitting design and missing decisions before new implementation
or expensive fitting. Keep the goal active and report milestone completion
separately from diagnostic activity.
