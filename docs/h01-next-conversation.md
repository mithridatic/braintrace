# H01 next conversation: unchanged scores; rejected experiments are closed

## Start here

The campaign did not achieve the requested result. All six headline values are
unchanged. Preserve the evidence and useful corrections, but do not treat the
fitting history as a recipe to repeat. The goal was paused at handoff; this
document and repository synchronization do not authorize restarting computation.

First read this document and the linked acceptance and causal records. Spend at
most ten minutes reconciling the existing acceptance contract and current evidence
before proposing the next bounded decision. Do not launch another PAX6 fitting
campaign, change the score, or open reserved recordings as a startup action.

Research snapshot: `2c31c6a4`, branch `campaign/h01-human-unity-20260914`, based on
`23ed2421`. The user requested integration into `main`, push, Vast.ai checkout
synchronization, and removal of the stale worktree. Verify actual Git state;
do not assume the old `.worktrees/h01-human-unity` path still exists. The local
repository is `C:\Users\J\Documents\Projects\connectome-agent\braintrace-source`.
Create a fresh worktree for any subsequently approved implementation.

Integration checkpoint: `3f4d5c98` was pushed to `origin/main` and fast-forwarded
on Vast.ai instance 50616476 (`braintrace-gpu`), `/workspace/braintrace`, branch
`main`. The obsolete remote `/workspace/braintrace-q2` worktree was removed after
preserving all 133 untracked/ignored files (1,085,412,882 bytes), with before/after
SHA-256 verification, under `/workspace/braintrace/.cache/h01-q2-retained-20260914`.
The local worktree cache was moved intact to `.cache/h01-human-unity-retained`;
all six cached NWB hashes were checked. Historical `.cache/human-source-followup`
paths from this campaign now resolve beneath that retained cache directory.
The unrelated dirty `docs/evidence/h01-arc-manifest/manifest.json` remains intact
locally and remotely. The 91 affected helper tests passed again on integrated
main. No new physiological experiment was launched during integration.

## What changed, and what did not

The original objective remains **all six values at 1.000, true human only, all
104 cells**. No new donor, physiological anatomy transfer, or population driven
window qualified. Helper test coverage and better training residuals do not
change these results.

| Term | Before and now | Evidence scope |
| --- | ---: | --- |
| donor_accuracy | 0.7178289854126294 | B3's ten-element comparison to its own recording; donor for 28/104 cells |
| type_coverage | 55/104 = 0.5288461538461539 | Layer/class matches, not validated physiology |
| construction | 1.000 | Historical synthetic forward, 104 cells, 808,495 compartments |
| driven_window | 0.000 | No qualified 104-cell physiological window |
| timestep | 1.000 | Isolated-cell 10 ms qualification; not final population convergence |
| anatomy_transfer | 0.000 | Corrected anatomy has no qualified physiological transfer reading |

The old anatomy failure used a malformed conversion. The repaired export retains
11,524 nodes and 11,523 edges. This corrects the interpretation of the old negative
reading; it is not a positive physiological result. Construction and timestep
ones do not qualify the final human-only population.

Read the [ledger](evidence/h01-population-accuracy-ledger.json),
[original goal contract](specs/2026-09-14-h01-human-unity.md), and
[six-hour worktree assessment](evidence/h01-human-unity-20260914/worktree-assessment.md).
There are no paired voltage recordings for the actual H01 cells. Validation
against other human cells constrains a transfer prediction, not measured H01
physiology.

## Reconcile existing acceptance; do not invent a replacement

An [approved recorded-response contract](specs/2026-09-05-h01-recorded-response-acceptance-proposal.md)
already exists. Its validation tier requires exact event counts, 1 ms timing,
1 mV voltage, and 0.05 ms spike-phase errors at the specified observations.
These are engineering limits, not measured biological standard deviations.
Use original clocks, no trace alignment, and every event/input separately.
The separate usable tier is not a validation-grade pass.

The legacy ten-element score in
[h01_topographic_swap.py](evidence/h01_topographic_swap.py) averages
`max(0, 1 - abs(model - recorded) / scale)`. It has no tolerance plateau:
an exact unrounded 1 requires zero discrepancy in the scored quantities.
It is not a probability of human fidelity. Preserve its historical value.
Any change to the six-term goal or acceptance requires explicit user agreement;
do not silently turn 0.718 into a pass by redefining the metric.

The older contract calls E sweep 53 unopened. It has since been exposed in the
B3 comparison. Reconcile exposure receipts before using historical holdout
labels. Existing E repeats 56–62 are at 200 pA, include two no-spike records,
and do not establish 310 pA train-adaptation uncertainty. Reuse the existing
[contract scorecard](evidence/h01-contract-scorecard.md) and repeatability work.

## Proposed next scientific decision

Start from [causal model Y4](h01-causal-model.md), not from another general source
search or new framework. B3's first spike rises at about 570.4 V/s versus human
347.7 V/s. Count is 10 versus 10 at 310 pA, but counts at 200/250/350 pA are
4/8/12 versus 1/5/13. Its threshold climb is also missing. Recording-chain
correction does not explain the rise discrepancy. Existing analysis locates
excess rise above roughly -40 mV and implicates the soma sodium contribution;
it does not uniquely identify a biological mechanism.

1. Reconcile the approved direct-response limits, legacy score, human-only
   provenance, and available repeats. Report the precise unresolved decision.
2. If further experiments are authorized, select one source-supported whole-cell
   hypothesis targeting this measured failure. Reuse existing BrainCell/NEURON
   equivalence and failed intervention evidence.
3. Before execution, write the predicted waveform change, protected behaviours,
   falsifying result, parameter/source justification, and cumulative runtime and
   cost cap. Follow the repository spec and code-approval rules.
4. Evaluate the complete waveform and train across released inputs. Slowing the
   first rise by losing spikes or worsening adaptation is a failed intervention.
   Freeze a qualified candidate and predictions before opening a valid holdout.
5. Only a qualified donor can advance the population objective. Final anatomy,
   provenance, construction, timestep, and driven controls still need their own
   registered gates; a short diagnostic cannot substitute for them.

Proposed campaign control, not an already approved acceptance revision: a fixed
cumulative time/token budget and review after two failed experiments. Prior
bounded runs still accumulated into an expensive, unsuccessful campaign.
Ask before estimated >15-minute or multi-hour runs, as previously requested.

## Preserve these results; do not repeat the rejected campaign

- `e4ec8dcd`: the source-recorded four-pole 2 kHz observation filter reduced
  excluded early-control error from 5.6803 to 1.97998 pA (65.1%). This is an
  accepted conditional recording correction, not a qualified human channel.
- `3976d700`: active-current fit through that frozen observer reached 10.5516 pA
  full-sample RMS, but reversal potential hit its bound; candidate rejected.
- `2c31c6a4`: independent relaxation-voltage fit lowered the same training RMS
  to 8.27016 pA (21.6% reduction), but four parameters hit bounds and some recovery
  records worsened. Candidate rejected. Local sensitivity found near-collinear
  `vh`/`kh` effects (cosine 0.9999999818); this is local weak distinguishability,
  not a global identifiability proof. Ninety-one helper tests passed; no donor
  qualified. Do not add parameters, widen bounds, or rerun this fit unchanged.

See the [human-data record](evidence/h01-human-unity-20260914/human-data/README.md),
[latest fit specification](specs/2026-09-14-h01-pax6-independent-relaxation.md), and
[exposure receipt r14](evidence/h01-human-unity-20260914/human-data/fitting-inputs-r14.json).
The observer, raw exports, anatomy repair, provenance corrections, and sealed
failure records are useful. Most campaign changes are diagnostic evidence;
the production-package change since the baseline is a provenance docstring fix.

Also closed: B3 Toronto sodium swap; HL23PYR replacement (20 spikes versus 10);
fast axon recovery scaling (train collapse); PAX6 initial-state-only repair and
uniform gain; sodium two-scalar thermal retuning; negligible global sodium
recovery opportunity; failed strength-only second-synapse transfer. The SP16
slow gate has already been implemented/tested historically. Do not reintroduce
animal sodium kinetics or pooled mouse/human potassium as human-only inputs.
Do not tune corrected anatomy merely to restore spiking.

## Protect independent evidence

- PAX6 calibration session 840043481: sustained sweeps 105, 106, 107, 109, 110,
  111 remain reserved. Sweeps 104, 108, 112 were exposed. Consult receipts for
  every other sweep rather than assuming independence from a filename.
- External PAX6 session 835648738 response arrays remain unopened.
- Whole-cell PAX6 session 811953264: passing holdouts 13, 15, 28–34 and reserved
  short-pulse/ramp data must retain their access restrictions.
- Preserve original cached NWB files and sealed evidence during worktree cleanup.
  Raw voltage-clamp data are total amplifier current, and DAC command is not
  measured patch voltage.
- The local human-channel data request is unsent. No authorization exists to
  send email or messages to the authors.

## Required reporting and lesson

Report baseline → current qualified values, the exact new observation, the gate
it passes or fails, cumulative spending/runtime, and the next decision. If no
qualified score moved, say so first. Never equate commits, helper tests, or a
lower fitting objective with physiological progress.

The execution failure was continuing a narrow fitting campaign without a
campaign-wide stop decision while the requested whole-cell/population measures
stayed unchanged. Reuse the causal diagnosis and acceptance work to avoid that
mistake. Guidance may have contributed; no API or model-configuration defect was
established. A prompting change is not evidence that the biology is solved.
