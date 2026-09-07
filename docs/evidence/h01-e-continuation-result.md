# H01 E bounded continuation: completed, candidate not accepted

2026-09-07, `feat/h01-braincell`. Three successful donor-NEURON evaluations,
using one frozen B3 profile: somatic sodium factor 0.9, axonal NaTs 3.814 S/cm2,
somatic SK factor 0.35, calcium decay factor 1.0; other B2 settings unchanged.
This is donor-geometry evidence, not a BrainCell or H01-anatomy transfer pass.

## Same profile at all three inputs

| Input | Human/model spikes | Human/model rate (Hz) | Usable rate | Usable adaptation | Failed paired width cycles |
| --- | --- | --- | --- | --- | --- |
| 250 pA, sweep50 | 5 / 8 | 4.790 / 7.728 | FAIL | pass | none among 5 paired cycles; 3 extra model events |
| 310 pA, sweep53 | 10 / 10 | 10.002 / 10.045 | pass | pass | 2 |
| 350 pA, sweep55 prediction | 13 / 12 | 13.269 / 11.634 | pass | pass | 3; human event13 unmatched |

[Per-cycle tables and verdicts](h01-e-usable/b3-all-inputs-usable.md),
[machine-readable rows](h01-e-usable/b3-all-inputs-usable.json).
Rate is the existing mean-full-cycle rate, not count divided by pulse duration.
Every AHP row is unresolvable under the usable tier because the human repeat
spread is 1.0106 mV against half the 2 mV limit. This is not an AHP pass; many
troughs also fail the original 1 mV contract.

The dose fixes the 310 pA count/rate but overfires at 250 pA. It is not a shared
usable E profile. At 310 pA, cycle2 width is 1.01146 vs 1.26903 ms: error
0.25757 exceeds the per-cycle 0.25381 ms allowance. At 350 pA, cycle3 width is
0.99629 vs 1.42961 ms: error 0.43332 exceeds 0.28592 ms. No new mesh/tolerance
comparison was run, so these physiological comparisons are not numerically
qualified. The fixed profile is not promoted.

## Registered predictions

B3 sweep53: predicted 10 spikes, observed 10. Cycles4 onward average 122.587 ms,
but the individual 115-130 ms band was missed by cycle5 (134.700 ms).
B3 sweep50: predicted 5-7 spikes, observed 8; late-cycle 180-300 ms band also
missed. Axon-first initiation and the registered first-spike bands held at both.
[Calibration decision](h01-e-usable/stage-b3-decision.json).

Sweep55 prediction was committed at `306459e` before export and run. All ten
registered model-output bands held, including 11-14 predicted spikes (12),
10-14 Hz (11.634), and cycles4 onward 80-125 ms (80.157-121.528).
These are model predictions, not human acceptance tolerances. Human count is 13,
and the original event contract fails. Count metadata was previously seen and
was never claimed blind; waveform/timing/peak/cycle data were opened only after
registration. Sweep55 is now spent. No parameters changed after opening.
[Prediction](h01-prediction-e2.json), [decision](h01-e-usable/stage-p2-decision.json).

Original event-contract rows, including missing events, are retained for
[sweep50](h01-e-usable/b3-sweep50-event-contract.json),
[sweep53](h01-e-usable/b3-sweep53-event-contract.json), and
[sweep55](h01-e-usable/b3-sweep55-event-contract.json). These files score count,
crossings, peaks, phases and derived intervals; they do not establish complete
recovery/subthreshold coverage. The first peak at 310 pA is 3.166 mV too high,
and its third rising crossing is 36.250 ms early.

## Repairs and verification

- The driver rejected registered sweep55 while the exporter accepted it. A
  reproducing test preceded the fix; the driver now applies the shared seal
  check after CLI/candidate defaults, before source loading. Candidate JSON
  cannot bypass it. The export-only h5py dependency is imported lazily so the
  NEURON image can use the seal check. Actual-image startup and the full P2 run
  verified the repaired path. Two failed startup attempts integrated no model.
- The usable scorer incorrectly applied 20 percent of an input's mean width to
  every cycle. A regression test reproduced that error; width limits and
  resolvability now use each matched human cycle. Baseline I/E and B2/B3 tables
  were regenerated. This changes sweep55 cycle2 from fail to pass; cycle3 fails.
- 98 targeted tests passed, covering driver guards, exporter, campaign runner,
  initiation, usable and event-contract scoring. Exporter and usable scorer
  each have 100 percent line coverage. File-backed scoring tests retain extra
  events and empty repeat sweeps. No full repository suite was run.
- Runs completed in 649, 622 and 665 seconds. All saved traces are finite,
  strictly time-ordered, and reach 2100 ms. Input and frozen candidate hashes,
  nseg factor9 and CVode tolerance1e-10 match all three reports.
  [Artifact hashes](h01-e-continuation-artifacts.json). Raw/derived NPZ traces
  remain local; JSON evidence and reports are versioned.

## Remaining H01 work

The E input-response relation and early-spike widths need a new bounded causal
hypothesis; another dose fitted only at 310 pA is not supported. Before any
promotion, the same profile must satisfy all required inputs, numerical
refinement, and BrainCell/H01 transfer gates. Sweep43 subthreshold qualification
was not rerun here.

The [I initiation audit](h01-i-initiation-audit.md) finds axon-first crossing
already present at all three inputs; the missing burst cannot be attributed
simply to its absence. I reserve and Noise1 holdout remain unused. Pair evidence
is diagnostic and depolarising in the tested state, with no E spikes in either
control; functional spike inhibition remains unverified. Per-type donor gaps,
fragmented anatomy, the stopped connectivity rescan, population rollouts beyond
one step, and the Example21 adapter also remain open. See
[population status](h01-population-status.md).
