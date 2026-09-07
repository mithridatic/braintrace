# H01 bounded E continuation (2026-09-07)

Authorized by the user's request to continue remaining H01 work autonomously.
Use the existing campaign and scoring tools on feat/h01-braincell.

## Scope and gate

Three evaluations maximum, one at a time, 1500 seconds per evaluation:
1. B3: prior B2 flags with SK:soma:0.35 at sweep 53 (310 pA).
2. B3 calibration: exactly the same flags at sweep 50 (250 pA).
3. P2: freeze those flags, register the sweep 55 prediction before export,
   then run sweep 55 once. No fitting after opening.

The extra calibration run is necessary: B2 was evaluated only at sweep 53;
its sweep 50 behavior cannot be inferred from the earlier initiation survivor.
Stop on failed execution, block, or loss of axon-first initiation. A usable
failure is retained as a failure, with no additional parameter search this run.
No BrainCell profile promotion follows without a separate transfer gate.

## Predictions before evaluation

B3 sweep 53: 10 spikes, late cycles (cycles 4 onward) 115 to 130 ms;
threshold -57 +/- 1 mV; first peak 37 to 40 mV; first trough -70 +/- 1 mV;
first width 0.906 to 1.106 ms; axon crosses -20 mV before soma.
Reject the mechanism prediction if late mean exceeds 175 ms, depolarization
block occurs, or any listed initiation band breaks.
B3 sweep 50: 5 to 7 spikes, late cycles 180 to 300 ms; same initiation bands.
These are prediction bands, not relaxed acceptance thresholds.

Score existing usable limits row by row, retaining input and cycle. AHP remains
unresolvable if human repeat spread exceeds half its limit. Counts are exact
under the original contract. Partial width/AHP/count columns do not establish
full event-level contract acceptance. Use the same exact profile at both inputs.

## Evidence corrections

B2's decision JSON records 8 of 9 width rows passing; the prose claim that every
usable row except rate passes is unsupported. Its 250 pA claims refer to an
earlier profile. The population page also calls a stopped scan running.
Avoid repeating these errors by deriving summaries from per-input row verdicts
and identifying the exact candidate flags used for each result.

## Driver release-path repair

The first P2 launch returned argparse exit 2 before integration: the exporter
supports a registered sweep55 release, but the driver still permits only
43/50/53. The waveform has been exported after prediction registration; the
failed launch is not a model evaluation and is retained in the campaign log.

Repair only the driver input guard: include 55 in supported choices and apply
h01_l2_sweep_export.sealed_reason after CLI/candidate defaults resolve, before
source files are read. The driver must still reject 55 without its prediction,
including when supplied through candidate JSON; unknown sweeps stay rejected.
Use sibling driver tests to reproduce registered-55 rejection before the fix,
then test sealed CLI/default paths and old input acceptance. No numerical model,
parameter, input array, or solver changes. Retry the same frozen P2 prediction.

Prevention: future holdout release checks must test exporter and simulator
entry points together before opening the source waveform.

The seal reuse initially exposed a container dependency boundary: h5py is installed
on the export host but absent from the NEURON image. A regression test reproduced
that import failure; h5py now imports only inside export_sweep. Driver --help was
verified in the actual pinned image before the successful P2 retry. Both failed
launches exited before integration and remain in the campaign log. Future driver
changes must include this lightweight actual-image startup check.
