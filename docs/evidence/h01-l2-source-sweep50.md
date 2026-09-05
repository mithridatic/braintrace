# Layer-2 source-model response

The retrieved Allen fit runs with all eleven source mechanisms compiled
without edits under NEURON 9.0.2. The first run uses the source command
waveform and a 0.005 ms fixed step. It produces seven complete events
above -20 mV during the main pulse. The human recording has five.

The model first crossing is 4.582202 ms early. Its first peak is
1.376236 mV higher than the human sampled peak. Its first duration
above -20 mV is 0.020633 ms shorter. Later ordinal times diverge more:
the fifth model event occurs 295.906518 ms before the fifth human event.
These are ordinal comparisons, not proof that each pair has the same
underlying cause. Two model events have no human ordinal partner.

The baseline model voltage is -84.077718 mV, compared with human
-84.232297 mV. Baseline agreement does not establish agreement of
the spike sequence. The [direct record](h01-l2-source-sweep50-human-comparison.json)
retains all crossings, peaks, durations, and unmatched events.

## Setup and input checks

The driver reads back each applied genome parameter. It saves section
properties, solver settings, voltage, and applied current. The morphology
importer warns of multiple trees with roots at source lines 4, 10245,
10303, 10474, and 10589. No attachment was invented to remove this warning.
The instantiated model has 91 sections after the source axon replacement.
The [raw component audit](h01-l2-source-morphology-components.json) identifies
four detached axon-only components. The
[NEURON topology audit](h01-l2-source-topology-audit.json) finds a single
soma root and verifies that all 91 final sections reach it. The section
names match the saved simulation setup. The importer also exposes only
one root before the axon replacement. Thus, the raw-file warning does
not establish a disconnected simulated cell. The earlier qualification
note conflated source components with instantiated sections. Future
topology claims must inspect the realized section graph. The replacement
axon remains a model assumption rather than measured axonal anatomy.

The first input check incorrectly assumed that only the main pulse had
transitions. It failed at the two edges of an earlier 50 pA test pulse.
The [failed check](h01-l2-source-input-check-initial.json) is retained.
The complete command changes at 5, 15.02, 1020, and 2020 ms. Recorded
current changes one model step after each command edge. Away from all
four edges, recorded current equals command current. This corrected
account is diagnostic evidence, not a prospectively declared pass.
Future input checks must derive edges from the full waveform, not assume
that the principal stimulus is the only stimulus.

## Limits

The source fit lists sweep 51, while this comparison uses sweep 50.
This run has no added bias. A bias intervention must remain separate.
The source model has not passed physiological validation, and its
spatial discretization remains unqualified. No source parameter has been
tuned to this result.

The declared 0.005 to 0.0025 ms comparison rejects temporal qualification.
Both runs have seven events with the same peak-sign class. The largest
onset change is 0.330878 ms, above the 0.1 ms limit. Peak changes are
at most 0.011807 mV and duration changes at most 0.000183 ms, within
their limits. The first onset changes by only -0.006536 ms; later event
times accumulate larger changes. See the
[complete temporal comparison](h01-l2-source-time-comparison.json).
This refinement does not remove the seven-versus-five event mismatch.

The next halving, from 0.0025 to 0.00125 ms, also rejects qualification:
maximum onset change is 0.162563 ms. All seven events remain. Maximum
peak change is 0.005809 mV and duration change is 0.000081582 ms.
The [repeatable audit](h01_l2_time_audit.py) reproduces the first failure
and records the [second comparison](h01-l2-source-time-comparison-dt00125.json).
Its 17 direct-trace tests pass with 100% statement coverage. These tests
cover shifted onsets, peaks, durations, counts, signs, missing events,
invalid traces, changed metadata, and an invalid refinement step.
The 0.00125 to 0.000625 ms comparison satisfies the declared temporal
limits. Maximum onset change is 0.080492 ms, peak change 0.003146 mV,
and duration change 0.000037905 ms. Both runs retain seven events.
The [complete comparison](h01-l2-source-time-comparison-dt000625.json)
qualifies this successive time refinement only. Spatial convergence
and physiological agreement remain incomplete.
