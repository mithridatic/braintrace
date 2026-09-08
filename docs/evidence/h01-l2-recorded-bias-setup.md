# Recorded-bias intervention setup

The [declared split](../specs/2026-09-05-human-l2-bias-split.md) changes
only the input by adding the recorded -0.003711858945210089 nA bias.
The driver retains command-only input by default and records the selected
mode and added bias. It uses the same complete waveform, including the
early test pulse and pre-pulse history.

The input helper has tests for unchanged default commands, signed bias
addition at every sample, no mutation of source samples, malformed arrays,
and nonfinite bias. Adding this option exposed a temporal-audit gap: the
audit compared the input label but did not compare the actual added bias.
A failing regression test reproduced a false supported decision for a
changed bias. The corrected audit rejects changed bias and missing bias
for the intervention mode. Older command-only records imply zero added
bias, consistent with their driver and recorded mode.

The combined input and temporal audit suite passes 31 tests with 100%
statement coverage for those two modules. The actual source temporal
comparison was rerun through the corrected audit and still passes.
Future input interventions must add both the value and its invariant
check when the option is introduced.

The full NEURON driver is not covered by that unit-coverage claim. Its
source runs, saved parameter readback, actual current traces, and separate
topology audit provide integration evidence. The recorded-bias run is
complete and [rejects sufficiency](h01-l2-recorded-bias-result.md) at the
tested setup. No physiological promotion is made.
