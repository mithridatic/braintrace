# Human PAX6 control-gain test

A gain fitted only from the small command pulse does not account for the larger
same-command response in the later conditioning sweeps. The long leak-control
sequence also contains a sustained current change during sweep 80 while its
command remains constant. A shared stationary recording cannot be assumed, and
a uniform gain correction is not supported by this test. No H01 score changes.

## Data and tested hypothesis

All 28 early controls from sweeps 70-97 were extracted from the previously
hash-verified human specimen 840043506 recording, session 840043481. Its source
NWB SHA256 remains
`30abbb3cca63b0629242c7ad96f595d6e6ceea2ce7e522fdb5ebf8a3bc2ac944`.
Each control JSON binds both the earlier full source export and the new NPZ.
The original 25 kHz samples, commands and clocks are retained over 35-65 ms.
Baseline centering is separate from the raw current, and the exact pulse mask
uses source indices, avoiding a floating-point boundary shift at 55.04 ms.

Each small command is +10 mV from the holding level over 45-55.04 ms. The
baseline is 35-44 ms. The descriptive late response uses 52-55 ms. Its current
increment divided by command increment is an apparent response slope, not
qualified membrane conductance or access resistance. Notebook zero series
resistance is not interpreted as a measurement of zero resistance.

For each target sweep 89-97, one positive gain is fitted by least squares through
zero against control sweep 72, using all 251 baseline-centered pulse samples.
That gain then predicts the target's first 100 ms after the actual -90 to -20 mV
command using the independently recorded large response in sweep 72. The main
responses do not enter the gain fit. Their baselines are the original 900-990 ms
windows. Full original and centered main currents and clocks remain in each
gain bundle. Matching amplifier commands does not establish actual patch voltage.

## Visual review and result

[All controls](all-controls.png), [gain predictions](gain-predictions.png),
[all nine residuals](all-gain-residuals.png) and the [leak transition](leak-transition.png)
were opened and inspected. All 28 small controls share a sharp onset transient,
a declining positive pulse response and an opposite return transient. Later
control samples overlap across families with visible repeat variation. The
fitted controls reproduce much of the waveform, but their transferred predictions
stay substantially below the target main responses over most of the post-5 ms
window. The residual heatmap preserves all nine targets and every phase sample.

| Target | Control-fitted gain | Control fit RMS (pA) | Main residual RMS, 5-100 ms (pA) |
| --- | ---: | ---: | ---: |
| 89 | 0.9460 | 2.326 | 27.384 |
| 90 | 0.9497 | 1.789 | 28.594 |
| 91 | 0.7989 | 1.839 | 33.495 |
| 92 | 1.0161 | 1.733 | 26.958 |
| 93 | 0.9571 | 1.626 | 30.663 |
| 94 | 0.9853 | 1.430 | 29.974 |
| 95 | 1.0215 | 1.643 | 28.496 |
| 96 | 0.9864 | 1.728 | 29.167 |
| 97 | 0.9572 | 1.497 | 30.303 |

These RMS values describe different-amplitude commands and are not a statistical
significance test. Direct phase samples and complete trajectories support the
interpretation. At phase 20 ms for target 91, reference current is 38.734 pA and
target current is 61.703 pA after their own baseline centering. The control-derived
gain predicts approximately 30.945 pA, leaving a 30.758 pA residual. Eight targets
have positive residuals at every post-5 ms sample. Target 92 has a brief small
negative excursion (minimum -0.377 pA) amid the broadly positive discrepancy.
No negative sample or failed condition is removed.

The conclusion concerns the specified uniform multiplicative gain explanation.
It does not exclude voltage-dependent clamp effects, changed biological states,
or other recording changes. No rescaling is applied to the human source data.

## Time boundary of the leak change

The raw current in control 80 becomes less negative around 1700-1800 ms and stays
at the higher level for the remainder of its negative command. The command remains
near -99.987 mV throughout 1100-2100 ms. For example, retained individual samples
are -20.625 pA at 1500 ms and -13.125 pA at 1800 ms. The next control also has a
less negative late current than control 79. The early small test pulse in control
80 precedes this change and cannot establish stationarity later in that sweep.

The full waveforms locate the change; the phase samples are reference points,
not change-point confidence estimates. The record establishes a response change
under constant command, not its unique cause. In particular, a leak or seal
mechanism has not been isolated. The earlier total protocols precede this event;
the conditioning protocols follow it. Their initial-state/recording equivalence
therefore needs an explicit model or independent evidence.

The first render clipped some transient extrema with fixed limits. The full
views now derive common limits from all displayed samples; separate late-current
views remain magnified and are labeled accordingly. The revised figures were
reopened. This avoids mistaking a display limit for observed transient amplitude.

## Verification and next inference

The affected suite passes **141 tests**. The new helper has 100% line coverage.
Its 22 tests cover known gain recovery, independent prediction, preservation of
baseline offsets and times, zero-energy/negative-gain controls, mismatched
commands and clocks, extra transitions, malformed masks and incomplete windows.
The tests explicitly show that changing the main response cannot change the
control-fitted gain. Computation used local CPU array algebra only.

Candidate fitting may proceed with explicit protocol histories and recording
epoch assumptions. The earlier total-current block and later conditioning block
must remain distinguishable; a stationary difference or a uniform gain correction
cannot silently merge them. Retain the earlier block as a separate challenge,
and use within-sweep transitions to constrain a candidate before attributing
between-block differences to channel rates. This does not justify discarding
unfitted observations or assigning a physiological cause to the change.

Kinetic deployment and external-response access remain gated. The reserved
external donor's currents are unopened, the original whole-cell split is unchanged,
and no human physiology, anatomy-transfer or 104-cell driven-window qualification
is claimed by this control analysis.
