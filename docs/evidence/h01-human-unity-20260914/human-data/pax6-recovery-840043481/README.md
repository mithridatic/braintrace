# Human PAX6 paired-pulse recovery

All 19 recovery conditions from human specimen 840043506, session 840043481,
are now paired on their original 25 kHz clocks. The joint fit constrains a
conditional recovery envelope; it does not qualify a channel implementation or
promote any of the six H01 scores. External specimen 835648767 remains unopened.

The source NWB SHA256 is
`30abbb3cca63b0629242c7ad96f595d6e6ceea2ce7e522fdb5ebf8a3bc2ac944`.
Each sweep bundle retains both pulse currents, their original times, their
difference, baseline samples and all source arrays. JSON records bind the NPZ
bytes and source. Command potentials are amplifier commands, not measured patch
voltage; liquid junction and series-resistance corrections remain unqualified.

The 300 ms pulses have command +60.0063 mV, recovery -89.9937 mV and holding
-19.9937 mV in all 19 conditions. Gaps range from 5 to 4000 ms. The first and
second voltage jumps differ, so subtracting their responses does not establish
isolated potassium current. Onset transients remain in the source bundles.

## Visual review and direct observations

[Paired currents](paired-currents.png) and [baseline windows](baseline-windows.png)
show little second-minus-first current after a 5 ms gap, a clear increase after
50 ms, and a large decaying second-pulse response after 4000 ms. Final holding
current is elevated in the longer-gap sweeps and continues to fluctuate. Its
change from the initial baseline reaches about 27 pA; this is not established as
instrument drift and was not subtracted from the source data.

[Joint predictions](joint-fit.png) and [all residuals](all-residuals.png) were
visually inspected. Residuals are structured by condition and phase: 30, 100 and
200 ms gaps have positive stretches, 1000 ms has broad negative deviations, and
4000 ms has positive early deviations that diminish later. Aggregate fit error
does not establish adequacy. Every condition and original fit sample is retained.

[Direct phase samples](condition-observations.json) retain first and second
currents separately. At phase 200 ms, second-minus-first current is 40.625 pA
after a 30 ms gap, 26.250 pA after 35 ms and 20.625 pA after 45 ms. These are
individual observations, not repeated-condition uncertainty estimates. Identical
command levels do not establish identical biological state or recording quality.
Noise, recording history and baseline changes remain possible explanations.

For the candidate's allowed parameters, each recovery contribution is
nondecreasing with gap: its derivative is proportional to
`A * (1 - f*exp(-300/d)) * exp(-gap/r) / r`, times the positive phase envelope.
Adding more independent populations of this form cannot reproduce a decrease
exactly under otherwise identical conditions. This is a property of the candidate,
not a universal statement about biological recovery.

## Conditional fitting result

The specification defines two availability populations, with nonnegative
amplitudes and initial fractions in [0,1]. All 6750 original samples per condition
in phase 10-280 ms enter jointly, without per-sweep free offsets.

| Quantity | Unadjusted | Baseline sensitivity |
| --- | ---: | ---: |
| Recovery constants (ms) | 60.0, 510.7 | 57.2, 513.0 |
| Decay constants (ms) | 31.7, 747.7 | 29.0, 673.6 |
| Initial availability fractions | 0.153, 0 | 0.149, 0 |
| Residual RMS (pA) | 13.718 | 13.003 |

Both optimizers converged in under two seconds on local CPU. The second initial
fraction is at its zero bound. Fully inactivated initial availability is physically
admissible, so this bound alone does not invalidate the fit. It also does not
establish identifiability or physiological correctness. These are conditional
constants at two commands, not full voltage-dependent rate functions.

The sensitivity case subtracts final-minus-initial baseline once per condition
before fitting. This is an alternative assumption, not a measured drift correction
or a confidence bound. Its plot adds that baseline change back for comparison with
the unchanged current difference. No animal-derived kinetics were introduced.

## Verification and next evidence

The seven targeted source/QC/analysis test modules pass: 98 tests. Coverage is
94.44% for pulse pairing and 100% for joint fitting; the other three measured
analysis modules have 100% line coverage. Tests include known state transitions,
retained drift and timestamps, unequal pulses, extra transitions, nonuniform
clocks, missing windows, invalid matrices and protected external-source access.
The two uncovered pairing branches are defensive phase/baseline checks.

The initial test command incorrectly loaded the root JAX conftest in this
data-only environment; the next coverage command used package-qualified names
although tests import local modules. Use `--confcutdir=docs/evidence` and local
module names for coverage, as in the receipt, to avoid repeating these mistakes.
The final run completed without those collection or coverage warnings.

Next, inspect recording history and baseline behavior, and constrain the candidate
with the actual total/prepulse/sustained command families. Freeze predictions,
QC and scoring before external currents are opened. Whole-cell validation and
transfer onto audited H01 anatomy remain required. None of this fit closes the
104-cell driven-window or anatomy-transfer gate.
