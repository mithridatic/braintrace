# H01 direct observations: visual review

Reviewed 2026-09-11 (analysis completed September 12 UTC). This record covers
retained B3 sweeps 56, 50, 53 and SP12 stage-0 A, B, and Nap-zero at sweep 56,
with the corresponding human recordings. The analysis ran on the CPU of the
Vast GPU host in `/workspace/h01-direct-analysis`; no new simulation was needed
because the saved traces contained the required observations.

## Book basis and chart design

Hartshorne's rotor example, Figure 101, nests individual measurements within
parts and consecutive groups; its interpretation uses their common vertical
datum before reducing the sample. See [printed p. 118, Figure 101](<C:/Users/J/Documents/Diagnosing Performance and Reliability/pages/p108.jpg>)
and [printed p. 119](<C:/Users/J/Documents/Diagnosing Performance and Reliability/pages/p109.jpg>).
The electroplating example, [printed p. 121, Figures 103–104](<C:/Users/J/Documents/Diagnosing Performance and Reliability/pages/p111.jpg>),
separates positions and cycles. Both example pages were visually inspected.
This is a design principle for connected physical observations, not a rule
that any three points establish a cause.

Here phases are nested within selected recovery cycles, within input/system.
Panels share voltage scales. First, second, and last complete cycles retain
their identity; repeated selections are deduplicated. Full original samples
inside selected windows accompany the phase points. No line bridges an
unselected interval. Missing phases remain gaps rather than extrapolations.

Initial inspection showed that 0, 2, and 6 ms after the trough missed the
human's slower depression. A second multivari view therefore uses 20, 60,
and 200 ms. These are separate views of the same retained response. A no-spike
trace has no trough: its offsets are from pulse onset. The saved absolute times
and pulse-clock trajectory must be read alongside event-aligned charts.

## Observed patterns and interpretation

The [B3 report and four charts](h01-e-currents/direct-contract-reanalysis-direct.md)
show a faster early rebound in the human traces than in the corresponding
model traces. At low input the human then develops a slower depression, while
B3 returns toward another threshold crossing. An early rebound slope alone
would misdescribe the later discrepancy. The higher-input groups also retain
different cycle counts and unavailable long offsets; those gaps are real
differences in event history, not equal samples or zero voltage.

The [SP12 report and four charts](h01-e-sahp/direct-contract-reanalysis-direct.md)
show both tested KsAHP doses suppressing repeat spikes while preserving the
first model spike. Dose A has a shallow slow depression and later rises;
dose B remains much more negative and gradually recovers. The latter's selected
trough is at about 1246 ms, long after its 1147.35 ms spike peak. Its flat early
multivari segment therefore samples that delayed minimum, not the immediate
post-spike rebound. This is why waveform and absolute time accompany the
event-aligned view. The human's peak is about 1225.76 ms; neither candidate
repairs that first-spike discrepancy.

The [registered-band reanalysis](h01-e-sahp/direct-contract-reanalysis.md)
still fails both doses. A's late sample median is 1.258 mV above the human;
B is 3.252 mV below late and 6.007 mV below mid-pulse. These numbers document
the original QC bands; they do not replace the response-shape observations.
These two doses do not exclude all slow outward mechanisms or establish a
unique sufficient repair.

Nap-zero has no spikes and approaches a subthreshold level near -63 mV.
Its late voltage is higher than B3's late sample median despite removal of
an inward current. The intervention changes spike history and hence the
other state-dependent currents. Under the saved B3 configuration at this
input, Nap removal prevents the first spike; the earlier prediction that
at least three spikes would remain is contradicted. A small baseline current
share did not bound its effect on the active system. This does not identify
the human's missing channel or prove universal Nap necessity.

The V-I charts show different paths on rising and falling voltage and a
distinct no-spike path. Their current is local charge-storage current, not
a guessed channel. The V-Q traces are approximately straight with parallel
slopes, consistent with the local capacitor relation. Each window resets its
charge origin; these are not whole-cell energy measurements. Individual
axial voltage differences, currents, charge, and dissipation remain in the
array bundle. Human ionic-current conjugates are unavailable.

The [earlier direct audit](h01-causal-direct-trace-audit-2026-09-11.md) separately
shows why the adaptive solver's sample mean differs from a time-weighted
mean, and why one terminal sample cannot represent the requested 2100–2300 ms
window. The scorer now refuses that incomplete interval and its derived
input-resistance/current-offset claim.

## Reproduction and persistence

Both scorer CLIs generate the direct bundle before their secondary QC report.
The [contract](../specs/2026-09-11-h01-direct-observation-contract.md) and
[local working instructions](AGENTS.md) require visual interpretation in future
sessions. Machine-generated `visual_review: pending` remains deliberate:
successful rendering cannot certify interpretation. This separate record
documents inspection of all eight charts and the subsequent spacing repair.

The JSON bundles retain source-file hashes, full source reports, software
hashes, and the selected-array archive's SHA-256. Raw NPZ archives remain on
Vast at the `selected_samples.path` recorded in each bundle; they are excluded
from Git. The [review manifest](h01-direct-observation-review-2026-09-11.json) binds the inspected charts
and final reports by SHA-256. Earlier `h01-direct-2026-09-11/b3` artifacts were
development views; the `direct-contract-reanalysis` reports linked here supersede
them. Historical stage decisions are preserved.

Validation: the focused suite passed 65 tests with 98% coverage across the four
new modules. Covered boundaries include adaptive time spacing, partial windows,
invalid time grids, missing geometry or current probes, charge balance, zero or
short spike trains, incomplete final spikes, and rendering with absent current
evidence. The chart-spacing adjustment also passed the three artifact tests.
For a new recording adapter, test its pulse clock and units explicitly before
using these comparisons. Human qualification remains a separate unmet contract.
