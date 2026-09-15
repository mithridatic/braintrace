# Recorded filter correction passes the control-prediction gate

Replacing the assumed immediate first-order observation response with the recorded
four-pole Bessel response fixes much of the control timing and peak-shape error.
The frozen correction reduces pooled error on six excluded-from-fit early controls
from **5.680301 to 1.979975 pA RMS (65.143%)**. Every early control improves. The
excluded long control improves from **1.356559 to 1.064275 pA RMS**. All registered
control gates pass. This accepts a conditional observation correction, not a human
channel or donor model; the six population scores remain unchanged.

## Source-backed correction

The original source, human PAX6 CDH12 specimen 840043506/session 840043481, contains
LPF Cutoff=2000, Secondary LPF Cutoff=10000 and Hardware Type=1 at headstage zero
in all eleven inspected sweeps. The [unaltered fields](recorded-filter.json) retain
the primary cutoff's empty source unit string. [MIES documentation](https://alleninstitute.github.io/MIES/labnotebook-descriptions.html)
identifies the primary cutoff field as Bessel and hardware type 1 as MultiClamp
700B. The [manufacturer manual, p144](https://neurophysics.ucsd.edu/Manuals/Axon%20Instruments/MultiClamp_700B.pdf)
specifies a four-pole primary Bessel filter and its 2000 Hz setting. The separate
secondary filter was not cascaded into the primary observation path.

The [new helper](../../../h01_pax6_recording_filter.py) uses the fixed 2000 Hz
analog magnitude cutoff, exact conjugate-pole impulse and exponential responses,
and signed superposition of the original command edges. The control response has
a leak term, unresolved fast charge, one slower charge relaxation and an observation
latency. These are conditional small-command response parameters, not measured
physical capacitances, isolated ionic currents or a demonstrated hardware delay.
All biological observations used here are from the pinned human source; the filter
itself is an instrument model.

The [frozen parameters](frozen-candidate.json) are gL=0.2464314 nS,
C0=1.6701939 pF, C1=2.3167097 pF, tau=1.1103688 ms and latency=0.1195983 ms.
The fit converged in nine evaluations, with no active bounds. It used the complete
early controls in 70,74,78,79,88 and the complete long-control responses of 79 and
88 without double counting early samples: 155875 original training samples.
Parameters were written before evaluating the six excluded early controls and
the complete long control 83. All these responses had been exposed previously;
this is diagnostic prediction excluded from this fit, not blind or independent
physiological validation. No new response arrays were opened.

| Excluded early control | Previous model RMS, pA | Corrected RMS, pA |
| --- | ---: | ---: |
| 72 | 5.810172 | 1.865637 |
| 76 | 5.434656 | 1.584982 |
| 83 | 6.075132 | 2.146890 |
| 99 | 5.687307 | 1.773105 |
| 101 | 5.848402 | 2.027554 |
| 103 | 5.181200 | 2.380131 |

## Direct review and remaining errors

Opened the [six complete early-control views](excluded-control-predictions.png),
[all six onset details](excluded-control-onsets.png), and the
[excluded long-control response](excluded-long-control.png). The corrected curves
remain near baseline during the observed delay and then follow the narrow positive
and negative peaks. This removes the previous model's premature broad response.
Amplitude differences remain: the common correction overpredicts control 103's
peak and slightly underpredicts several others. It also misses some of the late
shoulder after the peak. The complete long control retains its noise and isolated
excursions; no observations were discarded. Its early negative peak and late
level are both better represented by the fixed correction.

The [decision](decision.json) passes convergence, interior latency, at least 50%
pooled early-control improvement, no early control over 5% worse, and no more than
5% worsening on the long control. These are the original rules of this registered
correction, not replacements for donor or population acceptance. The full signed
predictions and residuals are retained in [predictions.npz](predictions.npz).

The next active-current fit can use this fixed observation response. It must
add acquired conditioning and paired-pulse constraints before treating recovery
parameters as identified, and must preserve all earlier rejected results. The
previous nineteen-parameter candidate remains rejected; no active-current
parameters were refitted here. Sustained sweeps 105,106,107,109,110,111, the external
PAX6 donor responses, and original whole-cell holdouts remain unopened.

## Exporter bug and prevention

The shared exporter omitted the available filter and hardware fields. A
[failing regression](regression-before.txt) reproduced the missing LPF Cutoff key.
The exporter now retains the primary/secondary cutoff, hardware type and signal
fields using the existing headstage-specific last-finite lookup. It preserves
empty source unit strings and does not invent defaults when a field is absent.

The first array-preservation test incorrectly compared floating SI conversion
results with exact integer literals and failed at 5.000000000000001 versus 5.
The corrected regression compares arrays before and after adding metadata,
which tests the intended byte-preservation property. The failed intermediate
test receipt is retained in all-tests.xml. No numerical conversion was changed.
Future observation models must read the available recording settings and check
their response against the control waveform before joint biological fitting.

## Verification and execution

[All 65 affected tests pass](passing-tests.xml): exporter coverage is 100%, and
the new observation helper has 98% line coverage ([coverage](passing-coverage.json)).
Independent state-space matrix exponentials verify the impulse and exponential
responses; other tests check DC/cutoff normalization, complex-step derivatives,
pulse superposition, long tails, original metadata scope and absent fields.

[Artifact verification](verification.json) checks all 236875 retained observations,
original sample indices and clocks, previous-model predictions, role selection,
errors and source/candidate hashes. [Re-export verification](export-verification.json)
confirms all current, command and time arrays remain identical for 890000 original
samples across eleven sweeps, while new metadata matches the source notebook.

The [terminal receipt](terminal.json) records normal completion in **80.141 seconds**
on local CPU, below the 180-second cap. No membrane or population simulation ran.
The [specification](../../../../specs/2026-09-14-h01-pax6-recording-filter.md)
defines the correction and its unchanged limits.
