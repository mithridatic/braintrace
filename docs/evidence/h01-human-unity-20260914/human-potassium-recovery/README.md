# Human-only potassium recovery estimate

A human-only effective recovery candidate has been estimated from 120 amplitude
ratios across 12 recordings. Its two-exponential form barely improves grouped
prediction over a single exponential: RMSE 0.15234 versus 0.15286, a 0.34%
reduction. Large between-record errors remain. No mechanism parameter was
installed, and none of the six qualification scores changed.

## Registered experiment and source

The [specification](../../../specs/2026-09-14-h01-human-potassium-recovery.md)
was written before implementation and fitting; its hash is in
[result.json](result.json). Inputs come from the separately sealed human rows of
Wilbers Fig. 4, Dataverse 10.34894/L5J0SD version 3.0. See the preceding
[source checkpoint](../human-pyramidal-channels/README.md) and
[input receipt](inputs.json). These are published derived amplitude ratios,
not original NWB current traces.

Twelve of thirteen human rows provide the registered recovery protocol; the
missing-protocol row is explicitly excluded. Each included row has ten complete
delay/amplitude pairs. Three duplicate amplitude columns per row were checked
against canonical columns and excluded from the objective, leaving 120 distinct
source observations. No mouse data, per-record gain, outlier deletion or source
response clipping was used. The source recovery plot uses a protocol filter and
does not supply an additional recovery QC filter.

The source [paper](https://research-portal.uu.nl/ws/files/206755646/sciadv.ade3300.pdf)
identifies recovery at commanded -80 mV and 34 C. The inactivation dictionary's
`ms-1` labels are interpreted as ms because the paper and source functions define
exponential time constants. That reconciliation is an inference, not an edited
source dictionary. This fit uses explicitly defined recovery delays and relative
amplitudes, not the disputed inactivation fields or `rec_ss` field.

## Candidate and cross-validation

The effective current-amplitude model is
`r0 + (rinf-r0) * [f*(1-exp(-t/tfast)) + (1-f)*(1-exp(-t/tslow))]`.
The estimate using all twelve human recordings is:

| Parameter | Estimate |
| --- | ---: |
| Initial effective ratio r0 | 0.298621 |
| Asymptotic effective ratio rinf | 0.796684 |
| Fast fraction f | 0.397553 |
| Fast recovery constant | 215.336 ms |
| Slow recovery constant | 1438.372 ms |

These are observation-model parameters, not qualified channel-gate rates. Both
zero-delay and infinite-delay ratios are extrapolated model limits; neither is a
direct observation. Measurements span 50 to 7538.671875 ms. The full-fit Jacobian
condition number is 92.9; all selected fits converge and remain away from the
registered bounds. These facts alone do not establish parameter identifiability.

The groups are filename prefixes H21.29.190 (seven recordings) and H21.29.191
(five recordings). Each model is fitted on one group and used to predict all
recordings in the other. Group identity is not verified donor identity. This is
cross-validation on previously acquired data, not a sealed blind holdout.

| Predicted group | Single-exponential RMSE | Two-exponential RMSE |
| --- | ---: | ---: |
| H21.29.190 | 0.164466 | 0.164265 |
| H21.29.191 | 0.134948 | 0.133862 |
| Equal recording weight, both groups | 0.152861 | 0.152336 |

The registered comparative rule favors two exponentials because both groups
improve. The improvement is small and is not physiological acceptance.
[result.json](result.json) retains every start, fit, sample prediction, signed
residual and early/intermediate/late per-record error. No means replace the
individual responses. The unchanged bounds and alternative fits remain available.

## Visual review

Both [predictions](grouped-predictions.png) and [signed residuals](grouped-residuals.png)
were opened and visually inspected. The two families predict nearly the same
response. H21.29.190.11.45.11 and .12 recover much less than either prediction,
with late positive residuals about 0.3 of initial amplitude. Conversely,
H21.29.191.11.41.03 remains above its predictions by roughly 0.2 to 0.25 across
much of the window. H21.29.190.11.42.06 has a sharp intermediate increase, and
H21.29.190.11.41.04 rises and then declines at long delays; the smooth monotone
families cannot reproduce those details. A common kinetic fit therefore leaves
record-dependent level and shape discrepancies. Noise, measurement/protocol
effects and biological variation are alternative explanations not separated by
these derived tables. Per-record amplitude fitting was not introduced to erase
the discrepancies.

## Verification and next boundary

The local CPU process exited 0 after 1.334 s within its 120-second cap
([terminal receipt](terminal.json)); no GPU or remote simulation ran. There are
34 passing tests across the selection and recovery modules, with 100% line
coverage ([tests](tests.xml), [coverage](coverage.json)). Tests cover synthetic
parameter recovery, analytic limits, invalid identities and pairs, duplicated
fields, invalid fitting inputs, equal record weighting and optimizer failure.

An initial weighting test incorrectly assumed changing a time grid could not
change the fitted optimum. Even constant but unequal record targets on different
grids can change the optimum. The corrected test checks the objective at a fixed
prediction, isolating record weighting from sampling and model fitting. No fitter
behavior or acceptance threshold changed to accommodate that test mistake.

Next, reconstruct and verify the experimental conditioning/test command and
current measurement operator before mapping this effective response to gate
rates. The released model driver uses a different delay grid than the published
experimental table, so it cannot stand as proof of the experimental command.
Original traces, their measurement/QC metadata and the unresolved source sodium
exclusion flag remain missing. The successful comparative fit does not authorize
copying these constants into the final-population model or closing any ledger term.
