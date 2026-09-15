# Sodium kinetic-scaling change: validation rejects the candidate

The tested model change reduces combined validation RMSE from 0.152330 to
0.117044, a 23.16% improvement in normalized sodium amplitude prediction.
It nevertheless fails the registered acceptance gate: four of nine validation
recordings worsen by more than 5%. No parameters are installed and no six-term
score is promoted. The two-scalar scaling branch is closed under this protocol.

This is an executed model comparison against human measurements, not a source
audit. It uses all 200 APs and 3,200 published human amplitude observations under
the matched Hwide command. Seven H21.29.194 recordings supply calibration;
five H21.29.195 and four H21.29.197 recordings supply validation. These are
filename groups, not independently verified donors or a blind holdout.

## Frozen experiment and outcomes

The [specification](../../../specs/2026-09-14-h01-sodium-thermal-transfer.md)
fixes model, split, bounds, optimizer budget, observation windows and the
decision rule. Its executed hash is recorded in [result.json](result.json).
[frozen-candidate.json](frozen-candidate.json) was written before validation
scoring. Every optimizer evaluation is retained in [evaluations.json](evaluations.json).

The baseline uses the published 25 C human sodium rates with both rates scaled
by 2.3**0.9 at 34 C. The candidate changes only the activation multiplier to
8.000000 and the inactivation multiplier to 1.349761. Activation reaches its
registered upper bound of 8; that bound was not expanded. Steady states, the
25 C kinetics, command samples, one -10 mV command correction and the assay
reversal of 141 mV remain fixed. Constant conductance cancels in normalization.
These fitted factors are not independently measured Q10 values.

| Group | Use | Baseline RMSE | Candidate RMSE |
| --- | --- | ---: | ---: |
| H21.29.194 | Calibration, seven recordings | 0.160956 | 0.085090 |
| H21.29.195 | Validation, five recordings | 0.169822 | 0.108587 |
| H21.29.197 | Validation, four recordings | 0.127125 | 0.126824 |
| Both validation groups | Equal recording weight | 0.152330 | 0.117044 |

The following individual validation failures invalidate promotion despite the
aggregate improvement. Ratios compare candidate RMSE with baseline RMSE.

| Recording | Error ratio |
| --- | ---: |
| H21.29.195.11.41.02.nwb | 1.938 |
| H21.29.197.11.41.01.nwb | 2.492 |
| H21.29.197.11.42.02.nwb | 1.466 |
| H21.29.197.11.42.03.nwb | 2.914 |

Both numerical checks pass: refining each held-command interval from 0.008 to
0.004 ms changes baseline amplitude ratios by at most 2.34e-15 and candidate
ratios by 4.22e-15, below the unchanged 0.001 limit. This checks the exact gate
solver under the same held command. It does not establish experimental command
tracking, acquisition filtering or accuracy for a freely firing neuron.

## Direct observations and visual review

[All human comparisons](all-human-predictions.png) and the
[direct gate/current comparison](direct-model-comparison.png) were opened and
visually inspected. The candidate lowers sustained amplitudes toward many of
the lower-response recordings, but it undershoots the higher-response records
listed above. Both models tend to recover upward after their early decrease,
while several human records decline across much of the train. The candidate
does not reproduce that long-train pattern or the larger AP-to-AP fluctuations.

Under identical command voltage, faster activation shifts the candidate's
current rise earlier. Slower inactivation leaves a broader initial current tail,
but lower availability before AP5 and AP200 reduces their normalized peaks.
These are observed model-state changes under the joint two-parameter intervention,
not independently isolated effects of each parameter or a complete explanation
of the human discrepancy. The direct arrays preserve command, both one-sided
current proxies and m/h gates around AP1, AP5 and AP200. Current proxies lack an
absolute conductance density and are not recorded human ionic currents.

## Runtime and checks

The first sequential-scan implementation was explicitly terminated after
126.7 seconds without a baseline result. Its [terminal record](../sodium-thermal-transfer/terminal.json),
interruption receipt and executed source/specification are preserved separately.
The replacement composes exact affine gate updates in blocks of 1024 and carries
block-end states through brainstate.transform.scan. It preserves every interval;
identity padding adds no physiological time or observations. A sequential oracle
across multiple block boundaries and analytic checks verify equivalence.

The full corrected process, including fitting and both refined runs, exited 0
after 6.384 seconds within its 600-second cap ([terminal.json](terminal.json)).
The optimizer converged after eight reported function evaluations and 24 actual
calls including finite-difference evaluations. All runs used local CPU. Seventeen
meaningful sibling tests pass with 100% line coverage of the new solver helper
([tests](tests.xml), [coverage](coverage.json)). Tests cover analytical solutions,
block boundaries, source-held substeps, normalization and invalid windows/factors.

The first implementation's runtime was a preventable execution mistake: exact
scalar updates were compiled but insufficiently batched for this workload. The
correction changes affine execution batching, not the biological model or gate.
Future fixed-command experiments can reuse this verified batching without
repeating the failed full-length sequential run.

## Consequence for the full goal

This shared two-factor temperature-scaling change is not sufficient to qualify
the sodium model across these human responses. Preserve the negative individual
results; do not retry with wider bounds or move validation records into fitting.
The discrepancy includes variation between recordings and an opposite late-train
trend in some records, so another uniform rate rescaling is not the next test.
Original-current and measurement-chain limitations remain explicit. Donor scoring,
type coverage, anatomy transfer and final-population qualification remain required;
none is replaced by the 23.16% improvement in this auxiliary channel comparison.
