# Combined sodium candidate

The candidate combines soma NaTg factor 1.10 with sodium closure factor
0.15. Recovery factor remains 1. SK density and calcium removal retain
their source values. Both runs use mesh factor 9 and CVode tolerance 1e-10.

| First-event error, model minus human | 0.19 nA | 0.27 nA |
| --- | ---: | ---: |
| Onset, ms | +8.591027 | -0.520249 |
| Peak, mV | -0.759961 | -0.304873 |
| Duration above -20 mV, ms | -0.005165 | +0.001745 |

The first peaks are closer to the recordings than in the closure-0.18
candidate. The lower-current duration error increases, and the higher-current
onset error increases. The predeclared dominance screen fails. This screen
does not supply a physiological tolerance; tiny interpolated width errors
must not be treated as precise biological uncertainty.

The full response gives a stronger reason to reject this candidate.
At 0.19 nA there are 39 complete positive-peak events. The first two
intervals are 8.843891 and 8.555505 ms. At 0.27 nA there are only three
positive-peak events. A fourth complete -20 mV excursion peaks at
-9.009753 mV near 292.325507 ms. No later complete events occur during
the pulse. From 500 to 1270 ms, voltage stays between -29.870584 and
-29.033319 mV. The human trace has a sustained spike train.

This is a depolarized nonspiking response after the initial events.
The observation alone does not isolate its channel cause or exclude a
numerical contribution. Do not label it a verified human depolarization-block
mechanism. The candidate is not promoted, and its kinetics are not transferred
to production BrainCell.

All events and intervals are in [the report](h01-pv-combined-candidate.json).
Raw time, voltage, and channel observations are in the two files with prefix
`h01-pv-combined-015-`. The reserved 0.23 nA trace was not used.
Run `python -m docs.evidence.h01_pv_combined_candidate` to reproduce the screen.
The analysis handles missing events as a failed screen and distinguishes
complete threshold excursions from positive-peak spikes. Five existing
datum, reference, and regional-audit tests passed during this evaluation.

The next calibration step must address sustained firing as well as first
shape. Further first-peak improvement alone cannot close the cell-validation gap.
