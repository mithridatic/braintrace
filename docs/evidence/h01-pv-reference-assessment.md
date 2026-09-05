# Published inhibitory cell: direct reference assessment

The original HL5BN1 cell runs in NEURON 9.0.2.
Its source is ModelDB 267587, commit `82cdd91bc93942ba19315371330a2412e064baf5`.
The [source manifest](h01-pv-neuron-reference-sources.json) records paths and hashes.
The source equations and HOC template were not changed.
They remain in the local cache, outside the BrainTrace package.

The template replaces the axon with two 30 µm sections.
It derives their diameters from the source axon.
The first section attaches to the soma at position 1.
This is an inferred axon, not a complete measured axon.
The template sets each remaining section to `1 + 2 * int(L/40)` segments.
The [reference geometry](h01-pv-neuron-019.json) records the resulting sections.

The driver uses the released circuit settings: 34 °C and initial voltage −80 mV.
It applies no background synaptic current.
The current step starts at 270 ms and ends at 1270 ms.
The default time step is 0.025 ms.

| Current (nA) | Human spikes | Model spikes | Direct voltage RMS error (mV) |
| ---: | ---: | ---: | ---: |
| 0.19 | 12 | 12 | 12.14 |
| 0.23 | 31 | 29 | 16.25 |
| 0.27 | 43 | 46 | 19.18 |

The comparison uses the recording clock and recorded voltage values.
It does not shift voltage or align spike peaks.
Each window contains 50,000 human samples at 0.02 ms intervals.
The model trace is interpolated onto those times.
The [comparison output](h01-pv-direct-comparison.json) retains individual spike observations.
The corresponding NPZ files retain the direct residual traces.

![Human recording and released model](h01-pv-direct-comparison.svg)

Model spike peaks are near 43 mV. Human peaks are near 17 mV.
The difference also affects spike timing and voltage between spikes.
The model does not pass direct waveform validation.
A matching spike count does not remove this failure.

Halving the model time step to 0.0125 ms retains 12 spikes at 0.19 nA.
However, the last spike moves from 1173.75 to 1162.46 ms.
Further numerical refinement is required before a final biological assessment.
See [the finer reference](h01-pv-neuron-019-fine.json).

At 0.003125 ms, the fixed-step response has 13 spikes.
NEURON's adaptive CVode solver also produces 13 spikes.
Absolute tolerances of 1e-8 and 1e-10 give sampled peak times within 0.000292 ms.
The spike peaks remain near 44 mV at these settings.
Thus, the coarse 12-spike match does not establish numerical convergence.
See [the adaptive comparison](h01-pv-neuron-adaptive-comparison.json).

Adaptive samples are not equally spaced in time.
An arithmetic mean of those samples biased the first baseline report.
A regression test reproduced this error with a linear voltage trace.
The driver now integrates voltage over the baseline interval and divides by elapsed time.
All saved baseline reports were recalculated from their unchanged voltage arrays.
Use time weights for future averages from adaptive samples.

This is the released circuit-cell reference, not a rerun of the original optimizer.
The cause of the remaining biological mismatch is not established.
Do not correct it with an arbitrary voltage offset.
The next transfer must preserve geometry, channel equations, and voltage conventions.
It must then separate numerical differences from a need for physiological calibration.

The container is defined by [the Dockerfile](Dockerfile.h01-neuron-reference).
The initial build lacked `make`; the actual mechanism compilation exposed that dependency.
The Dockerfile now installs both `g++` and `make` explicitly.
All eleven channel mechanisms compile without changing their equations.
The [driver](h01_pv_neuron_reference.py) advances NEURON through one `continuerun` call.
The [comparison script](h01_pv_compare.py) reads saved arrays and plots the observations.

## Direct human constraints

The [human datum file](h01-pv-human-datums.json) retains each spike peak and
the two -20 mV crossing times. The source optimizer uses -20 mV for detection.
Time above that threshold is not called AP half-width.
The crossing tests cover nonuniform time samples and incomplete edge events.
The first test import assumed that this evidence directory was a regular package.
The test now uses an absolute namespace import, which works under pytest.

The [hyperpolarizing comparison](h01-pv-hyperpolarization.json) uses active
channels, mesh factor 9, and CVode tolerance 1e-10 in NEURON.
No measured voltage or time shift is applied.
Stimulus RMS voltage errors are 0.57956 mV at -0.11 nA and 1.44607 mV at -0.05 nA.
Baseline differences are about 0.517 and 1.042 mV, respectively.
These observations do not establish a pass threshold.
The human sweeps have different baselines; this remains part of the fitting problem.

The source optimization script initializes voltage at -81 mV.
The released circuit configuration used here initializes at -80 mV.
An isolated comparison at mesh factor 9 changes only this initial voltage.
It retains 11 spikes but moves the final spike by 12.9267 ms.
Spike peaks remain near 44 mV, so this change does not resolve the peak mismatch.
The [initial-state audit](h01-pv-initial-state-audit.json) records Ih activation
and persistent-sodium inactivation at 0, 200, 270, 500, and 1270 ms.
Both gate states still differ at stimulus onset.
Adding the gate observers leaves time and voltage arrays exactly unchanged.
The source optimizer fits weighted features rather than the complete waveform.
Its weights must not be interpreted as biological standard deviations.
The [calibration specification](../specs/2026-09-05-human-pv-calibration.md)
defines the prospective 0.23 nA fitting holdout and its limits.

## Conductance sensitivity

Three independent interventions reduce one soma-and-axon conductance by 10 percent.
They use mesh factor 9, initial voltage -80 mV, and CVode tolerance 1e-10.
The original model has 11 spikes in this configuration.

| Conductance reduced | Spikes | First peak change (mV) | Voltage-change RSS (mV) |
| --- | ---: | ---: | ---: |
| NaTg | 9 | -0.7613 | 5079.79 |
| Kv3_1 | 11 | +0.2816 | 5335.91 |
| SK | 12 | +0.0004 | 4957.99 |

Response-change RSS uses the same 200000-sample grid for each intervention.
The human residual uses the separate original 50000-sample recording clock.
The [audit](h01-pv-conductance-audit.json) labels both sample counts.
No time alignment or voltage offset is applied.

None resolves the human peak-voltage mismatch. NaTg reduction also removes
spikes, although its human voltage RMS residual is smaller than those of the
other two interventions. SK reduction matches the human count at this current
but retains excessive peaks and incorrect timing. No candidate is promoted.
These observations are conditional on the chosen mesh and initialization.
