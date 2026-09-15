# Second human connection: electrical transfer passes, synaptic transfer fails

The frozen electrical response transfers to the second recording under its
registered test. The frozen synaptic waveform does not transfer by changing only
connection strength. No model was installed and no population score improved.

| Validation comparison, sweeps 15-29 | Baseline RMSE (mV) | Candidate RMSE (mV) | Decision |
| --- | ---: | ---: | --- |
| Electrical test pulse, original frozen cell | 0.6467403173 | 0.2522810122 | Pass; 60.992% improvement, no sweep >5% worse |
| Synaptic response, all six observation slots | 0.3516458273 | 0.3498954361 | Fail; 0.498% improvement, sweep 29 >5% worse |
| Synaptic response, five actual-spike slots | 0.3741219640 | 0.3721473803 | Fail; 0.528% improvement, sweep 29 9.439% worse |

The only calibrated parameter is peak effective current, 30.208931472171567 pA,
inside the fixed bounds. An independent SVD least-squares solution agrees within
5e-12 pA. Rise, decay, delay, cell resistance and membrane time constant remain
frozen from 0076. The original 130.0316503 pA strength gives still worse combined
validation RMSE, 0.4680872614 mV. Fitting a new strength was insufficient.

## Source, registration and full observations

The [original failed attempt](../human-second-connection/README.md) remains failed.
The [corrected prospective registration](../../../specs/2026-09-14-h01-human-second-connection-r2.md)
preceded this fit. All 30 source sweeps contain exactly five recorded IN3 spikes:
four during the train and one recovery spike, verified over the whole sweep.
The model sums those five spike inputs and their tails. It does not infer or
insert a sixth spike from an uncertain reconstructed command.

All six observation slots remain, including the no-spike interval anchored at
source epoch index 68125. That slot uses the epoch clock, not a fabricated spike
time. Every window retains original samples from -10 to +50 ms and its original
[-10,-2) ms baseline. No event, spontaneous excursion or baseline disturbance
was removed. Both all-slot and actual-spike-only gates fail. This is within-session
human evidence, not independent-donor or H01 physiology validation.

The source is 2015_11_04_0077.abf, Human/EXC IN3 to IN2, linked by the authors'
date/slice/cluster/response-channel fields to the first record. ABF reports
2015-11-05T00:02:40.015, crossing midnight from the table's session date.
[Acquisition](acquisition.json), [recording metadata](recording.json),
[original DAC3 epochs](presynaptic-epochs.json), [frozen candidate](frozen-candidate.json),
[all numerical results](result.json), [raw observations](observations.npz) and
[direct landmarks](direct-samples.json) preserve the evidence.

## Visual review

Opened the [calibration responses](calibration-responses.png),
[validation responses](validation-responses.png),
[presynaptic inputs](presynaptic-inputs.png),
[all 180 residuals](all-slot-residuals.png) and
[electrical transfer](electrical-transfer.png). Raw and centered postsynaptic
panels use common scales that include every plotted sample. No responses were
clipped or downsampled for these line plots.

The presynaptic first, second and recovery spikes have similar peaks but different
post-spike trajectories; the absent-spike slot stays near resting voltage.
Calibration sweep 0/slot 0 has a delayed large positive excursion, and slot 1
has another late excursion around +40 ms. Sweep 14/no-spike slot contains large
positive excursions despite no recorded IN3 spike. These observations do not
identify their current source or establish synaptic causality.

Validation sweep 15/recovery begins on a pre-existing large response and declines
through the window; its fixed baseline therefore leaves a negative residual.
Sweep 16/first has a delayed positive excursion larger than the model. Sweep 29's
small responses, especially recovery, are overpredicted. Across all validation
slots, 26 of 90 worsen by more than 5%. Large late positive and long negative
residuals remain visible throughout both groups, including the no-spike slots.
Neither a common strength nor the frozen waveform accounts for these full traces.

The electrical plots show a repeatable negative test-pulse envelope and return,
with superimposed positive excursions and baseline drift. The frozen cell model
captures enough of that envelope to pass the stated error gate. This does not
uniquely identify membrane/access resistance or measured ionic currents.

## Verification and limits

[Independent verification](verification.json) checks 150 source spikes, all 180
paired windows (1,080,360 voltage samples), 660,030 electrical/baseline samples,
72 landmarks, the analytical convolution and independent strength solution.
All 82 affected sibling tests pass. The command adapter has 94% coverage and
the strength helper has 100%; their [coverage](coverage.json) and [test report](tests.xml)
are retained. Edge cases cover ambiguous truncated pulses, complete final pulses,
missing/extra crossings, malformed source arrays, unidentifiable strength and
bound-active optima. No new physics or physiology claim follows from those tests.

The ABF voltages retain float32 samples and this driver's baseline mean uses that
precision. Its maximum difference from an independently evaluated float64 mean
is 7.63e-6 mV. The original values are preserved; this does not explain the failed
gate. The current waveform is inferred, not measured delivered/ionic current.
The unresolved Clampex truncation rule is not used as a synaptic input.

The [run receipt](terminal.json) records 5.295 s on local CPU under a 120 s cap.
No kinetics were refitted after validation and no long simulation was launched.
The strength-only transfer branch is closed. A future candidate needs a distinct
human-constrained mechanism and untouched validation; do not reuse this failed
validation set to select a replacement and call it independent validation.
