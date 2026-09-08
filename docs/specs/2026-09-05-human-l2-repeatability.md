# Direct repeatability near threshold

Before another channel fit, inspect the seven long-square trials with
metadata amplitude 200 pA: sweeps 56 through 62. Selection uses stimulus
metadata only. They are the only repeated long-square amplitude in this
file. The 250 pA calibration trial has no same-amplitude repeat.

Verify complete command waveforms, sampling rate, bias, pulse boundaries,
source units, and baseline stability for each trial. Use the established
Allen pipeline-1.0 stored-SI interpretation and minus-14 mV correction.
Retain every complete corrected -20 mV event, its onset, peak, duration,
and each interval. Record missing events explicitly. Do not replace the
individual observations with averages or fit these trials in this step.

Report input equivalence separately from response variability. If the
inputs or bias differ, do not claim repeated identical stimulation.
Baseline screening uses the existing 520--1020 ms window, SD below
0.5 mV, absolute drift below 1 mV, and absolute bias below 100 pA.
These checks are partial quality screens, not proof of healthy physiology.

The observations can constrain a later 200 pA threshold-response check.
They cannot establish variability at 250 or 310 pA, or between donors.
Do not inspect reserved sweep 53 voltage. Do not derive a convenient
acceptance range from model error or treat absence of repeats as zero
biological uncertainty. Further acquisition-quality evidence remains
necessary before selecting final physiological tolerances.
