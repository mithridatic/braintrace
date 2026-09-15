# Correct the human PAX6 observation response using recorded filter metadata

Continues the approved human-only donor plan after candidate d063a99c failed.
The model's immediate first-order output filter mispredicted the measured control
onset and peak shape. Original NWB metadata for the same eleven training sweeps
reports LPF Cutoff=2000 and Secondary LPF Cutoff=10000. The MIES field description
identifies LPF Cutoff as primary-output Bessel cutoff. The MultiClamp 700B manual
specifies a four-pole primary Bessel filter, with a 2000 Hz setting. Preserve
hardware identity and signal fields from the source before interpretation.

Sources: https://alleninstitute.github.io/MIES/labnotebook-descriptions.html
and https://neurophysics.ucsd.edu/Manuals/Axon%20Instruments/MultiClamp_700B.pdf
(manufacturer manual, specifications p144, hosted by UCSD).
The original metadata has an empty unit string for primary cutoff; document
the external instrument interpretation rather than changing the source field.
Do not cascade the secondary filter into the primary recording path.

## Control-response correction before active-current fitting

Use a fixed four-pole analog Bessel filter with 2000 Hz magnitude cutoff (-3 dB).
Implement its impulse response and response to an exponential input through
conjugate pole pairs, preserving complex-step derivatives without extracting the
real part of a perturbed result. Compare with independent polynomial/state-space
oracles, DC gain and magnitude at cutoff. No time-stepped Python model loop.

For a voltage step, the unfiltered conditional response is gL*step plus C0 times
an impulse plus C1/tau*exp(-t/tau). Filter each term with the fixed Bessel response
and apply one latency delta to the observation. Superpose the actual positive and
negative command edges. These terms describe the small control response; C0 is
effective unresolved fast charge, not a measured physical capacitance. The timing
offset is a conditional observation parameter, not a proven hardware delay.

Fit gL,C0,C1,tau,delta to the complete 35-70 ms early controls in sweeps
70,74,78,79,88, plus the complete 35 ms onward response of the long -10 mV controls
79 and 88 without double-counting their early samples. Use original 25 kHz samples,
baseline centered on 35-44 ms. Bounds are gL 0-3 nS; C0/C1 0-100 pF; tau 0.04-500 ms;
delta 0-0.3 ms. Initial values 0.3,2,2,1,0.12. These are candidate bounds, not
independent human measurements. Use one initial point and at most 80 evaluations,
complex-step derivatives, with a 180-second local CPU cap.

Freeze parameters before evaluating the other six early controls
72,76,83,99,101,103 and the complete long control 83. Their responses were already
exposed in previous work; this is excluded-from-fit diagnostic prediction, not
blind or independent physiological validation. Preserve all samples and compare
the fixed correction against the previous model's saved predictions. Require
at least 50% pooled early-control RMSE reduction, no individual early control
over 5% worse, and no more than 5% worsening on the complete long control 83.
Require optimizer success, finite complete outputs and latency off its bounds.
Other boundary/unidentifiability flags remain explicit, not physiological passes.

If the correction passes, freeze its observation parameters for the next active
current candidate rather than fit away control timing with kinetic parameters.
The next active fit must add acquired conditioning and paired-pulse constraints;
the previous pulse/tail-only fit does not qualify recovery. New active-fit scope
and predictions require their own written registration before execution. Keep
all six sustained prediction responses, external donor responses and whole-cell
holdouts unopened in this control correction. No six-term score is promoted.

## Verification and evidence

New reusable helper and sibling tests must cover pole-pair impulse/step/exponential
responses against an independent state-space exponential, DC/cutoff normalization,
complex-step latency and decay derivatives, pulse superposition, original clocks,
nonfinite or invalid inputs and long tails. Target >90% helper coverage. Retain
source SHA256, original filter/hardware metadata, actual on/off commands, source
and prediction arrays, fitting selection, frozen parameters, all residuals and
terminal receipt. Open full and onset-detail plots; record any surviving shape
or amplitude mismatch. Preserve the rejected model and its evidence unchanged.

## Prevent recurrence through the shared exporter

The original export omitted available filter and hardware fields. Reproduce this
omission in a failing sibling test, then preserve LPF Cutoff, Secondary LPF Cutoff,
Hardware Type, Scaled Out Signal and Scale Factor Units in instrument metadata.
Use the existing last-finite headstage-specific notebook lookup, retain original
unit strings (including empty strings), and never substitute assumed filter values
when a field is absent. Verify that current, command and time arrays are unchanged.
