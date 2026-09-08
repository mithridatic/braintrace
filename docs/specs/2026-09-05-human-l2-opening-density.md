# Test sodium amount with opening speed held fixed

The opening-factor-two candidate improves each rising phase but makes
all spike peaks too low. Test a sodium-density factor of 1.3 relative
to the original fit, with opening factor two fixed. This factor is an
inferred diagnostic value, not a measured channel density.

Control is h01-l2-sodium-opening-double. Change only somatic gbar_NaTs
through the existing density helper. Keep calcium factor 1.375, Ih 75
distributed, passive reversal shift -4 mV, recorded bias, recovery factor
one, mesh 9, CVode 1e-10, sweep 50, soma recordings, and endpoint 2100 ms.

Require all these prospective conditions:

- Five complete positive-peak spikes remain.
- Every absolute peak-voltage error is smaller than in the slow-opening
  density-one control.
- Every absolute rise-to-peak error remains more than 0.01 ms smaller
  than in h01-l2-sodium-opening-control, the opening-factor-one reference.

The phase margin retains the existing diagnostic definition. It is not
a physiological tolerance. Record all onset, fall, duration, interval,
recovery-minimum and phase residuals. Missing or extra events reject the
combined prediction; malformed data or unintended setup changes invalidate
it. Verify the single genome-row change, raw mapping, input and endpoint.

The prediction tests joint response constraints, not a full cell fit.
Changing density also changes voltage feedback, so it need not preserve
the rising-phase benefit. The experiment can reject that compatibility.
The reserved response remains excluded from calibration.

In parallel, tighten CVode tolerance for the unaltered opening-factor-two
control to 1e-11. Require equal event counts and peak signs, onset changes
<=0.1 ms, peak changes <=0.1 mV, and duration changes <=0.01 ms. Each
separate phase change must be <=0.01 ms. Retain all differences. Spatial
and intervention-specific checks remain separate.
