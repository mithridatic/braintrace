# Sodium inactivation causal test

## Claim and boundary

Slow transient-sodium inactivation sustains inward current and prolongs the
first spike of the published PV model under the 0.19 nA step.
This is a claim about this model. It is not a claim about measured human gates.

## Controlled intervention

Multiply NaTg hTau by 0.5 in soma and axon. Keep hInf, mInf, mTau,
maximum conductance, reversal potentials, geometry, input, and initial voltage
unchanged. Use a separate copy of the source mechanisms.
Run a factor-1 control from that copy and compare with the original reference.
The intervention changes recovery timing as well as inactivation timing.
It does not isolate only the depolarized phase of the spike.

## Nested tests and decision rules

1. At fixed voltage, hInf must remain unchanged and hTau must halve.
   Activation rates must remain unchanged. A failed check invalidates the test.
2. The factor-1 control must reproduce the original voltage within 0.01 mV
   on the fixed stimulus grid. This is an implementation check.
3. The first complete spike must remain. Its time above -20 mV must shorten
   by more than 0.02 ms. This margin is a diagnostic decision threshold,
   not a biological tolerance or a completed numerical error bound.
4. Record h, sodium current, and voltage at the rising and falling crossings.
   The waveform must support the proposed sequence of gate closure, loss of
   inward current, and repolarization. A shorter spike alone is insufficient
   to establish this mediation in the voltage-feedback system.
5. Record every spike. Loss of all spikes contradicts the proposed waveform
   correction. It does not establish a shortened action potential.

If the intervention is correct but the spike does not shorten, reject the
stated directional prediction at these conditions. If it shortens, retain
only a conditional causal contribution. Do not claim a unique cause.
Use matched-voltage gate tests to separate gate timing from voltage feedback.
Check numerical refinement before promoting a fitted model.

Keep experiment records in evidence. Keep the supported explanation and its
limits in the living causal document. Do not turn that document into a run log.

## Separate closure from recovery

Define closure as h greater than hInf and recovery as h less than hInf.
At equality the derivative is zero. Do not alter either equilibrium curve.
Use a separate mechanism copy with two positive time-scale factors.
The original single factor remains the closure factor. A recovery override
selects the recovery time scale; an unset override preserves the earlier model.

Test closure/recovery factors 0.5/1 and 1/0.5 against 1/1 and 0.5/0.5.
Prediction: closure-only acceleration shortens the first complete spike by
more than 0.02 ms. Recovery-only acceleration has a smaller effect on its width.
Do not assume that spike count increases with recovery speed; measure the
full train and report any contradiction to that proposed explanation.
If the recovery-only width effect equals or exceeds the closure-only effect,
reject the proposed separation under this protocol.
This state-dependent intervention is a diagnostic construction, not a claim
that human channel kinetics have this piecewise form.
Check fixed-voltage relaxation from both h=0 and h=1 before whole-cell runs.

## Spatial robustness of the causal result

Repeat the unchanged and closure-only cases at mesh factor 27.
Keep all other physical and numerical parameters fixed.
The first-spike shortening must retain its sign and exceed 0.02 ms on both
meshes. Report the change in the measured effect between meshes.
Do not call this full spatial convergence. Inspect each full spike train.
If the first-spike effect disappears or reverses, do not use the causal claim
to choose a fitted time-scale parameter until the numerical discrepancy is resolved.
