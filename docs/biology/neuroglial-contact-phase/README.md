# Delayed contact learning in the synthetic neuroglial session

This qualification follows `2b5ddaa`. The earlier 2.79% encoder-gradient error
was specific to a three-event, one-neuron fixture; it is not a general pp-prop
accuracy bound. This phase measures a different parameter group and timescale.

The fixture contains two synthetic instances of the existing seven-CV source
neuron, reciprocal excitatory contacts, one glial fragment and explicitly
declared shared extracellular pools. Contact construction recomputes and pins
both neuronal CV geometry maps. The contacts use 0.001 and 0.002 uS initial
weights, 2 ms receptor decay, 0 mV reversal and the retained 0.5 ms delay.
Contact release probability is one for this deterministic diagnostic.

## Stimulus and reference contract

`python -m docs.biology.neuroglial_contact_probe` constructs the complete session
and evaluates nine 0.1 ms events at the 0.005 ms cable clock. Feature currents
remain small. Fixed presynaptic impulses are imposed at event boundaries:
the first cell at 0.1 ms and the second at 0.2 ms. The diagnostic temporarily
supplies these spike values to native `enqueue_future_events`, then restores
the original spike states. It does not write a conductance directly into a
receptor, shorten the delay or seed a weight-independent delivery queue.
Queued conductance is computed through the model's actual contact-weight
operation, retaining its temporal credit across event boundaries.

These are explicit external diagnostic impulses, not endogenous neuronal
spikes. They do not invoke the chemical release driver a second time, and
therefore do not represent presynaptic glutamate release physiology. Normal
cable, glial, chemical and native delivery evolution remains active throughout.

The objective is summed squared soma voltage. An objective-specific parameter
view requests contact-weight derivatives only. One-event finite windows at
pp-prop decays 0.2 and 0.8 are compared with BPTT over the same physical path.
Both contact derivatives must be finite and nonzero, decay must matter, and
the higher-decay vector must have positive alignment and relative error below
one against BPTT. No equality requirement is imposed on the approximate rule.

A centered finite difference uses epsilon 1e-8 uS in the normalized high-decay
gradient direction. A step of length 1e-7 uS must reduce the objective while
remaining strictly above the absolute-weight kink at zero. The no-impulse
control must have exactly zero contact gradient. Event-end voltages through
0.6 ms must agree exactly with that control, then diverge after arrival;
both receptors must have positive final conductance. Parameters and all
physical/eligibility state are restored even if qualification raises an error.

## Result: direction passes, magnitude remains unqualified

The ten-test affected CPU gate passed in 236.95 seconds, including the native
delivery regressions. Probe coverage is 127/128 statements (99.22%); four
unused-readout-state warnings remain. The retained artifact came from the
test's fresh session and was copied after verifying its probe-source hash.
Its 216.06-second elapsed time is not an isolated performance benchmark.

| Quantity | Result |
| --- | ---: |
| Physical duration | 180 ticks / 0.9 ms |
| High-decay relative error against BPTT | **93.3788%** |
| High-decay gradient cosine against BPTT | 0.999528 |
| BPTT / finite-difference directional relative error | 1.01e-9 |
| No-impulse contact gradient | exactly [0, 0] |
| Objective before normalized update | 106990.088549 |
| Objective after normalized update | 106989.249411 |
| Final receptor conductances | 0.000860708, 0.001809675 uS |

At decay 0.8 the two contact gradients are approximately [-514906, -210220],
versus BPTT [-7866518, -2932949]. They point nearly in the same direction but
are much too small. Passing the prespecified bounded-error, direction and
normalized-descent criteria therefore **does not qualify gradient magnitude**.
Normalizing the update deliberately removes its scale; this does not establish
an appropriate raw learning rate or the behavior of the session's optimizer.

The contact result must not be pooled with the earlier 2.79% encoder result.
The comparison uses decays 0.2 and 0.8; `settings.json` also records the session
constructor's default 0.99 learner, which was not the derivative estimator used
in this comparison. Default-decay contact accuracy remains untested.

The matched trajectories agree exactly through 0.6 ms, before any scheduled
contact arrival can affect the sampled output, and diverge afterward. This
rules out pre-arrival stimulus differences as the source of the observed
contact effect. Both final conductances are positive and chemistry is valid.

## Evidence boundaries and next tests

`topology.json` records contact and clone provenance; `settings.json` pins the
biology and implementation; `contact-probe.json` records both gradient vectors
and full event-end control trajectories. The source archive checksum override
is scoped to the existing synthetic development fixture. No measured synaptic
or extracellular anatomy is claimed.

This is contact-weight credit conditioned on fixed impulses. It does not
qualify feedback-driven recurrent spiking, stochastic-release derivatives,
ARC accuracy, myelin learning, GPU or population-scale runtime. The next step
is an analytic delayed-contact reference that isolates temporal attenuation
from an implementation defect, including the configured 0.99 decay, before
altering pp-prop. Then replace imposed impulses with independently verified
endogenous spiking and add matched release/feedback controls. Measured source/domain
placements remain a separate anatomical qualification requirement.
