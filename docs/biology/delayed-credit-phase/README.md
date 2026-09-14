# Analytic delayed-credit diagnosis and configured-decay check

This phase follows `4e7a65c`. It tests why the earlier fixed-impulse contact
gradient was much smaller than BPTT, without changing production pp-prop or
reinterpreting the encoder-only 2.79% result as a general accuracy guarantee.

## Independent analytic model

A linear shift register emits one delayed unit impulse multiplied by
`element_wise(weight, weight_fn=abs)`, the same weight-only parameter operation
used for H01 contact magnitude. The weights are 0.5 and -0.7, away from the
absolute-value kink. For squared-output loss the exact gradient is `2*weight`.
An independent centered finite difference checks its directional derivative.

For an impulse emitted at zero-based event `s`, delivered at `t=s+d`, with
positive delay `d`, the predicted finite-window/BPTT gradient ratio is

```text
r = (1-a) * a^(d-1) / (1-a^t)
```

Here `a` is the output-trace decay and `t` is the number of completed events
before the output/loss event. Injection contributes `1-a`; crossing the other
`d-1` event boundaries contributes `a^(d-1)`; the warm-up denominator is
`1-a^t`. The weight-only operation has no input trace to multiply into this
output trace. Direct zero-delay credit remains exact because it uses the
within-window derivative.

The sweep tests decays 0, 0.2, 0.8 and 0.99 for immediate output, one-event
delay, five-event delay, and delayed emission after two silent events.
Every learning-rule comparison uses one-event finite windows through
`chunked_online_param_gradients(..., compiled_scan=True)`. BPTT and finite
differences supply independent exact references. Domain guards reject negative
delays/emission times and decay one.

For a five-event delay emitted at event zero, decay 0.8 retains about 12.18%
of the exact gradient, and 0.99 retains about 19.60%. As decay tends to one,
the formula tends to `1/t`, not one. Increasing decay alone therefore cannot
restore exact credit in this analytic weight-only example.

The agreement diagnoses attenuation produced by the implemented normalized
trace law. It does not demonstrate a queue indexing, finite-window oracle or
autodiff defect in this example. It also does not prove this is the only source
of approximation error in nonlinear H01 trajectories.

## Full configured-decay check

`python -m docs.biology.default_contact_probe` saves the analytic sweep and
reruns the unchanged two-neuron, reciprocal-contact, one-glia fixture at its
configured decay, 0.99. It keeps the 0.5 ms delay, imposed impulse schedule,
180 physical ticks, objective and parameter group from the earlier phase.
Fresh BPTT, directional finite differences and normalized descent are checked.
The baseline trajectory and BPTT gradient must match the retained earlier
artifact. No previously measured 0.2/0.8 result is overwritten.

`settings.json` pins the complete session implementation and biology;
`default-contact.json` pins the earlier reference artifact and this probe;
`analytic-sweep.json` records exact, predicted, online and finite-difference
results. Verification also pins the oracle and the imported fixture helpers.

## Recorded results

All 16 analytic cases matched the independently derived trace law and exact
references. The affected gate passed **20 tests in 117.51 seconds**, with
**100% statement coverage** for both new diagnostic modules and four existing
unused-readout-state warnings. The full evidence run took 106.13 seconds under
the gate; this is not an isolated performance benchmark. Artifacts were retained
from the passing test only after checking the executed probe's source hash.

| Full contact fixture, decay 0.99 | Result |
| --- | ---: |
| Relative gradient error against BPTT | **86.3673%** |
| Gradient norm / BPTT norm | **13.6353%** |
| Gradient cosine against BPTT | 0.999838 |
| Objective before normalized update | 106990.088549 |
| Objective after normalized update | 106989.249150 |
| BPTT directional derivative | 8394135.833576 |
| Centered finite difference | 8394135.835988 |

The contact vectors are [-1079639.475, -380566.604] at decay 0.99 versus
BPTT [-7866518.269, -2932949.187]. Increasing decay from 0.8 reduced error
from 93.38% to 86.37%, but magnitude remains unqualified. The baseline trajectory
and freshly calculated BPTT vector matched the previous artifact; the changed
decay did not improve this comparison by changing the forward problem.
Direction and normalized descent remain good, while raw gradient scale does not.

## Consequences and next implementation step

No production learning-rule change or post-hoc gradient rescaling is made here.
The next candidate is a separately specified exact influence accumulator for
weight-only operations, while retaining the existing approximation for other
ETP operations. It must pass mixed-operation, finite-window and H01 contact
regressions before adoption; a blanket scaling factor would not establish
correct temporal credit.

These remain synthetic fixed-impulse diagnostics. Endogenous recurrent firing,
stochastic-release derivatives, ARC optimizer performance, measured extracellular
placements, GPU execution and full-population physiology remain unqualified.
