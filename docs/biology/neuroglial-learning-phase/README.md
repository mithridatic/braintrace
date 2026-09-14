# Bounded neuroglial session learning qualification

This diagnostic follows `cb8a16e` and uses its explicit synthetic session:
one seven-CV neuron, one glial CV, and two declared extracellular volumes.
The donor neuronal dynamics, glial electrical K buffering, extracellular
transport, tonic receptor and astrocyte calcium run at the normal cable clock.
There are no recurrent contacts in this fixture.

Run `python -m docs.biology.neuroglial_learning_probe`. Source anatomy and domain
placements are synthetic; the development archive checksum substitution remains
scoped to the existing fixture. `settings.json` records biology, dependency and
implementation identities. `learning-probe.json` records numerical results.

## What is measured

Three 0.1 ms events use a 0.005-to-0.015 ramp, a 0.03-to-0.01 ramp and alternating
0.008/0.002 features. The input matrix must have rank three, preventing a
proportional-input direction comparison from passing trivially. The objective
is summed squared soma voltage. This is a diagnostic of encoder-to-cable temporal
credit, not an ARC task loss. `chunked_online_param_gradients` uses one-event
windows and a compiled scan. Sparse pp-prop at decays 0.2 and 0.8 must produce
finite nonzero and distinguishable encoder gradients.

A separate BPTT trajectory supplies the reference. Its directional derivative
is checked with a centered finite difference at epsilon 1e-5 along the normalized
decay-0.8 gradient. Approximate/reference cosine and relative error are reported;
the gate requires positive alignment and relative error below one, rather than
exact equality for the approximate rule. A normalized encoder update
of length 1e-4 must lower the objective. Every evaluation starts with a complete
physical/RNG reset. The probe restores the encoder and resets session eligibility
after collecting the result.

The parameter view deliberately requests only encoder gradients. The first
attempt requested all session parameters, including readout parameters absent
from the soma objective, and correctly failed BrainState's state-trace check.
The diagnostic now selects the actual subject of differentiation while retaining
all cable and chemical dynamics. State-trace checks remain enabled.

## Recorded result

The final rank-three diagnostic passed. The retained output was produced by
the evidence test's fresh session through the same `main` entry point, then
copied only after its probe-source hash was verified. Its elapsed time was
85.91 seconds under the affected test gate, not an isolated runtime benchmark.

| Quantity | Result |
| --- | ---: |
| Physical trajectory per objective | 60 ticks / 0.3 ms |
| Decay-dependent gradient difference | 8.1719% |
| Decay-0.8 relative error against BPTT | 2.7917% |
| Gradient cosine against BPTT | 0.999972 |
| BPTT / finite-difference directional relative error | 9.24e-10 |
| Diagnostic objective before update | 21966.029052 |
| Diagnostic objective after update | 21964.048232 |
| Extracellular K change norm | 1.297e-5 mM |
| Glutamate released | 0 |

The absence of release in this short trajectory means this learning gate does
not test event-triggered transmitter gradients. Existing separate forward tests
cover evoked GABA/glutamate release and calcium responses. The first exploratory
run used proportional inputs; only the final independent-pattern result is
retained here.

The affected CPU gate passed **18 tests in 326.06 seconds**, covering both new
diagnostics, the earlier chemical-learning test, coupled release/replay tests
and full-session construction/checkpoint regressions. Probe statement coverage
is **91/92 (98.91%)**. Ten existing unused-readout-state warnings remain visible;
this is not a warning-free, GPU or full-repository gate. The initial test
collection also exposed a docs namespace import mismatch; the sibling test now
uses its absolute package import. `verification-summary.json` pins the final
source and artifact bytes.

## Boundaries and next tests

This gate does not qualify recurrent-contact gradients, ARC accuracy, the Muon
optimizer, gradients across discrete stochastic release, measured extracellular
placement, long calcium waves, GPU execution or full-population physiology.
Physical validity and changing extracellular K establish an evolving coupled
forward state; they do not isolate how much each glial pathway contributes to
the gradient. Follow-up tests should add matched pathway controls, source-backed
placements, recurrent contacts and longer spatial/time refinement within the
authorized run limits.
