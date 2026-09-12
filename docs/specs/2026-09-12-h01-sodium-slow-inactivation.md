# SP16. Both cells: slow sodium inactivation, the shared-function test

Status: APPROVED 2026-09-12 (user: "sure why not do the new sodium mechanism file -
approved"). Mechanism written and unit-tested before any run:
`docs/evidence/h01_sodium_slow_inactivation.py` patches the pinned `NaTs` (E) and `NaTg`
(I) sources, both Colbert and Pan 2002, both shared by the two donor fits.

## What the search has established

SP15 stage 0 named two contrasts, in units of each human's own repeat spread:

| contrast | E | I |
| --- | ---: | ---: |
| spike-1 upstroke | 177 sigma | within 1 sigma |
| threshold climb along the train | 8 sigma | 10 sigma |
| post-spike level | 7 sigma | 28 sigma |

Both humans climb their threshold along the train and both models hold it flat, and the I
human's rise falls along its train while the model's does not. Those are the cyclical
contrasts, and they are shared by two different cells fitted from two different donors,
which places them in what the fits share rather than in per-cell anatomy.

SP15 stages 1 to 4 tested the input side of the isolation split for the elemental
contrast. The cable load does move the upstroke, at about -254 V/s per unit of whole-cell
conductance ratio, but the human's own load is only 1.13 times the model's, which covers
about a tenth of the upstroke gap. Adding enough load to move it further destroys the
protected spike count. So the input branch is real and small; the function branch carries
the rest.

## Hypothesis

The sodium channel equations both fits inherit are rat and bullfrog templates with fast
inactivation only. Real cortical sodium channels also inactivate slowly, over hundreds of
milliseconds to seconds, and recover at rest. A slow inactivation gate is therefore the
minimal missing element of the shared function, and its signature is specific:

1. it must leave the first spike alone, because at rest the gate is open;
2. it must grow the threshold along the train;
3. it must reduce the rise of later spikes while sparing the first;
4. it must recover between sweeps, so it cannot change the subthreshold rows.

Point 1 is what distinguishes it from the elemental upstroke contrast, which it is
explicitly **not** predicted to fix.

## The mechanism

One extra gate multiplying the conductance, `g = gbar*m*m*m*h*s`, with

    sInf = 1 - slow_inactivation/(1 + exp(-(v - s_vhalf)/s_slope))
    sTau = s_tau_ms

Defaults `slow_inactivation = 0`, `s_vhalf = -60 mV`, `s_slope = 6 mV`,
`s_tau_ms = 1000 ms`. At the default depth `sInf` is identically one, so the patched
mechanism is the source mechanism; that is what makes the stage-0 reproduction check
meaningful. The patch is pinned to the two source digests and refuses anything else, and
the unit tests check that only the conductance line and the RANGE line are rewritten.

## Stages

**Stage 0, reproduction (one evaluation).** The library is rebuilt with the patched
sodium mechanism and the unchanged B3 candidate is rerun at 310 pA with
`slow_inactivation = 0`. Prediction: the count is 10 and the spike-1 maximum rise is
639.1 V/s, both reproducing the retained fine-mesh reference; every rise is measured on a
uniform grid. Rejection: any difference beyond the campaign's transfer limits means the
patch changed the mechanism and no dose is run.

**Stage 1, the E dose scan (two evaluations).** Depths 0.2 and 0.4 at the default
half-point and time constant, at 310 pA. Prediction: the threshold climb grows towards the
human's +2.1 mV monotonically with depth; the spike-1 upstroke stays within its repeat
band of 639.1 V/s; the count stays at 9 or 10. Rejection: the count falls below 9, or the
spike-1 upstroke moves outside its band, either of which means the gate is acting on the
first spike and not only along the train.

**Stage 2, the I cell (two evaluations).** The surviving depth on the PV finalist at 0.19
and 0.27 nA. Prediction: the threshold climb grows towards the human's +4.8 mV, the late
rise falls towards the human's 450 V/s at 0.19 nA, the first spike is unchanged, and the
counts stay within the campaign's existing bands. Rejection: the counts leave those bands,
or the climb does not grow.

The reading is the dissection the strategy registered: the same single change is applied
to both cells and must move both humans' shared signature, or the shared-function
explanation fails. Both tiers reported at every stage; sweeps 54 and 48 stay sealed; a
pass is a mechanism result for the two fits, not a promotion.
