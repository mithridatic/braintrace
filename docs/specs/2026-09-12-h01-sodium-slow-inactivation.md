# SP16. Both cells: slow sodium inactivation, the shared-function test

Status: APPROVED 2026-09-12 (user: "sure why not do the new sodium mechanism file -
approved"); AMENDED 2026-09-12 after the review against Hartshorne (user: "Go") and
before any run. Mechanism written and unit-tested:
`docs/evidence/h01_sodium_slow_inactivation.py` patches the pinned `NaTs` (E) and `NaTg`
(I) sources, both Colbert and Pan 2002, both shared by the two donor fits.

## What the search has established

SP15 stage 0 named two contrasts, in units of each human's own repeat spread:

| contrast | E | I |
| --- | ---: | ---: |
| spike-1 upstroke | 177 sigma | within 1 sigma |
| threshold climb along the train | 8 sigma | 10 sigma |
| post-spike level | 7 sigma | 28 sigma |

SP15 stages 1 to 5 closed the input side for the elemental contrast: the cable moves the
upstroke at about -254 V/s per unit of whole-cell conductance ratio, the recorded cell's
own load is 1.13 times the model's, and at that load the fit stops firing at 200 pA. The
function branch carries the rest, and the use-dependent element it lacks is shared by
both cells.

## What the cyclical family looks like, cycle by cycle (SP15 stage 6, no run)

The review found that stage 0 had compressed the cyclical family to one number per train
(threshold first to last), which Hartshorne chapter 5 warns loses the answer. Read spike
by spike from the retained traces
([tables](../evidence/h01-topographic/cycle-tables.md)):

| | E human, 230 to 350 pA (one sweep per drive) | I human |
| --- | --- | --- |
| threshold | steps +1.2 to +3.6 mV between spike 1 and spike 2 (inside 11 to 174 ms, the interval), keeps moving through spike 5, then holds for the rest of the second at every drive | 0.27 nA: +4 mV over the first five spikes (43 ms) then holds; 0.19 nA: climbs across the whole second as the intervals lengthen |
| maximum rise | spike 1 at 345 to 352 V/s in every sweep; spike 2 at 0.86 to 0.93 of it; then a further 5 percent drift over the second | 602 to 550 V/s over five spikes then holds (0.27); 597 to 452 gradually (0.19) |
| recovery | spike 1 the same in all eight sweeps (rise sd 2.7 V/s); at 200 pA one spike then no re-fire for the remaining 800 ms, and no spike at all in 2 of 7 repeats | spike 1 the same in all four repeats |
| fits | threshold and rise flat at every drive (E -57.2 +/- 0.1 mV; I -60.5) | flat |

So the use-dependence **enters within the first interspike interval, does not recover
within the 800 ms hold after a single spike, and has recovered by the next sweep.** One
gate with a single one-second time constant accumulates gradually and cannot make the step;
the spec as first written would have read a wrong-shaped monotone climb as a pass.

## Hypothesis

The sodium channel equations both fits inherit have fast inactivation only. A slow
inactivation gate is the minimal missing element of the shared function, with the time
course the recordings fix: fast entry while the membrane is above the gate's half-point
(during the spike) and slow recovery below it. Its signature is specific:

1. it leaves the first spike alone, because the gate is shut below threshold and moves
   only during the spike;
2. it steps the threshold up and the rise down at spike 2 and then holds, at every drive;
3. it removes the later spikes at 200 pA, where the recorded cell fires once and holds,
   because 800 ms is short against recovery;
4. it recovers between sweeps, so it cannot change the subthreshold rows or spike 1.

Point 1 is what distinguishes it from the elemental upstroke contrast, which it is
explicitly **not** predicted to fix.

## The mechanism

One extra gate multiplying the conductance, `g = gbar*m*m*m*h*s`, with

    sInf = 1 - slow_inactivation/(1 + exp(-(v - s_vhalf)/s_slope))
    sTau = s_tau_recovery_ms + (s_tau_entry_ms - s_tau_recovery_ms)/(1 + exp(-(v - s_vhalf)/s_slope))

Defaults `slow_inactivation = 0`, `s_vhalf = -50 mV`, `s_slope = 6 mV`,
`s_tau_entry_ms = 10`, `s_tau_recovery_ms = 1000`. At the default depth `sInf` is
identically one, so the patched mechanism is the source mechanism; that is what makes
the stage-0 reproduction check meaningful. The half-point sits above the 200 pA plateau
(-65 mV): at depth 0.6 the steady state there removes under 5 percent, and with the
recovery time constant in force below the half-point the gate has not moved when the first
spike arrives. The first spec's `s_vhalf = -60` removed 12 percent of the sodium at
-65 mV at depth 0.4, which contradicted point 4; that is why the half-point moved. The
patch is pinned to the two source digests and refuses anything else, and the unit tests
check that only the conductance line and the RANGE line are rewritten.

Doses are set with the drivers' `--mechanism-parameter MECH:REGION:NAME:VALUE` flag on
every region that carries the mechanism (E: `NaTs` in soma and axon; I: `NaTg` in all),
so no candidate flag but the dose changes between the reproduction and the doses.

## Decision limits

Human repeat spread (SP15 stage 0): E threshold 0.25 mV, rise 1.5 V/s; I threshold
0.38 mV, rise 5 V/s. A threshold change counts only above four times that: **1.0 mV (E),
1.5 mV (I)**. The fit's own qualified re-reads of one reference differ by up to 3.5
percent in rise (SP15 stage 1 control re-read), so a spike-1 rise is "unchanged" inside
**3.5 percent** of its reference and a first-spike time inside the human's 23 ms. Every rise
is measured on the uniform 0.02 ms grid. Both tiers are reported at every stage.

## Stages (eight evaluations)

**Stage 0, reproduction (two evaluations).** Both libraries rebuilt with the patched
sodium mechanism; the unchanged candidates rerun at depth 0. E at 310 pA: count 10,
spike-1 rise 639.1 V/s within 3.5 percent, threshold -57.2 mV on every spike within
0.1 mV. I at 0.19 nA: count 14, spike-1 rise 592 V/s within 3.5 percent, threshold
-60.5 mV flat. Rejection: any difference beyond those means the patch changed the
mechanism, and no dose is run.

**Stage 1, the E dose scan (four evaluations).** Depths 0.3 and 0.6 at 310 pA and at
200 pA, defaults otherwise. Predictions, per cycle:

- 310 pA: spike-1 rise and first-spike time unchanged; the threshold step from spike 1 to
  spike 2 is at least +1.0 mV and grows with depth towards the recorded +1.4 mV (sweep
  53), with the spike-2 rise falling towards 0.86 of spike 1; from spike 5 on the
  threshold holds (spike 10 minus spike 5 within 1.0 mV); count 9 or 10.
- 200 pA: spike-1 rise and first-spike time unchanged; the count falls from 4 towards 1
  (at most 3 at depth 0.6) and the post-spike level moves from -65.0 towards the recorded
  -67.1 mV.

Rejection: spike-1 rise or first-spike time moves at either input (the gate acts before
the spike); the count at 310 pA falls below 9; the threshold accumulates instead of
holding (spike 10 minus spike 5 above 1.0 mV); or no step of 1.0 mV appears at depth 0.6.

**Stage 2, the I cell (two evaluations).** The surviving depth on the PV finalist at 0.19
and 0.27 nA. Predictions: at 0.27 nA the threshold rises by at least 1.5 mV by spike 5
and then holds, the rise falls towards 0.91 of spike 1, spike 1 unchanged, count within
10 percent of the finalist's 37; at 0.19 nA the threshold rises by at least 1.5 mV by the
last spike, spike 1 unchanged, count within 10 percent of 14. Rejection: spike 1 moves,
the counts leave those bands, or the threshold does not rise.

The reading: one change applied to both fits must move both humans' shared signature
with the recorded shape, or the shared-function explanation fails. Sweeps 54 and 48 stay
sealed; a pass is a mechanism result for the two fits, not a promotion. What will not
work, stated in advance: a single time constant, at any depth, because it cannot both
step inside one interval and hold for a second.
