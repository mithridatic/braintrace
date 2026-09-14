# SP15 (proposed). H01 cells: topographic re-characterisation before any further lever

Status: DRAFT 2026-09-12, written after reading Hartshorne, *Diagnosing Performance and
Reliability* (introduction and chapters 1-7) against the campaign record. Not approved,
not run. Stage 0 needs no simulation. Search tree:
[h01-reasoning/h01-search-tree.svg](../evidence/h01-reasoning/h01-search-tree.svg).

## What the book prescribes, in one page

1. A causal explanation states the necessary and sufficient conditions and the how-why
   mechanism. A root cause, a list of levers, or a fitted model is not one.
2. Topographic diagnosis answers "what is happening", not "what is wrong". It is a
   progressive search: each pass is a binary split of the remaining space, phrased so
   that both branches together cover everything not yet eliminated.
3. Four natural splits: Matryoshka (elemental, cyclical, structural, temporal families
   of variation), Isolation (inputs against function, in series), Dissection (structural
   half-split, in parallel), and the z-strategy (the device as a source-load energy
   network, split by moving the observation point).
4. The Y must be information-rich: characterise above the OK/NOK line, keep the
   spatio-temporal framework of one cycle, and never reduce a picture to a number when
   the number loses what the picture shows. Counts, means, medians and ranges are lossy
   transforms. Energetic function needs a conjugate pair (effort with flow, or effort
   with displacement).
5. Sparsity of effects: one steep X carries most of the variation; the families combine
   by root-sum-square. Search for the steep X only, with contrasts larger than 3 to 4
   sigma of the repeat variation of Y. Small stratified samples (three) suffice.
6. Observe only Y. Do not name suspects (X variables) before the split says where they
   live. A statistically designed experiment on named factors is the finishing point,
   not the start.
7. Cartoon the framework first, keep a notebook, walk the gemba, fail fast, hours not
   weeks.

## The campaign scored against the book

Kept: single-change isolation with matched controls (SP10); pre-registered predictions,
rejections and caps; the pre-spike identity check as a reproduction test (SP12/SP13);
sealed holdouts; the soma current recording as an energetic characterisation of the
model (SP11); the Vast executor for hour-scale turnaround.

Broken:

- **Y was a lossy transform.** Counts, window medians and an offset divided by an input
  resistance sized every dose since SP11. The book calls this characterising below the
  line and then fitting a black-box model to it. The parallel direct-observation audit
  of 2026-09-11 made the same point independently.
- **Levers before splits.** Nap, Im, SK, Ih, leak and KsAHP were named and dosed one at
  a time. The book's rule is to locate the family and the input-or-function branch
  first, from Y alone, then name a mechanism once.
- **No characterisation of the human as an energy network.** The model's soma balance
  was measured; the human's whole-cell net current as a function of voltage and time
  after a spike (the conjugate pair the book asks for) was never computed, although the
  retained traces allow it.
- **Contrasts not referred to the repeat variation.** The human's five 200 pA repeats
  fire exactly one spike each, with the first spike at 153 to 209 ms (57 ms spread),
  threshold -54.5 to -55.2 mV and AHP -69.5 to -70.2 mV. The count contrast (4 against
  1) and the post-spike level contrast (model -62 to -65 mV against -67 mV) are large
  against that spread. B3's 78 ms early first spike is about 1.4 times the human's own
  spread and is not a steep X to chase.
- **The structural family was read as a curiosity.** Both human cells (a PV interneuron
  and an L2/3 pyramidal cell) show the same pattern against their donor models: higher
  rheobase, steeper late gain, a threshold that climbs along the train, and a post-spike
  slowing that does not persist into a fast train. Two different cells sharing one
  deviation from two different fits is the book's strongest kind of contrast: the steep X
  lives in what the two models share (the Allen-derived channel kinetics), not in what
  differs between them (morphology, per-cell densities).

## The search tree

**Q1 (Matryoshka, retained data, no runs).** In which family does the human-model
contrast live, measured against the repeat envelope?

- Elemental: within one spike cycle. Phase-plane (dV/dt against V) of cycles 1, 2, the
  middle and the last cycle, human against model, per input. The post-spike trajectory
  of cycle 1 at 200 pA. Prediction: the contrast lives here (cycle 1 trajectory at low
  input; threshold and width along the train at high input).
- Cyclical: spike to spike within a sweep. Prediction: the human's threshold climb and
  second-spike broadening are cyclical signatures the models lack; the I cell's early
  burst is a cyclical signature the model lacks.
- Structural: cell to cell. Prediction: E and I humans share the signature; E and I models
  share its absence.
- Temporal: sweep to sweep. Known small (12.8 ms cycle, 1 mV). Not the family.

**Q2 (Isolation).** Does the contrast come in on the inputs (anatomy, passive cable,
per-cell densities) or live in the function (shared active kinetics)? Evidence in hand:
at 110 pA without a spike the model matches the human within 1 mV (inputs are not the
steep X below threshold); the difference appears only after a spike (use-dependent,
function); both cells share it (function common to both fits). Tactic to close the split:
the model's inputs can be swapped while the function is held (the same genome on the H01
morphology and on the donor morphology, one input each); if the signature stays, it
follows the function.

**Q3 (z-strategy).** The pipette is a flow source (Norton); the cell is the load. From the
retained traces compute the human's net membrane current I_net(t) = I_inj(t) - C dV/dt
(C from the onset transient at 110 pA) and plot it against V through the interspike
trajectory of cycle 1 at 200 pA and of the late cycles at 310 pA, human over model. Where
the two load curves separate, in voltage and in time after the spike, is the located
mechanism: a voltage-dependent branch separates in V, a use-dependent branch separates in
time after the spike. This replaces dose-sizing from medians.

**Only then, one mechanism.** If Q1 to Q3 point to a use-dependent reduction of inward
current shared by both cells (the climbing threshold says sodium availability, not an
added potassium brake), test it once as a 2x2 dissection: {E, I} x {kinetics as fitted,
kinetics with slow sodium inactivation}. If the signature moves toward both humans with
one change, it follows the function. If it moves for one cell only, the shared pattern was
a coincidence and the search returns to per-cell inputs.

## Policies for every step

Contrasts larger than 3 sigma of the human repeat envelope only. Samples of three sweeps
where repeats exist. Y only; no named lever before its split. Every split written in the
tree before its data are collected; prediction and rejection registered; fail fast under
the cap; both tiers reported; sweep 54 and sweep 48 stay sealed. SP14 (brake plus gain
lever) is deferred: it names two levers before Q1 to Q3 are answered.

## Stage 1 addendum (registered 2026-09-12 after stage 0, before the run)

Stage 0 named the E spike upstroke (332 against 598 V/s, 177 sigma at every cycle) as the
largest elemental contrast, with equal onset capacitance. Q2 for it: hold the function
(the B3 genome and every B3 flag) and change the input (the cable). The H01 L2 pyramidal
skeleton 955432427 (proofread104 component 0, 2460 um of cable) replaces the Allen
reconstruction; the B3 soma sphere is held so that only the cable differs. Both runs use
nseg factor 1 (the H01 skeleton has ten times the nodes), so the donor-anatomy run is the
mesh control against the nseg-9 value. Manifest
[h01-e-morphology-manifest.json](../evidence/h01-e-morphology-manifest.json); converter
and scorer `h01_e_morphology_swap.py`.

Prediction: mesh control within 10 percent of 597.7 V/s. Function branch if the
H01-anatomy upstroke stays within max(3 x mesh deviation, 10 percent) of the control;
inputs branch if it moves by more. Rejection: mesh control outside 10 percent (no
reading), or no spike on the H01 anatomy (a count contrast of the input, recorded, no
upstroke reading). One evaluation.

## Stage 2 addendum (registered 2026-09-12 after stage 1, before the run)

Stage 1 met its registered no-reading outcome: the H01 skeleton is so large a load that the
cell stops spiking (onset capacitance 785 against 125 pF), so the upstroke could not be
compared. The book's rule is that a split must keep Y observable, so the same isolation
question is asked with a graded input change on the donor anatomy: the dendritic and apical
membrane area (capacitance, leak and the distributed Ih density together) is scaled 1.5, 2
and 3 times at fixed geometry and with every soma mechanism held. Driver flag
`--membrane-area-factor REGION:FACTOR`; manifest
[h01-e-cable-load-manifest.json](../evidence/h01-e-cable-load-manifest.json); scorer
`h01_e_morphology_swap.py grade`. The x1 point is the stage-1 control run (653.3 V/s,
count 4, first spike 126.5 ms, threshold -57.4 mV).

Prediction: function branch if the spike-1 maximum rise stays within 166 V/s of 653 V/s at
every dose that still spikes; inputs branch if it falls monotonically and leaves that band
while the cell is still spiking. Rejection: every dose that leaves the band also fails to
spike (a repeat of the stage-1 no-reading case), or the departure is not monotone across
the three doses (the dose is not acting through the cable load). Three evaluations.

## Stages 3 to 5 addendum (registered 2026-09-12)

Stage 2 met its no-reading rejection again: at 200 pA every dose silenced the cell, because
B3 sits just above its rheobase there. The doses did act on the cable load exactly as
intended and monotonically (onset capacitance 125, 153, 177, 214 pF; input resistance 95,
75, 60, 45 MOhm), so the cable load is a very steep rheobase lever, steeper than any
channel dose this campaign has tried.

Stage 3 asked the same question at 310 pA, where model and human both fire ten spikes, but
used the coarse mesh for speed. Its control was invalid: at that mesh the 310 pA response
is 72 spikes and then silence, against ten steady spikes at the fine mesh. The registered
band had tested one number (the spike-1 maximum rise, 9 percent deviation) and so passed a
control that does not reproduce the response. The scorer now takes the reference spike
count and refuses to read a series whose control misses it. Stage 4 repeats the series at
the fine mesh.

Stage 5 was registered before stage 4 was read, because the decisive quantity is not the
dose but the human's own load. Measured from the 110 pA sweep, where neither cell spikes:
the human's input resistance is 80.4 MOhm against B3's 90.9, and its onset capacitance
128 pF against 124, so the human is 1.13 times B3's conductance and 1.03 times its
capacitance. That is the observable range of this input, and the dose-to-load map puts it
at a x1.25 area dose. Stage 5 sets the model's cable load to the human's and asks whether
the upstroke follows: within 10 percent of the human's 347.7 V/s means the input branch,
within 10 percent of the model's 639.1 V/s means the function branch.

## Registered stage 0 (no simulation)

Produce, from retained traces: the phase-plane small multiples (rows: cycle 1, 2, middle,
last; columns: input) for E and I, human over model, with the repeat envelope; the
I_net(V) load curves of Q3; and the Matryoshka table naming the family that carries the
largest contrast in units of the repeat sigma. Decision rule: the family and the branch
(input or function) with the largest contrast are named; no lever is named. Cost: one
analysis script and its test; hours.

## Stage 6 erratum and stage 7 addendum (registered 2026-09-12, no simulation)

Stage 6 tabulated the E staircase as sweeps 48 to 55. Sweep 54 (330 pA) is a sealed
holdout; its rows were tabulated and are withdrawn from every reading below and from the
regenerated cycle tables. Stage 6's readings do not change without it.

Stage 7 asks the elemental question the cycle tables raised: is the take-off a fixed
voltage, or is it set by the trajectory that reaches it? Y only, no lever. From the
retained traces:

* **Take-off against approach.** For every spike of every retained train, the threshold
  (campaign definition, last 10 V/s crossing before the maximum rise, on the recording's
  own 0.02 ms grid) and the approach rate (least-squares slope of the voltage over the
  window 10 to 2 ms before that threshold). A second threshold definition, the first 10 V/s
  crossing on the way up, is tabulated beside it so that the reading cannot depend on the
  definition. Both humans and both fits; E spike 1 across the staircase (48 to 53, 55) and
  the 200 pA repeats (56, 59 to 62); the 1260 pA, 3 ms short squares (27 to 31) as a
  second, fast approach; I spike 1 at 0.19 and 0.27 nA and the 120 pA repeats (40 to 43).
* **The twins.** The seven 200 pA repeats (56 to 62), five with one spike and two with
  none, compared on the level in 1820 to 2020 ms and after the pulse, as the registered
  reading of what a spike leaves in the level 800 ms later.
* **The state space.** The subthreshold staircase (sweeps 32 to 47, -110 to +190 pA):
  rest, steady level, sag or overshoot extreme in the first 300 ms, post-pulse extreme,
  and the input resistance fitted within each holding-current group (sweeps 32 to 41 at
  +2.5 pA, 42 to 47 at -3.7 pA; the 6.2 pA step between the groups is 0.5 mV at 80 MOhm
  and is not fitted across). Reported as an observation; no decision rule.

Decision limits are the E recording's repeat spread of 0.25 mV in threshold (3 sigma =
0.75 mV) for E; the I repeat spread of 0.38 mV comes from single-spike 120 pA sweeps and
qualifies spike 1 only, so the I along-train rows are reported without a sigma.

Predictions, registered before the module is run: (1) the E human spike-1 take-off moves
by more than 3 sigma across the staircase's range of approach rates, in the same direction
under both threshold definitions; (2) each fit's take-off moves by less than 1 sigma across
a wider range of approach rates; (3) the human's later spikes sit above the spike-1
relation at the same approach rate by more than 3 sigma; (4) the twins' 1820 to 2020 ms
levels agree within 3 sigma of the level spread (0.9 mV) between the sweeps with and
without a spike. Rejections: the two threshold definitions disagree in the sign of the
slide (no reading; the definition, not the cell, is being measured); the human spike-1
slide is below 3 sigma (the trajectory reading is withdrawn); the twins differ by more
than 3 sigma (a spike does leave something in the level, and the Y4 level row stands as
written). Module `h01_topographic_threshold.py` and its test; outputs under
`docs/evidence/h01-topographic/threshold-approach.{json,md,png}`. Cost: one script; hours.

The slide and the twins were first seen in scratch scripts during the SP16 close-out;
stage 7 fixes the definitions, the limits and the record before those readings are used.

Definition amendments made while the module was written, before any reading was used: the
10 to 2 ms window contained the previous spike in cycles shorter than 12 ms and lay before
the pulse under the 3 ms short squares, so the approach is the slope over at most 10 ms
before the threshold, ending 0.5 ms before it and never before the previous trough or the
pulse onset; the first-10 V/s-crossing definition is not applicable where the approach
itself exceeds 5 V/s (the short squares), and the sign check uses the long pulses; the fits
are read across every spike of every train (two trains give two spike-1 points); prediction
3 is read on the median residual with the minimum and the count below the relation
reported beside it; prediction 2's "wider range of approach rates" is judged against the
human's long-pulse range (the short squares have no second-definition reading and no model
counterpart); the verdict is per cell.

Result: E PASS on every check (slide -2.2 mV, 9 sigma, both definitions; short squares
-61.9 mV at 7 mV/ms; fit 0.2 mV over 0.23 to 1.16 mV/ms; later spikes +1.9 mV median,
none below; twins +0.16 mV, 0.5 sigma). I: the human slides -4.0 mV (10 sigma, both
definitions), but the fit moves 0.9 mV across its trains (2.5 sigma, not fixed to 1 sigma)
and the later spikes sit +1.0 mV above the relation (2.6 sigma), so prediction 2 and 3 fail
for I and the I reading is partial. [threshold-approach.md](../evidence/h01-topographic/threshold-approach.md).

## Stage 8 (registered, not run): soft or hard take-off

If stage 7 reads a trajectory-set take-off in the recording and a fixed one in the fits,
the next split is an isolation, phrased in Y: at the onset of each spike, does the
somatic phase plane show a gradual take-off (the local membrane turning over) or a kink
(a current arriving from elsewhere)? Both recordings, both fits, at the fit's soma and at
its axon initial segment (the observation point moved, book chapter 6). No lever until
this is read.

## Stage 8 addendum (registered 2026-09-12 before the module is run; no simulation)

Both fits' retained traces already carry an axonal observation point (E: `axon[1](0.5)`,
45 um out on the 60 um stub; I: `axon[0](0.5)`, the first axon section), so stage 8 needs
no run. Definitions, fixed here: every trace on the uniform 0.02 ms grid; the rate of rise
is a central difference over 0.1 ms (five samples), the same for recording and model, so the
recording's 5 V/s sample noise does not set the crossing; for each spike the onset is read
between the first 10 V/s crossing on the way up and the 100 V/s crossing:

* **span**: the voltage covered from 10 to 100 V/s, mV. A kink covers under 1 mV; a
  gradual, locally regenerative take-off covers more than 2 mV; between is unresolved.
* **rapidness**: the least-squares slope of dV/dt against V from 10 to 50 V/s, per ms
  (the phase-plane onset slope).
* **axon lead** (fits only): the axonal 10 V/s crossing minus the somatic one, ms, and the
  axonal voltage at the moment the soma crosses 10 V/s.

Spike 1 of every retained train and every later spike; both recordings at the soma; both
fits at the soma and at the axon point. Repeat spread from the E 200 pA repeats (56, 59 to
62) and the I 120 pA repeats (40 to 43), spike 1.

Predictions: (1) each fit's somatic onset is a kink (span under 1 mV) at every spike, and
its axon point crosses 10 V/s before the soma at every spike; (2) the E recording's spike-1
onset covers more than 2 mV, i.e. is gradual, and differs from the E fit's by more than
3 sigma of the repeat spread; (3) the I recording's spike-1 onset also differs from its
fit's by more than 3 sigma. Readings, written before the numbers: recording gradual and fit
kink means the recorded take-off is made locally and the fit's initiation site is the input
that has to change (the next split is the AIS as input: its sodium density, its distance,
its coupling); recording kink and fit kink means the recorded take-off is also imposed from
an initiation site, and that site itself answers to the approach, so the axonal sodium and
its neighbours are the family to split; recording kink and fit gradual is not expected and
would be a no-reading. Rejections: the span is unresolved (1 to 2 mV) in either recording;
or the recording's own repeats spread by more than 1 mV in span (the metric cannot resolve
the contrast). Module `h01_topographic_onset.py` and its test; outputs
`docs/evidence/h01-topographic/onset-shape.{json,md,png}`. Cost: one script; hours.

Result (2026-09-12): prediction 1 is refuted for both fits. E: recording and fit have the
same somatic onset, 5.0 against 5.1 mV from 10 to 100 V/s (0.4 sigma; rapidness 34 against
41 per ms), both gradual, although the fit's axon point crosses 10 V/s 0.28 ms before the
soma at all 22 spikes and is at -40 mV when the soma starts. Both-gradual was not a
registered reading: recorded as no discrimination by onset shape. Added from the same
traces: the E fit's take-off at the axon point is -54.3 to -55.2 mV at every spike, and
under the SP16 gate at depth 0.6 it climbs 2.1 mV by spike 10 while the soma's climbs 1.5,
lead unchanged; the fit's take-off is set at the axon point and answers to availability by
2 mV per 40 percent. I: the recording's onset is sharp, 2.7 mV (rapidness 52 per ms), the
fit's a 14.2 mV turnover (5 per ms), 20 sigma of the four-repeat spread and ten times its
range. The registered rejection fired for I (repeat span range 1.16 mV over 1 mV), so the I
reading is PROVISIONAL: it stands on the ratio of the spans (5.2) rather than on the absolute
bands, which were written for the E cell and are not applied to I, and stage 9's I probe run
is its confirmation. Under that caveat: the fit's soma and first axon section rise together
(spike 1 within 0.06 ms; later spikes soma first by up to 0.18 ms), so the fit has no sharp
initiation; the earlier axon-first readings were -20 mV crossings and describe the upstroke.
The E axon point is 45 um out on the stub, so "set at the axon point" means at or proximal
to it.
[onset-shape.md](../evidence/h01-topographic/onset-shape.md).

## Stage 9 (registered, not run): the take-off's own variables

Two Y questions before any lever, both from retained traces or one cheap run each:

* **E.** In the recording, is the extra 1.9 mV after a spike a function of the time since
  the spike (recovering along the interval) or a fixed step? Read the take-off of spikes 2
  and later against the preceding interval at the same approach, across the staircase. In
  the fit, does the axon point's take-off move when the approach is varied at fixed
  availability, i.e. under ramps of 0.1 to 10 mV/ms (one run per ramp family, three
  evaluations, recorded at the soma, at `axon[0](0.5)` proximal and at `axon[1](0.5)`, so
  that the initiation site is bracketed rather than assumed at the distal point)? Prediction: the fit's axon take-off moves under
  1 mV across ramps; the recording's moved 2.2 mV under the retained step pulses alone.
* **I.** Where does the fit's spike begin? Record the axon at three points (first section
  proximal, middle, distal) and the soma under 0.19 nA (one evaluation) and read the order
  of the 10 V/s crossings and the span at each point. Prediction: no point crosses more
  than 0.1 ms before the soma and every span exceeds 8 mV; the fit's initiation site is
  the input to change, and the change is registered as a 2x2 with the E cell only if the E
  question above also lands on the initiation site.

## Stage 9 execution addendum (registered 2026-09-12 before the runs)

Driver changes, both additive: the L2 driver gains `--ramp-pa-per-ms RATE`, which replaces
the recorded command from 1020 to 2020 ms by a linear current ramp `RATE x (t - 1020 ms)`
capped at 2 nA (the bias is handled as before; the sweep supplies the time base only), and
always records `axon0_voltage_mv` at `axon[0](0.5)` beside the existing `axon[1](0.5)`; the
PV driver gains `--record-axon-probes`, which records `axon[0]` at 0.1, 0.5 and 0.9 and
`axon[1](0.5)` when present. Neither changes a default output.

Ramp rates: 1.1, 11 and 110 pA/ms, chosen so that with the fit's 91 MOhm and 124 pF the
quasi-steady approach is about 0.1, 1 and 10 mV/ms; the approach that results is measured,
not assumed. Base candidate: the nseg-9 B3 reproduction of SP16 stage 0 (`s0-repro-n9`),
unchanged. E manifest `h01-e-takeoff-manifest.json` (cap 3, three inputs); I manifest
`h01-i-takeoff-manifest.json` (cap 1, the finalist at 0.19 nA with probes). Scorer
`h01_topographic_stage9.py`: (a) the retained-trace E reading (later-spike take-off residual
against the preceding interval, from the stage-7 tables); (b) the ramp runs: take-off (first
10 V/s crossing) at the soma, at `axon[0]` and at `axon[1]` against the measured approach,
and the onset span at each; (c) the I probe run: the order of the 10 V/s crossings at the
five points and the span at each.

Predictions: E (a) if the post-spike step recovers with the interval, the residual falls
with interval at fixed approach by more than 3 sigma (0.75 mV) across the retained
intervals (7 to 200 ms); if it is a fixed step, the residual is flat within 3 sigma.
E (b) the fit's take-off at every point moves less than 1 mV (4 sigma) across the three
ramps. I (c) no probe crosses 10 V/s more than 0.1 ms before the soma and every span exceeds
8 mV. Rejections: a ramp that produces no spike inside the pulse (no reading at that rate);
a probe reading in which the earliest point is `axon[1]` (the initiation lies beyond the
recorded points; the run is a no-reading and the probes move outward). Four evaluations.

Stage 9 amendment (registered before any ramp was read): the first E launch ramped to the
2 nA cap across the whole second and the 1.1 pA/ms run had not finished in 30 minutes, since
every spike under CVode at nseg 9 costs about a minute; the reading is spike 1, so each ramp
now returns to zero shortly after its expected first spike (`--ramp-end-ms` 1400, 1120 and
1060 ms; ramp maxima 0.42, 1.1 and 2 nA) and the run observes to 2100 ms at rest, because
the driver requires the observation to reach 2020 ms. The killed run is not counted against
the cap; the intermediate launch with `--stop-ms`, refused by the driver before construction,
spent one evaluation by the runner's rule (prior_evaluations 1, cap 4).

Stage 9 result (2026-09-13, `docs/evidence/h01-topographic/stage-9.md`; four evaluations
spent, one refused launch counted). (a) E recording: unresolved. The post-spike residual is +2.0, +2.2 and +1.5 mV after
intervals under 30, 30 to 100 and over 100 ms, a spread of 0.69 against the 0.75 limit and
not monotone, so the metric cannot tell a fixed step from one recovering at 0.6 mV per
decade; it does say the step does not recover measurably inside the pulse. (b) E fit under ramps: the somatic take-off is fixed (0.07 mV from
0.37 to 2.53 mV/ms) but the take-off at the axon points is not: `axon[1](0.5)` falls 4.1 mV
and `axon[0](0.5)` 2.4 mV as the approach quickens, the axon leading by 0.30 ms at every
rate and sitting 3.6, 2.8 and 0.7 mV above the soma at its own take-off. One millisecond before the
axon's take-off the soma-to-axon gradient is +1.2, +0.5 and -0.5 mV: under the slow ramp
the site leads and fires from above the soma, under the fast one it is driven and fires
from below. Prediction (b) is refuted at the axon points. The ramps confound approach with
current (0.24, 0.45 and 1.08 nA at the soma's take-off, a factor 4.4, against the
recording's factor 1.75 across its steps), so the slide is as much a function of how hard
the soma drives the site as of the approach; both readings say the same thing, that the
site's take-off depends on the source-load relation between it and the soma. The fit
therefore carries a trajectory-set take-off of the recorded sign and size at its
initiation site, hidden from its soma by the coupling; along the retained trains that
axonal take-off is a function of the approach only, so the recorded post-spike step has no
counterpart at either site. (c) I fit: no probe leads the
soma by more than 0.08 ms, every span is 8.9 mV or more; the registered rejection names
`axon[1](0.5)` as the earliest point at spike 1 by four samples, and the model's axon ends
there: the finalist's spike is a whole-cell turnover; the stage-8 I reading is confirmed on
a second run with five points.

## Stage 10 (registered 2026-09-13, not run): the coupling between the soma and the site

An input split on the E fit, every channel held. The fit's axon is the Allen replacement
stub, two sections of 30 um at 1 um diameter; the recorded L2/3 pyramidal cell's initial
segment starts on a hillock several micrometres wide. A thicker stub changes three things
at once, axial conductance (as diameter squared), membrane area (as diameter) and, if the
density is held, the total axonal sodium; so the split has three arms: x2 diameter with the
axonal NaTs density divided by the area factor (total sodium held; `NaTs:axon:3.814` becomes
1.907), x2 diameter with the density held (total sodium x2), and x4 diameter with the total
held (density 0.954). Every other flag of `s9-b3-n9` unchanged. Driver flag
`--axon-stub-diameter-um`; the stub's segment count is held at the nseg-9 value (nine per
30 um), which is finer than the d-lambda requirement at every diameter tested, since a
thicker stub is electrotonically shorter; the stage-9 runs at 1 um are the control (stage
3's lesson: a control must reproduce the response, and these reproduce the retained
step-pulse take-offs). The same three ramps, read as in stage 9 (b), for
each arm, plus the 310 pA step (sweep 53) for the two x2 arms: eleven evaluations, manifest
`h01-e-coupling-manifest.json`.

Predictions: with the x4 diameter the somatic take-off slides by more than 1 mV (4 sigma)
across the three ramps in the recorded direction (lower at the faster approach) and the
axon lead falls below 0.15 ms; with x2 it slides by more than 0.5 mV in both x2 arms, so
that the slide follows the coupling and not the density; the somatic onset span stays
within 3.5 to 5.7 mV (the recorded range); the 310 pA count stays within 2 of 10. If the
two x2 arms disagree by more than 0.5 mV, the density and not the coupling is acting and
the input branch is not confirmed. The post-spike step is not expected to appear: nothing in the fit carries history.
Rejections: a ramp or the step without a spike at a dose (the thicker stub loads the site;
no reading at that dose); the somatic take-off slides but the onset leaves the recorded
span range (the coupling changed the spike, not only its reading). If the x4 prediction
holds, the input branch is confirmed for the approach signature: the recorded soma reads
its initiation site directly, and the fit's does not; the remaining recorded element is
the fixed +1.9 mV after a spike, which needs its own split.

Execution note: the three arms ran as three runner invocations with separate output
directories (`h01-e-coupling`, `-b`, `-c`; manifests split accordingly) so that they could
run in parallel on the 255-core box; nothing else differs between the manifests.

Stage 10 result (2026-09-13, `docs/evidence/h01-topographic/stage-10.md`, eleven runs).
The somatic take-off slides with the approach in every thickened arm: control -0.07 mV,
x2 total held -2.9, x2 density held -2.1, x4 total held -3.4 mV from about 0.4 to 2.6 mV/ms
(recorded -2.2 from 0.2 to 0.75); the axon lead falls to 0.16, 0.20 and 0.04 ms; the 310 pA
counts are 10 and 9. Both registered rejections fire by the letter: the x2 arms differ by
0.8 mV (density modulates a slide that is thirty times the control's in both), and the
somatic onset span leaves 3.5 to 5.7 mV in every arm, by 0.3 mV at one ramp in the
density-held x2 arm and by 3 to 6 mV in the total-held arms, where the site merges into the
soma. The input branch is confirmed for the approach signature with the qualification that
the coupling also changes the onset, least at x2 with the density held, which is the
closest run of the campaign to the recorded take-off as a slope (-2.8 mV per decade of
approach against the recording's -3.8; the ramp and long-square ranges overlap only
between 0.45 and 0.75 mV/ms, where the arm sits about 1 mV above the recording; span 4.1
to 6.0, count 9). The split is sufficient for the approach signature but did not separate
the reading of the spike from the spike itself. No arm produces the post-spike +1.9 mV, and no arm moves the rise (579 and 655
against 639 V/s; recorded 348).

## Stage 11 (registered 2026-09-13, not run): two elements remain, split apart

The take-off's approach signature is now an input matter. Two recorded elements have no
counterpart in any run: the rise (348 against 579 to 664 V/s, 177 sigma, established at
SP15 stage 0 as somatic-sodium territory and not a cable matter at the recorded load) and
what a spike leaves in the take-off (+1.9 mV, not recovering measurably inside the pulse,
gone by the next sweep). Before either is given a lever, one Y question for each:

* **Rise.** On the x2 density-held geometry, does the rise follow the somatic sodium
  density, and does the take-off slide survive the dose? No retained slope transfers: the
  stage-4 slope (254 V/s per unit of whole-cell conductance) is a cable-load slope, the SP16
  slope (12 to 19 percent per mV) is against availability, and stage 10 has just shown the
  geometry moves the rise on its own (579 and 655 against 639). So the read is a two-dose
  slope, not an interpolation: somatic NaTs density factors 0.7 and 0.5 at 310 pA, everything
  else as the x2 density-held arm (two runs). Prediction: the rise falls monotonically with
  the density and the somatic take-off at the 310 pA onset moves less than 0.5 mV; no pass
  band on the value of the rise. Rejection: no spike, count outside 8 to 12, or the take-off
  moving more than 0.5 mV (the density is a take-off lever at this geometry, and the rise
  cannot be dosed through it without moving what stage 10 placed).
* **What a spike leaves behind.** In the recording, is the +1.9 mV present after the 3 ms
  short-square spikes as well (sweeps 27 to 31 fire one spike each; the next long-square
  sweep's spike 1 is the read), and does it depend on the number of spikes in the train
  (spike 2 residual against spike 10 residual at the same approach, from the stage-7
  tables)? Retained traces only; no run. If it is present after one spike and does not
  grow with the count, it is a per-spike step with a time constant longer than a second;
  if it grows, it accumulates.

### Stage 11 execution addendum (2026-09-13, before the runs)

**Rise.** The doses are absolute values of the driver's `--sodium-density-factor` (B3 runs
at 0.9): 0.7 and 0.5 replace 0.9, everything else copied from `s10-d2-density`
(`h01-e-rise-manifest.json`, output `h01-e-rise`, two candidates at the one 310 pA input, cap
2). The retained `s10-d2-density-sweep53` run is the third point of the same series at 0.9,
so monotonicity is judged over 0.9, 0.7, 0.5 and the take-off is compared against that run's
spike 1. Lowering the somatic sodium moves the rheobase, so at a fixed 310 pA the approach of
spike 1 will move too; the take-off is therefore read with its approach, and the 0.5 mV limit
is applied to the take-off corrected for the approach difference by this geometry's own
ramp slope (stage 10, x2 density held: -2.8 mV per decade), with the raw difference and both
approaches recorded beside it. The rise is `max_rise_v_s` of spike 1 from the campaign
landmarks on the soma trace, the same reading at all three doses and for the recording.

**What a spike leaves behind.** The read as registered cannot fire, and this is recorded as
a fired "no reading" before any trace is opened: the five short squares (sweeps 27 to 31)
are followed by sixteen subthreshold long squares, and the first long square that spikes
(48) starts 173 s after sweep 31, in the regime already established as "gone by the next
sweep"; and the short-square spikes sit at an approach of 7.2 mV/ms with the take-off at
-61.3 to -62.0 mV, where the recorded relation has slid 6 mV from the long-square regime, so
a next-long-square read would compare across regimes as well as across 173 s. The adjacent
read from the same traces, registered here in its place: sweep 27 fires after five sweeps
that did not spike (30.7 s after the last spike, sweep 21), so it is the unprimed baseline;
sweeps 28, 29 and 30 each follow a single spike by 4.15, 4.18 and 2.72 s, and sweep 31
follows one by 15.8 s, all at the same approach (7.18 to 7.30 mV/ms) and the same 1260 pA.
Prediction if the step outlasts seconds: the primed take-offs sit above the unprimed one by
about +1.9 mV. Prediction if it does not: they sit within 3 sigma of it (the sigma is the
spread of the five short squares themselves, since no other repeat at this approach
exists). The read does not reach inside the first 2.7 s, and it assumes the step is as
visible at 7 mV/ms as at 0.2 to 0.75; both limits are recorded with the result. The
second half, does the residual grow with the count, reads the stage-7 tables as registered:
residual by spike index, spike 2 against spikes 6 and later.

Stage 11 result (2026-09-13, `docs/evidence/h01-topographic/stage-11.md`, two runs, PASS).
Rise: 655, 586 and 505 V/s at somatic NaTs density 0.9, 0.7 and 0.5 (about 375 V/s per
unit of the factor, read between 0.5 and 0.9 only); the somatic take-off holds at -55.46,
-55.36 and -55.32 mV with the approach at 0.63, 0.61 and 0.59 mV/ms (corrected moves +0.07
and +0.06 mV); the count is 9 at every dose; the onset span widens 4.1, 5.0, 5.4 mV, inside
the band. No rejection fired. The recorded 348 V/s lies 157 V/s below the lowest dose; the
slope is not carried outside its range. What a spike leaves behind: the primed short
squares sit -0.16 to +0.56 mV from the unprimed one (median +0.27; inside 3 sigma of the
long-square repeat, 0.75 mV, the independent yardstick; the five's own sigma 0.28 agrees)
at 2.7 to 15.8 s after one spike, so the +1.9 mV step is gone within 2.7 s; inside a
train it is full-sized after one spike (spike 2 residual +1.83 mV, n 6) and does not grow
(spikes 6 and later +2.23, difference +0.40 mV, limit 0.75). A per-spike step, present in
full after one spike, not recovering measurably within 0.73 s and gone within 2.7 s. Limits:
the read does not reach inside the first 2.7 s, and it assumes the step is as visible at
7 mV/ms as at 0.2 to 0.75.

## Stage 12 (registered 2026-09-13, retained traces only, no run): the recording's measurement chain

A re-read of the book (ch. 4, Table 3: "observed variation is the product being measured
or the measurement process itself"; ch. 6: quantities compared at different observation
points are not the same quantity) found a split the campaign never made. Every recorded
quantity is read at the top of a patch pipette through the amplifier's bridge and
capacitance neutralisation and the acquisition filter; every model quantity is read at the
membrane. The two are one quantity only through the recording's own chain. The E file
records access resistance 7.68 MOhm, bridge balance 7.94 to 10.32 MOhm, capacitance
compensation 0 F on every long square (the I file 9.01 / 10.01 MOhm / +3.4 pF). A repeat
sigma measured through one chain (E rise 1.5 V/s) bounds that chain's repeatability, not its
bias against the model, so every "N sigma" that crosses the two chains (rise, peak, onset
span, rapidness) is mis-framed until the chain is measured; sigmas within one chain
(threshold, take-off, level) stand.

**Y and the measurement.** The chain is measured, not estimated, from the recording's own
step edges: the subthreshold 3 ms short squares (E sweeps 15 to 18 and 22 to 26 at 0.9 to
1.25 nA; I sweeps 7 to 13 at 0.1 to 0.7 nA), pulse onset at 1020.0 ms. In the first 0.4 ms
the membrane is linear (tau 10.3 / 6 ms), so the recorded response to the current step is the
chain's response to a known input: V_m(t) = (I0/C)·t (C from the stage-0 onset capacitance,
128 / 70 pF) plus the bridge term (R_s·LP[u] − R_b·u)·I0, where LP is the pipette pole
tau_p = R_s·C_p and the bridge R_b is the recorded value; then the acquisition filter (a
4-pole Bessel with corner f_c) and a delay t0. A live C_p shows as a negative transient of
amplitude R_b·I0 (9.6 mV at 1.21 nA) lasting tau_p; a neutralised one leaves only the bridge
mismatch step (R_s − R_b)·I0. Fitted jointly over every onset edge: tau_p, f_c, t0, and
R_s − R_b; tau_p is then profiled (0 to 100 us) and its bound is the largest tau_p whose
residual is within 20 percent of the minimum. The chain (tau_p, f_c) is then applied to the
model traces and every quantity re-read with the campaign's own readers.

**Cells, registered before the traces are opened.**
(a) *Chain.* tau_p bound and f_c for each cell, with the residual profile; "no reading" if
the edges do not constrain tau_p to within a factor of two of the acquisition corner.
(b) *Rise.* The fit's spike-1 rise at 310 pA (x2 density-held arm 655 V/s; B3 control 639)
through the fitted chain, and the fraction of the gap to the recorded 348 that the chain
accounts for. Upper bound independent of the fit: the tau_p at which the filtered peak
equals the recorded peak (35.3 mV) and the rise there. Prediction: the chain accounts for
less than half of the gap under the peak bound; the residual is real at more than 3 sigma
of the within-chain repeat (1.5 V/s).
(c) *Slow set held.* Through the same chain, on the x2 density-held ramps and step: somatic
take-off moves less than 0.1 mV, approach within 5 percent, axon lead within 0.02 ms, count
unchanged; the recording-only relations (interval, short-square series) are untouched by
construction. Rejection: any of these moving beyond its limit re-opens stages 7 to 11.
(d) *Onset.* Span and rapidness of the E fit through the chain against the recording's; the
I fit (finalist probes run, 0.19 nA) through the I chain against the recorded I onset (2.7
mV span, 52 per ms). Prediction: the E span stays within 1 mV of the recording's and the I
fit's span stays above 8 mV (the stage-8 I reading survives its chain); either failing is
recorded as a stage-8 erratum.
(e) *Qualification rule* for the causal model, independent of the outcome: a fast model
quantity is comparable to a recorded one only through the recording's own chain; a repeat
sigma bounds repeatability, not bias.

Unchanged: no run, no lever, holdouts sealed; the model traces are the retained ones
(`h01-e-coupling-b/s10-d2-density-*`, `h01-e-cable-load/c10-area-310-n9-sweep53`,
`h01-i-takeoff/s9-finalist-probes-019-probes`).

### Stage 12 execution addendum (2026-09-13, recorded before the second pass)

The first execution as registered, with the acquisition corner free in the edge fit,
returned f_c 2.65 kHz (E) and 3.3 kHz (I); applied to the fit that corner gave a spike-1
rise of 384 V/s against the recorded 348 and would have read the whole gap as the chain.
Two facts refute it before it is believed: the unstimulated baseline's noise floor is flat
to Nyquist in both files (-0.7 and +0.1 dB per octave between 10 and 20 kHz, against -24
for a 4-pole corner below 10 kHz), and the edge's own jump is two samples wide where a
2.65 kHz corner would spread it over seven. The step edge measures the round trip, the
command's own pole included, and a step cannot separate that pole from the output filter.
Amendment: the pipette pole is read from the transient's area (invariant to the command
pole) with the corner held; the corner is bounded from the noise floor (a reading only if
the floor is pipette noise, which cannot be told from the digitiser's floor here) and the
chain is applied at an assumed 10 kHz corner as the conservative case (no filter setting
is recorded in the file), with the 20 kHz case beside it. Cell (a) gains the noise-floor slope; the other cells stand.

Stage 12 result (2026-09-13, `docs/evidence/h01-topographic/stage-12.md`, no run,
verdict FAIL by the letter on two cells, both recorded). Chain: pipette pole 5 us (E) and
2 us (I) at best, bound 30 us (the command pole 120 / 36 us absorbs the profile); the
neutralisation is live in practice whatever the metadata field holds, since a live 5 pF
pole would put a 9 mV transient at every edge and none is there; bridge mismatch +0.24 /
-0.96 MOhm; corner at or above 20 kHz if the noise floor is pipette noise, 10 kHz applied.
Rise: the E fit's 655 V/s reads 609 at 10 kHz and 638 at 20 kHz (the control 639 reads 570
and 615), 5 to 15 percent of the gap to the recorded 348; the residual 260 to 290 V/s is
real. The peak bound (the pole at which the filtered peak meets the recorded 35.3 mV, 60 us)
would cover 55 percent: the registered "less than half" fired by the letter at that bound,
an order of magnitude above the measured pole. Third leg: the recording repolarises faster
than the fit (-104 against -91/-95 V/s) while rising slower, which no chain produces.
Slow set: the take-off shifts by a constant 0.13 mV at 10 kHz (the registered 0.1 fired by
the letter) while the slide across the ramps holds to 0.01 mV; approach within 1 percent,
axon lead within 0.02 ms, count unchanged: stages 7 to 11 stand. Onset: the E span moves
4.10 to 5.45 (10 kHz) and 5.74 (20 kHz) and non-monotonically across the ramps, so the 10
to 100 V/s span is a fragile reading at a 0.6 mV/ms approach and the stage-8 E cell gains
that qualification; the I fit's span survives its chain (12.8 against the recorded 1.9),
the stage-8 I reading stands. New: the I fit's rise, equal to the recording's when read
raw (592 against 597), reads 530 through the 10 kHz chain, so that match was a
coincidence of observation points. The qualification rule (e) is adopted.

## Stage 13 (registered 2026-09-13): the base as a sub-system, a Dissection half-split

**The frame that was never split.** Every stage so far dosed densities and geometry on
one base and asked why it does not behave like the recorded cell. The base is an input
carried into every stage. Facts: the E recording is Allen human specimen 541563728
(Homo sapiens, 28 y, L2/3 spiny); B3 is Allen model 626170538, the perisomatic fit of
that same cell, whose densities were optimised to its own sweeps under a kinetic set
(NaTs from Colbert and Pan 2002 rat, Kv3_1, K_T, K_P, SK, Ih, Im, Ca) that is fixed across
every Allen mouse and human perisomatic model, and whose optimiser never saw the rise. The
I base is Toronto HL5BN1 (ModelDB 267587), a human-fitted model with Hay-2011-lineage
kinetics and human-refit densities, and it shows the same missing threshold climb. The
three elements now on the table, the rise, the threshold climbing along a train with the
rise kept, and a repolarisation faster in the recording than in the fit, are all
kinetic-family signatures. Ch. 5: two systems with a large contrast, swap the sub-system.

**Systems.** A: B3 on its own anatomy (in hand: `h01-e-cable-load/c10-area-310-n9-sweep53`
and the stage-7 tables). B: Toronto HL23PYR, the human L2/3 pyramidal model of Yao et al.
2022 (ModelDB 267595 `biophys_HL23PYR.hoc`, byte-identical to the GPL-3.0 267587
`biophys_HL23PN1.hoc` but for the procedure name; morphology `HL23PYR.swc` from 267595;
the eleven mod files identical to the imported HL5BN1 set). Its NaTg carries human-fitted
voltage shifts (soma vshiftm 13, vshifth 15, slope 7; axon vshifth 10, slope 9), a
20 + 30 um tapered initial segment and a 1000 um myelin; it was fitted at the population
level to Krembil human L2/3 recordings, not to this cell, so it is a different cell's
model of the same class and layer, which is what the split needs.

**Execution.** The PV driver runs the Toronto template unchanged (pristine 267595 mods
compiled on the box in `human-pv/hl23-source`); one flag is added, `--stimulus-on-ms`,
so the pulse sits at 1020 to 2020 ms as in the recording and the campaign readers apply
without change. Inputs: the recorded long squares as steps, 0.2 nA with the recorded
+2.5 pA bias, 0.25 and 0.31 nA with -3.7 pA, and 0.4 nA with -3.7 pA in case the model's
rheobase lies above the recorded cell's; nseg factor 9, CVode 1e-10, 2100 ms; cap 4,
`h01-e-donor-manifest.json`, output `h01-e-donor`. No holdout is touched (sweep 54 is
0.33 nA; 0.4 is not a recorded input of this cell).

**Cells, registered before the runs.** Every fast quantity is read through the recording's
chain (stage 12, 10 kHz conservative).
(a) *Rise.* HL23PYR spike-1 rise at the lowest drive that fires, through the chain, against
the recorded 348 V/s; within 15 percent is inside the recorded band (short squares and
long squares give 336 to 351); B3 reads 570 to 615. Prediction if kinetics are the family:
inside the band.
(b) *Threshold climb.* Along a train of at least five spikes: threshold at spike 2 minus
spike 1 and spike 5 minus spike 1, cycle by cycle, against the recorded +1.4 and +3.6 mV
(B3 0.0 and 0.0); the rise of spike 5 against spike 1 against the recorded 0.86. Prediction
if kinetics are the family: climb at least half the recorded and the rise kept above 0.8.
(c) *Fall.* Spike-1 maximum fall through the chain against the recorded -104 V/s (B3 -91 to
-95).
(d) *Take-off with its approach.* Spike-1 take-off against approach across the drives that
fire, on the stage-7 relation (recording -4.3 mV per mV/ms of approach; B3 flat); read at
the soma with the qualification that the site may be tightly coupled here (a 1.75 um AIS,
not a 1 um stub).
(e) *Count and rheobase.* Counts at 0.2, 0.25, 0.31 nA against the recorded 1, 5, 10 and
B3's 4, 8, 10; a model that does not fire at 0.31 nA is read at 0.4 and its rheobase
recorded as above the cell's.
Rejections: no spike at any drive (no reading, the model is not comparable at this
cell's drives); HL23PYR missing the climb the way B3 and HL5BN1 do (then the family is not
the fitting pipeline but something both lack, and the base stays B3); HL23PYR inside the
rise band but with the count at 0.31 nA outside 5 to 15 (the rise is not bought with the
count). Second half, only if (a) and (b) hold: HL23PYR biophysics on the 541563728
anatomy (the anatomy sub-system swapped), so that the base is replaced on a 2x2 and not
on a leap.

Unchanged: the E recording, its chain, the readers, B3 and its retained runs.

Stage 13 result (2026-09-13, `docs/evidence/h01-topographic/stage-13.md`, four runs of
54 to 73 s plus one evaluation spent on a path error; verdict FAIL, three cells fired).
Through the chain HL23PYR's spike-1 rise is 592, 594, 600 and 600 V/s at 0.2, 0.25, 0.31
and 0.4 nA (B3 545 to 570; recorded 332 to 351): cell (a) fired at the lowest firing
drive and at the matched drive alike. The rise is not the fitting pipeline; two
independently fitted human L2/3 models on different anatomy agree on it, and what they
share is the sodium equation family. The threshold climbs: +0.77 / +0.90 mV (spike 2 /
spike 5) at 0.2 nA, +1.38 / +1.78 at 0.31, +2.24 / +2.36 at 0.4, with the rise kept at 0.95
(recorded +1.38 / +3.63 at 0.86; B3 -0.02 / +0.24); cell (b) fired on the lowest firing
drive by the letter and sits at the limit at the matched drive. The shape: a step in the
first interval that then holds, where the recording adds +2.2 mV more over spikes 2 to 5.
So a human-fitted base gives the step and not the accumulation, and the registered
rejection "climb missing as in B3" did not fire in its strong form. Excitability is the
wrong cell's: rest -74 mV against -84 under the same bias, counts 16 / 18 / 20 / 24
against 1 / 5 / 10 (cell (e) fired); fall -80 V/s against B3's -94 and the recorded -104.
The take-off relation is read over 1.1 to 1.9 mV/ms, disjoint from the recorded 0.2 to
0.75, and is not compared as a value; the 0.4 nA take-off reading is a reader failure at a
2.6 mV/ms approach and is excluded. The elemental picture came free: the three spike-1
loops coincide from take-off to about -40 mV and separate above it, so the fits' excess
rise is in the somatic phase, not the site-delivered one. Consequence: the base stays B3;
the second half (HL23PYR biophysics on the 541563728 anatomy) is worth one run for the
climb only; the rise now has the same value on two bases that differ in everything but
their sodium equations, and that is the sub-system the next split addresses.

## Stage 14 (registered 2026-09-13, retained trace only, no run): the rise decomposed by observation point

**Why this before the dose.** Stage 13 named "the shared sodium equations" as the cause of
the rise both bases share, but a Dissection only tells you what two sub-systems have in
common, and B3 and HL23PYR share three unquestioned inputs, not one: the sodium equation
family, the ideal current clamp and 34 degrees. Chapter 6 is emphatic that a dose is the
finishing point, not the start ("Experiments should be employed sparingly, primarily to
verify physical mechanisms once a causal explanation is at hand... DoE... is the finishing
point", printed p. 238), and that the way to localise a source is to change the observation
point (p. 230; ch. 7 summary p. 270, "the source impedance... is where the problem lives").
So before the sodium dose, one observation with no lever: decompose the fit's rise into the
currents that carry it, at the soma, and place the excess on the voltage axis. This is the
z-strategy conjugate observation the book asks for, and it converts stage 13's structural
naming into a mechanism-localised claim or refutes it.

**No run.** The observation already exists. The stage-0 retained run
`h01-e-currents/m0-b3-currents-sweep53` is the identical B3 fit (nseg factor 9, 34 degrees,
CVode) whose spike-1 rise reads 570.4 V/s through the stage-12 chain in stage 13, and it was
recorded with `--record-soma-currents`: NaTs, Nap, K_P, K_T, Kv3_1, Im, SK, Ca_HVA, Ca_LVA,
Ih and pas as local soma(0.5) densities, i_cap, and the eleven electrical neighbours of the
soma midpoint with their axial resistances and the segment geometry (area 65.8 um2, cm 1.0).
Stage 14 is re-analysis of that trace: no evaluation is spent, no holdout is touched. The
new scorer reads voltage and every current onto the campaign's 0.02 ms grid with
`load_points`, computes dV/dt with the same `rate_of_rise` estimator that produced 570.4,
and converts each density to a segment current in nA (i_nA = i_mA_cm2 x 65.8 x 1e-2) and each
neighbour to an axial current in nA ((V_neighbour - V_soma) / R_mohm).

**The observation point is chosen so that intra-soma sodium is not miscounted as axial.**
The eleven neighbours are not one branch. Two are the adjacent soma segments at R = 0.01
MOhm (near shorts): at nseg 9 the observed compartment holds about a ninth of the soma's
sodium, and the rest arrives through these intra-soma legs, so intra-soma axial current IS
somatic sodium, from neighbouring segments. One neighbour is `axon[0](0.056)` at R = 2.0
MOhm: that single leg is the site-delivered current from the initiation stub. Eight are
`apic[0]` and dendrites (R 0.15 to 4.1 MOhm): the dendritic load. The load-bearing
dichotomy is therefore **site-delivered (the axon[0] leg) against somatic-total (this
segment's own ionic sodium plus the two intra-soma axial legs)**, with the dendritic legs
reported as a sink. Lumping all axial together would read "axial dominates" for a trivially
structural reason and refuse the dose on an artefact.

**Two frames, stated.** The decomposition is at the membrane, where the balance
`i_cap = axial_in - ionic_out` holds; the contrast with the recording is through the chain,
where the recording lives (the recording is already at the pipette, the model is chained to
meet it). The -40 mV threshold (where the stage-13 loops separate) is applied in each frame
on its own voltage. The stage-11 intercept (~320 V/s, from a whole-soma density dose) is
cited as consistent in direction only, never matched as a value to these segment shares
(different observation point).

**Cells, registered before the shares are computed.** Spike 1 at 310 pA.
(d) *Closure (a hard gate; qualification, not biology).* For the soma midpoint over the
spike, `i_cap + sum(ionic) - sum(axial_in)` is within 5 percent of the peak inward current,
and the recorded i_cap reproduces C_seg x dV/dt from the voltage within the same tolerance.
If closure fails the reading is void (a unit- or sign-error detector; it lets the sign
conventions be read empirically).
(a) *The somatic-sodium share above -40 mV.* Of the model's positive dV/dt from -40 mV to
the peak, the fraction carried by somatic-total (local ionic Na + intra-soma axial) against
the fraction carried by the axon[0] site-delivered leg. Prediction if the shared sodium is
the rise's drive: somatic-total is the majority above -40 mV. Rejection/redirect: if the
axon[0] leg is the majority above -40 mV, the somatic phase is still site-delivered, the
sodium equations are not the somatic-phase drive, the parked dose is refused, and the
sub-system is the coupling/geometry (stage 10), not the channel.
(b) *The site-delivered leg below -40 mV.* From the take-off (about -57 mV) to -40 mV, the
axon[0] leg's share of the positive dV/dt. Prediction: the axon[0] leg dominates below
-40 mV, consistent in direction with the stage-11 intercept and the phase-plane coincidence
to -40 mV. This is the observation, not a pass/fail cell.
(c) *The contrast on shared pipette axes.* The recorded spike-1 dV/dt-against-V loop
(sweep 53, raw, already at the pipette) beside the fit's loop through the chain, from -40 mV
to the peak. Prediction if the somatic sodium is the lever: the recorded dV/dt above -40 mV
is roughly half the fit's and the gap widens with voltage (an inward-current deficit in the
somatic phase, not the site-delivered phase). The recording gives only the loop; the split
is the model's.
(e) *Limit, registered.* HL23PYR cannot be decomposed the same way (the Toronto driver
carries NaTg, not the L2 driver's NaTs, and no soma-current probes), so the decomposition is
B3's alone; stage 13's shared-rise result licenses reading the shared drive from one base.

**Consequence set before the run.** If (a) and (c) hold, the parked sodium dose is justified
as the finishing stage: swap or dose the shared NaTs kinetics/density as a characteristic
curve of the rise through the chain, with the count and the take-off held. If (a) fails, the
dose is refused and the coupling/geometry is the sub-system. Either way the climb's second
half (HL23PYR biophysics on the 541563728 anatomy, one run) remains available and untouched.

Unchanged: the E recording, its chain, the readers, B3 and its retained runs; no holdout.

## Stage 15 (registered 2026-09-13, before the runs): the human-for-rodent sodium swap, and the climb on this cell's anatomy

**What stage 14 earned.** The rise's excess is the soma's own sodium above -40 mV, on a human
cell, under a human-fitted set of densities -- but under Allen's fixed **rodent-lineage**
kinetic equations (NaTs, Colbert and Pan 2002 rat). That equation lineage is the last
mouse-derived component in the E model and it is exactly where stage 14 put the defect. Ch. 6
says the dose comes last, once the mechanism is localised; it is now localised, so this is the
finishing step.

**Arm A (rise): swap the sodium equations, human for rodent, and read the characteristic
curve.** System: B3 on its own human anatomy (Allen 541563728) with its own human-fitted
densities, the recorded command, nseg 9, CVode 1e-10, 34 degrees -- everything held except the
somatic sodium equations. The somatic rodent NaTs is zeroed
(`--regional-density NaTs:soma:0`, from gbar 2.641 S/cm2) and the Toronto human sodium NaTg is
inserted on the soma (`--insert-density NaTg:soma:D`) carrying HL23PYR's human voltage shifts
(`vshiftm 13`, `vshifth 15`, `slopem 7`, from `biophys_HL23PYR.hoc`). A new mechanism library
`human-sodium-source` carries the Allen mods plus `NaTg.mod` (sha256 24ad0563...) compiled
together. Dose series D = 0.068, 0.136, 0.272, 0.544 S/cm2, that is 0.25x, 0.5x, 1x and 2x
HL23PYR's own somatic value 0.272; sweep 53 (310 pA) only. Cap 5, manifest
`h01-e-human-sodium-manifest.json`, output `h01-e-human-sodium`. No holdout is touched.

**Cells, registered before the runs.** All fast quantities through the stage-12 chain.
(a) *Characteristic curve.* Spike-1 rise against the dose, monotone, with the dose at which the
curve crosses the recorded 348 V/s reported by interpolation and flagged as inside or outside
the series. This is the curve, not a pass/fail.
(b) *The rise is not bought with the count or the threshold.* At the crossing dose (or the
closest dose), the 310 pA count is within 5 to 15 (recorded 10) and the spike-1 take-off is
within the recorded band. Prediction if the equation lineage is the rise's remaining input:
some dose in the series puts the rise within 15 percent of 348 while the count holds.
(c) *Against the rodent curve.* The same reading for the rodent equations is already in hand
(stage 11: 655 / 586 / 505 V/s at somatic NaTs 0.9 / 0.7 / 0.5 with count 9; the stage-11
intercept at zero somatic sodium is about 320 V/s). The question is whether the human equations
reach 348 at a dose that holds the count, where the rodent equations reach it only by removing
nearly all somatic sodium.
(d) *Rejection.* If no dose brings the rise within 15 percent of 348, or every dose that does
also leaves the count band, then the equation lineage is not the remaining input for the rise
and the residual belongs elsewhere; that is reported as the result, not repaired.

**Arm B (climb): HL23PYR biophysics on this cell's anatomy.** The second half of the stage-13
split, one run: the Toronto human-fitted biophysics (`biophys_HL23PYR.hoc`, unchanged) on the
Allen human morphology of the recorded cell (`allen-541563728.swc`, sha256 956784de..., copied
from the L2 `source-model/morphology.swc`), driven at 0.31 nA as the recording is driven.
(e) *The step.* Threshold at spike 2 minus spike 1 against the recorded +1.38 mV (HL23PYR on
its own anatomy gave +1.38; B3 gives -0.02). If the step survives the anatomy swap it is a
property of the human sodium, not of the Toronto anatomy.
(f) *The accumulation.* Spike 5 minus spike 1 against the recorded +3.63 mV (HL23PYR on its own
anatomy gave +1.78). Registered outcome: the accumulation is still expected to be missing.
(g) Count, rest and spike-1 rise reported, not banded (this arm is registered for the step).

**Percent accuracy, defined before the numbers exist.** So that the summary is not chosen after
the fact: at 310 pA, ten elements are scored against the recording, each through the chain --
spike-1 rise, fall, peak, threshold, take-off, the 310 pA count, rest, the first-interval
threshold step, the spike-5 step, and the rise ratio spike 5 over spike 1. Per element the
accuracy is `max(0, 1 - |model - recorded| / |recorded|)` for quantities with a meaningful
zero, and for the two threshold steps and the rest (which do not) the denominator is the
recorded element's own scale stated in the scorer. The overall figure is the unweighted mean,
reported for B3 (rodent equations) and for the best human-sodium arm side by side. This is a
lossy transform of the kind chapter 2 warns about -- it is a summary for the reader, and the
per-element table beside it is the reading that carries the information.

Unchanged: the E recording, its chain, the readers, the sealed holdouts.

## Stage 16 (registered 2026-09-13, before the observation and before the runs): the threshold climb

**Why the climb and why now.** Stage 15 closed the rise: the sodium equation lineage is not its
remaining input, and the fully human-fitted base is no closer to this cell than B3 (72.3 against
71.8 percent over the ten registered elements). The accuracy that remains is concentrated in one
place: of the 28.2 points B3 is missing at 310 pA, the two threshold-climb elements carry 19.4
and the rise 6.4, and every other element is already between 90 and 100 percent. The recording
climbs +1.38 mV in the first interspike interval and +3.63 mV by spike 5 with the rise kept at
0.86; B3 climbs -0.02 and +0.24. HL23PYR reproduces the first-interval step exactly (+1.38) with
its rise kept, so the step is reachable by a human L2/3 model; B3 holds the excitability that
HL23PYR loses. The split is therefore: what does B3 lack that lets a threshold move between one
spike and the next.

**Part 1, an observation on retained traces (no run): what sets B3's threshold, and what changes.**
The stage-14 machinery is re-pointed from the peak of the upstroke to the take-off. On
`h01-e-currents/m0-b3-currents-sweep53` (the retained B3 run at 310 pA carrying every somatic
current on the solver's native grid), the somatic current balance is read at the take-off of
spike 1 and again at the take-off of spike 2, at the same membrane voltage, and decomposed by
carrier.
(a) *Why B3's threshold cannot move.* The net outward current at the take-off of spike 2, read at
spike 1's take-off voltage, against the same quantity at spike 1. Prediction if B3 lacks an
accumulating brake: the two agree within 10 percent, so nothing has accumulated between the
spikes and the threshold has no reason to move. Registered alternative outcome: they differ by
more than 10 percent and the threshold still does not move, in which case the threshold is set by
something other than the outward current at take-off and the dose below is refused.
(b) *Which carrier holds the outward charge at take-off.* Kv3_1, Im, K_P, K_T, SK, Ih and leak
ranked by their share at the take-off. Prediction from the fitted densities (soma Kv3_1 0.369,
Im 3.0e-4, K_P 2.3e-4, K_T 1.1e-8 S/cm2): the fast Kv3_1 dominates and the slow, accumulating
carriers are negligible. This names the carrier to dose; it is an observation, not pass/fail.

**Part 2, the dose (runs): a characteristic curve of the climb.** The slow outward carrier that
part 1 identifies is dosed on the soma of B3 across four densities, at 310 pA, sweep 53 only,
everything else held (the human anatomy, every other fitted density, the recorded command plus
bias, nseg 9, CVode 1e-10, 34 degrees). Manifest `h01-e-climb-manifest.json`, output
`h01-e-climb`, cap 5. No holdout is touched. All fast quantities through the stage-12 chain.
(c) *The climb.* Threshold at spike 2 minus spike 1, and at spike 5 minus spike 1, against the
recorded +1.38 and +3.63 mV. Prediction if the climb is a somatic slow-outward deficit: some dose
reaches at least half the recorded first-interval step (+0.69 mV) and the curve is monotone in the
dose.
(d) *It must not be bought with the count or the rise.* At that dose the 310 pA count stays within
5 to 15 (recorded 10) and the spike-1 rise stays within 15 percent of B3's own 570 V/s, so the
climb is not obtained by crippling the cell. The rise of spike 5 over spike 1 is reported against
the recorded 0.86.
(e) *Rejection.* No dose reaches half the recorded step (the climb is not a somatic slow-outward
deficit and the residual is reported, not repaired); or every dose that does also leaves the count
or the rise band; or the climb does not move monotonically with the dose (the dose is not acting
through that carrier).

**Percent accuracy.** Recomputed on the ten elements and the definition already registered at
stage 15, for B3 and for the best climb dose, reported side by side.

Unchanged: the E recording, its chain, the readers, the sealed holdouts.

### Stage 16 part 1 result and the corrected part 2 (2026-09-13, recorded before the runs)

**Part 1 fired the registered alternative outcome, and the somatic dose is refused.** On the
retained B3 currents run at 310 pA, read on the solver's native grid at a common subthreshold
voltage (-60 mV) on the ascending limb before each spike, the outward ionic current does
accumulate: 0.0108 nA before spike 1, 0.0133 before spike 2 (+22.5 percent, past the registered
10 percent), 0.0209 before spike 3 and 0.0229 before spike 4, carried mostly by Kv3_1 deactivation
tail and the intra-soma axial legs. And the take-off does not move at all: -57.20, -57.42, -57.16,
-57.20, -57.20 mV across spikes 1 to 5. The reason is in the same balance: at the take-off the
axon[0] leg delivers +0.33 nA while the whole accumulated somatic brake is about 0.012 nA, some
3 percent of the trigger. Cell (b) confirms the carriers at take-off are leak (69 percent) and
Kv3_1 (30 percent), with Im at 0.4 percent, so no somatic slow carrier is available to dose
either. B3's somatic threshold is the arrival time of the axonal spike, as stages 9, 10 and 14
found; no somatic conductance can move it. **The registered part-2 dose (a somatic slow outward
carrier) is refused and not run.** Four evaluations saved.

**Corrected part 2: accommodation at the initiation site, on a geometry where the soma can see
it.** The climb, if it is an availability effect, must act where the spike starts. Stage 10
established that the soma's take-off tracks the site only when the Allen 1 um stub is thickened:
the x2 density-held arm `s10-d2-density` slides -2.1 mV with the approach where the default stub
gives -0.07. So the dose is the axonal sodium's recovery from inactivation,
`--mechanism-parameter NaTs:axon:h_recovery_factor:X` (a RANGE parameter of the
`kv3-closing-source` NaTs), on that arm's geometry and flags, at 310 pA, sweep 53 only. Doses
0.5, 0.25 and 0.125 against the retained control at 1.0. One further run puts the strongest dose
on the **default 1 um stub** instead, as the coupling control. Manifest
`h01-e-climb-manifest.json`, output `h01-e-climb`, cap 5. No holdout is touched.
(c) *The climb.* Threshold at spike 2 minus spike 1 and at spike 5 minus spike 1, through the
chain, against the recorded +1.38 and +3.63 mV. Prediction if the climb is site accommodation
read through the coupling: the climb grows monotonically as the recovery factor falls, and some
dose reaches at least half the recorded first-interval step (+0.69 mV).
(d) *Not bought with the count or the rise.* At that dose the 310 pA count stays within 5 to 15
and the spike-1 rise stays within 15 percent of this arm's own control (s10-d2-density), so the
climb is not obtained by crippling the cell. Rise of spike 5 over spike 1 reported against 0.86.
(f) *The coupling control.* The same strongest dose on the default 1 um stub. Prediction from
stage 9: the somatic climb there is at most a fraction of the coupled arm's, because the soma
cannot see the site. If instead it is the same, the coupling is not required and stage 10's
account of the take-off is wrong.
(e) *Rejection.* No dose reaches half the recorded step (site accommodation is not the climb's
mechanism, reported not repaired); or every dose that does also leaves the count or rise band; or
the climb is not monotone in the dose.

**Accuracy caveat, stated now.** This arm's baseline rise is further from the recording than plain
B3's, so a climb won here need not raise the ten-element mean above B3's 71.8; the stage tests the
climb's mechanism, and the accuracy of the resulting model is reported honestly either way.

---

## 2026-09-13 — ledger correction; the campaign's population term was stale, and the binding constraint moved

Not a stage: no evaluation spent, no run launched. An audit of the campaign's own accuracy
arithmetic against committed evidence, occasioned by a recommendation that turned out to target a
blocker already repaired.

**What was wrong.** The stage-16 report stated the population product as
`71.8 percent x 55/104 x 12/104 = about 4 percent`, and recommended the next stage repair the
one-point SWC branch that stopped 7 of the 104 largest components from loading. That reader defect
was repaired on 2026-09-08, six days earlier: `h01-population-import-104.json` records 104 of 104
components importing with zero missing segments, `h01-ready-104-implicit-build-decision.json`
records 104 cells constructing to 808,495 compartments with no failures, initialisation passes,
and `h01-arc-probe-104.json` records a forward pass completing all six phases. The recommendation
is withdrawn; the build term is 1.000, not 12/104.

**Why the correction does not raise the product.** The same audit exposes three factors that were
never in it. The defensible form is a six-term product, each term read from a committed decision
JSON, with a term that has no evidence scoring zero and never being dropped — the rule the
stage-15/16 scorer violated once, and for the same reason: dropping a requirement quietly raises
the score of exactly the case that fails it.

| Term | Value | Measured |
| --- | ---: | --- |
| Donor accuracy (B3, ten elements, 310 pA; donor for 28 of 104) | 0.718 | yes |
| Type coverage (layer and class) | 0.529 | yes |
| Construction (import, construction, initialisation, synthetic forward pass) | 1.000 | yes |
| Driven window at a physiological duration | 0.000 | no |
| Timestep qualified against the 1 mV contract | 1.000 | yes (closed 2026-09-14) |
| Anatomy transfer (donor fit on H01 anatomy) | 0.000 | **yes, negative** |

As measured: 0 percent. With the three unqualified terms assumed away: 38 percent, quotable only
with the assumptions attached.

**The binding constraint moved.** It was believed to be an engineering blocker, 75 percent likely
fixable. It is anatomy transfer, which is the only term with a negative reading rather than a
missing one: B3's fit fires 4 spikes at 200 pA on the donor's own reconstruction and 0 on the H01
skeleton, which sits on a −78.2 mV plateau (onset capacitance 785 pF against 125; input resistance
about 38 MΩ against about 98). Donor accuracy, type coverage and a working build all describe a
model that is silent on the anatomy it is meant to run on.

**The next split, registered but not launched.** Scale the donor's dendritic load (`cm` and
`g_pas` together) by 1.5 and 2 on the donor's own anatomy and read the 200 pA count and the
upstroke. Prediction if the H01 silence is a graded load effect: the count falls monotonically and
the cell is still spiking at 1.5, so a load or conversion correction restores a readable response.
Rejection: the count falls to zero between 1.0 and 1.5, i.e. a cliff, in which case the conversion
itself is suspect and the membrane area must be checked against the H01 surface mesh before that
skeleton is used again. Cap 2 evaluations. This needs no population and no box-side population
cache.

Artifacts: [`h01-population-accuracy-ledger.md`](../evidence/h01-population-accuracy-ledger.md),
[`h01-population-accuracy-ledger.json`](../evidence/h01-population-accuracy-ledger.json),
[`h01_population_ledger.py`](../evidence/h01_population_ledger.py) (8 tests),
causal model Y5 consequence, accuracy tree (24 nodes, 28 edges).

---

## 2026-09-14 — the timestep term closes; the cleanup recovered the data the ledger said was gone

Not a stage: no evaluation spent, no simulation run. Arithmetic on traces recovered while removing
the stale worktrees.

**What the ledger got wrong.** The 2026-09-13 entry scored `timestep` 0.000 with the cause "three
finer rungs were run but their traces were deleted with the raw-trace sweep ... and `.cache/h01`
no longer exists". Both halves were false. The 2026-09-09 worktree-recovery stash under
`.cache/worktree-recovery-2026-09-09/` held two complete copies of the H01 source cache — the
proofread-104 archive, the nine export shards, the synapse table, and a 1.17 GB `readiness/`
directory containing all five dt-ladder traces. One copy was restored to `.cache/h01` before the
stash was removed.

**The ladder, closed as arithmetic.** Each rung is an exact bisection of the one above, so adjacent
rungs are compared at the coarse rung's own sample times by index arithmetic; nothing is
interpolated. That matters: the trace reaches +384 mV on a 0.165 µm² output compartment, and an
interpolated comparison across a spike edge would manufacture tens of millivolts the solver never
produced.

| pair | max ‖ΔV‖ | at | 1 mV gate |
| --- | ---: | ---: | --- |
| dt 0.005 → 0.0025 ms | 6.365 mV | 5.0150 ms | FAIL |
| dt 0.0025 → 0.00125 ms | 3.471 mV | 5.0150 ms | FAIL |
| dt 0.00125 → 0.000625 ms | 1.820 mV | 5.0150 ms | FAIL |
| **dt 0.000625 → 0.0003125 ms** | **0.933 mV** | 5.0144 ms | **PASS** |

Ratios 1.83 / 1.91 / 1.95 — first order in dt, independently matching what the I-cell transfer
study (SP2) found. **dt 0.000625 ms is qualified.** The committed decision
`h01-ready-cell7196644737-implicit-decision.json` reported the gate as FAIL because it compared
only the two coarsest rungs; it is superseded.

**What this changes.** The driven-window blocker is no longer an accuracy problem. The r2 run that
died at the wall cap was *already* running at the now-qualified 0.000625 ms, so it failed on cost
alone. The next attempt must be a cheaper window — fewer cells, a shorter duration, or a coarser
step justified by a per-cell ladder — not a fourth full-population run at the same settings.

Ledger terms now: donor 0.718, coverage 0.529, construction 1.000, driven window 0.000, timestep
**1.000**, anatomy transfer 0.000 (measured negative). Product as measured still 0 percent; the
binding constraint is unchanged and is still anatomy transfer.

**Housekeeping recorded here because it changes reproducibility.** Three evidence modules
(`h01_topographic_stage0.py`, `h01_e_morphology_swap.py`, `h01_sodium_slow_inactivation_test.py`)
resolved the human recording and the proofread archive through the recovery stash path. They now
point at `.cache/`, where the data was restored. Four PV audit tests still fail on raw `.npz`
traces deleted deliberately in 368e6d1 (319 unreferenced traces, 7.15 GiB); those were not in the
stash and are unrelated to this work.

Artifacts: [`h01-timestep-ladder.json`](../evidence/h01-timestep-ladder.json),
[`h01_timestep_ladder.py`](../evidence/h01_timestep_ladder.py) (7 tests).
