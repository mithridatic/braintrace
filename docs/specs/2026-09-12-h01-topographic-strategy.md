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
