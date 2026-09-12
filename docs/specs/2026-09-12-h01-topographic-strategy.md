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

## Registered stage 0 (no simulation)

Produce, from retained traces: the phase-plane small multiples (rows: cycle 1, 2, middle,
last; columns: input) for E and I, human over model, with the repeat envelope; the
I_net(V) load curves of Q3; and the Matryoshka table naming the family that carries the
largest contrast in units of the repeat sigma. Decision rule: the family and the branch
(input or function) with the largest contrast are named; no lever is named. Cost: one
analysis script and its test; hours.
