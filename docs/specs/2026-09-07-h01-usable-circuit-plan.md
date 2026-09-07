# H01 BrainCell: from validated waveforms to a usable circuit

Status: approved by the user on 2026-09-06 (plan mode), executed on
`feat/h01-braincell` in `.worktrees/h01-braincell`. Companion specs:
[energetic isolation plan](2026-09-06-h01-energetic-isolation-plan.md),
[acceptance contract](2026-09-05-h01-recorded-response-acceptance-proposal.md),
[verified network](2026-09-06-h01-verified-network.md) (other session).

## Why

The energetic search explained the I cell's loop and trough and located the E
cell's fault (perisomatic fit, no initiation site). Both cells still fail the
approved 1 mV contract on count and one peak, and both holdouts are spent. An end
user of H01 BrainCell does not feel a 0.15 mV peak; they feel firing rate,
adaptation, spike width, the afterhyperpolarisation, and whether the I cell
inhibits the E cell measurably. None of those is scored, the one verified I-to-E
contact moves the E soma by 0.003 mV with assumed synaptic numbers, and the
104-cell network has three verified edges and two frozen profiles for 104 cells.

## Doctrine that fixes the method

From comsol-llm-lab `docs/ENERGETIC_FRAMEWORK.md`, `conjugates.yaml` and
`DIAGNOSIS_DOCTRINE.md` §3 and §4, and Hartshorne 2020:

1. **Retained coordinates, never a summary statistic.** Every scored reading keeps
   its input level and cycle index (elemental and cyclical families). Rate and
   adaptation ratio are derived views of the per-cycle table, computed only for the
   acceptance verdict, never stored in place of it.
2. **Both members of a conjugate pair.** Electrical domain: effort = membrane
   voltage (a delta to the extracellular ground), flow = membrane current,
   displacement = charge, momentum = flux linkage (not applicable). The current
   clamp is a flow source (Norton) feeding the membrane load; the load determines
   the effort. The human-comparable within-cycle pair is voltage against the
   capacitive flow C·dV/dt (the phase plane); charge per boundary per cycle is the
   displacement view, available for the model only.
3. **Tolerance is set for function; repeatability must be under half of it**
   (p130). A usable-tier limit whose human repeat or cyclical spread exceeds half
   the limit is `unresolvable`, never `pass`.
4. **Quality control reads one value against a tolerance; diagnosis reads the
   shape** (p063–p064, p095). The usable tier is the QC view; the energetic map and
   the retained table are the diagnosis surface.

## Workstreams and pre-registered predictions

### W0. Energetic map per cell (before any limit is set)

`docs/evidence/h01-z-map-i.md` and `h01-z-map-e.md`: source-to-load maps in the
framework's block vocabulary (Supply, Transmit, Contain/Release, Modulate,
Dissipate), one table per cell "usable row → controlling property → lever flag",
built from the Stage R budgets. Predictions per row are written on those pages.

### W1. Usable tier

`docs/evidence/h01_usable_tier.py`: `cycle_table` (stored form, one row per input
and cycle), `qc_view` (rate, adaptation ratio, per-cycle width and AHP),
`usable_limits`, `resolvability`, `verdict`, `render`. Limits (pre-registered):

| Row | Limit | Basis |
| --- | --- | --- |
| rate_hz per released input | 15 % of the human rate | functional tolerance (user default) |
| adaptation_ratio (last cycle / first full cycle) | 25 % of the human ratio | functional tolerance (user default) |
| width_ms (time above −20 mV) per cycle | 20 % of the human width | functional tolerance: the width sets the charge (calcium) entering per spike |
| ahp_mv (cycle minimum) per cycle | 2 mV | functional tolerance: about a tenth of the threshold-to-trough excursion, which sets the recovery to threshold |

Resolvability: the human spread (repeat range × DLF, or cyclical range × 1.47)
must be below half the limit; otherwise the row is `unresolvable`.

Amendment recorded before the first scored page was committed: the plan first
set the width and AHP limits equal to the human repeat spread. That rule is
degenerate under rule 3 (a limit equal to its own spread is always
unresolvable), so the two rows received functional tolerances and the spread
became the resolvability check, as for the other rows.

Prediction before scoring the existing traces (I finalist `e-kv3-close2` at 0.19
and 0.27, `p-close2-023` at 0.23; E candidate `r-candidate-sweep50`,
`p-candidate-sweep53`): I passes width and AHP at every input, fails rate at 0.23
(model 26 vs 31, 16 % low) and fails adaptation at every input (the model's late
cycles stay near 40 ms while the human's first cycles are 6 ms); E fails rate at
both sweeps (5 vs 5 at 250 pA is a pass by count but the late cycles are twice the
human's, so the mean-cycle rate fails; 6 vs 10 at 310 pA), passes AHP at both.

### W2. Sealed holdouts

PV Noise 1 sweep 48 (278.12 pA, 28 s) with siblings 44/50 and Noise 2 45/47/49/51
closed as the future repeat family; L2 sweep 55 (350 pA). Both seals state that
the Allen spike counts were on disk and were seen, so count is not closed;
waveform, timing, peaks and per-cycle rows are. `--sweep 55` stays rejected by the
L2 driver until opened. Replaying Noise 1 needs a waveform path in the PV driver
that does not exist; built only when the I cell is next opened.

### W3. E cell, new campaign, cap 8

Manifest `h01-e-usable-manifest.json`, sweep 50 without current probes, abort
1500 s. Driver gains `--insert-density MECH:REGION:VALUE` (absolute S/cm²; appends
a genome row and a reversal row for the region) and an axon voltage probe.

Stage A (2 evaluations), initiation site. `a-axon-na`: NaTs in the axon at 3.81
S/cm² (the somatic reference 2.934 × the candidate's 1.3). `a-axon-na-soma065`:
the same with somatic sodium at 0.65. Prediction: the axon crosses −20 mV before
the soma in both arms; a somatic shoulder of 100 to 300 V/s between −55 and −45
mV; threshold toward −56 mV; main rise below 550 V/s in the second arm; peak 28 to
36 mV; count 5 ± 1 at sweep 50. Rejection: no shoulder and no axon-first crossing
in either arm: the 60 µm stub cannot host initiation, and the initial-segment
geometry is returned as a family change beyond this cap.

Stage B (up to 6), count and gain, gated on A. One boundary per arm from the W0
map: `b-sk-half` (SK soma × 0.5), `b-ca-decay` (calcium decay factor 1.0 against
1.33), `b-leak-rev` (leak reversal −2 mV). Each scored by the usable rate row at
sweep 50 and once at sweep 53. Stop: first arm inside the usable rate limit at both
sweeps without breaking AHP or width, or cap.

Stage P: prediction on sealed sweep 55, written before export; opening spends it.

### W4. Synapse

Human neocortex PV-to-pyramidal paired-recording values (unitary IPSP amplitude,
rise, decay) fetched and pinned with DOI in `h01-ie-synapse-literature.json`; no
number from memory enters code. The E soma sees the synapse as a Norton source
feeding the E load; `inhibitory_weight_us` and `tau` are set so one I spike from
rest gives the pinned unitary IPSP, predicted first from the E input resistance.
Pair test `h01_ie_pair_response.py`: connected against disconnected, retained per
event: amplitude, latency, decay, and E cycle-length change when E fires.

### W5. I cell

The reserve is held until W3 Stage A answers whether an axonal initiation site
reproduces a two-stage upstroke on this stub geometry.

### W6. Population, what is in scope and not delivered

Delivered by the other session (129841f, 9184269, dff10c7): 104 signed nodes,
3 verified contacts, a fixture-tested builder with two frozen profiles, a
construction-cost fix. Left in scope and done here without touching their files:

- W6a typing beyond E/I (`braintrace/datasets/h01_cell_types.py`,
  `h01-population-types.{json,md}`). Prediction: fewer than 40 of the 104 cells have
  a type-matched donor (E donor is an L2 pyramid, I donor a PV basket model).
- W6b component inventory (`h01_population_components.py`,
  `h01-population-components.json`): which cells a population build would model as
  fragments.
- W6c complete the C3 export scan (`full_export_scanned` true), fail-fast at 10
  minutes without a progress line.
- W6d 5-voxel re-verification retried one edge at a time, 5-minute abort per
  edge, on the 9 candidates that matter first.
- W6e `h01-population-status.md`: what a user gets today.

## Fail-fast rules

A run is killed at the first missed progress line; a killed run is reported as
untested, never negative. No stage opens without the previous stage's decision
file. Predictions are written before the run that tests them; a wrong band is
recorded against the prediction, not the model.

## Outcome

Appended per workstream as it closes.

- **W0.** `h01-z-map-i.md`, `h01-z-map-e.md` written with the per-row levers and
  predictions before scoring.
- **W1 (scored on the existing traces).** I: rate fails at 0.19 (14 vs 12 spikes,
  the model overshoots) and 0.27 (37 vs 43), passes at 0.23 (26 vs 31 is inside 15
  percent); adaptation fails at every input (model ratio 1.9 to 2.4 against the
  human 4 to 13); width fails at most cycles (0.22 against 0.27 to 0.30 ms, 20
  percent narrow, resolvable: spread 0.006 ms); AHP passes every cycle at 2 mV.
  E: rate passes at 250 pA, fails at 310 pA (5.3 against 10 Hz); adaptation fails
  at both (model 11 to 14 against 8 to 10); width passes; AHP is unresolvable at 2
  mV because the human 200 pA repeats spread 1.01 mV. Against the prediction:
  the I rate verdicts were wrong at 0.19 and 0.23 (predicted fail at 0.23 only),
  the I width was predicted to pass and failed, the E 250 pA rate was predicted to
  fail and passed. Pages `h01-usable-tier-{i,e}.md`.
- **W6a.** 30 of 104 cells match their donor in layer and class (prediction: fewer than
  40, held): 28 L2 pyramids and 2 plain L5 interneurons. 47 differ in layer only, 13
  in layer and a morphology modifier, 9 in class. Every interneuron's subtype is
  unknown; the PV donor is assumed (`h01-population-types.md`).
- **W6b.** 3,327 components; every cell has a soma-bearing component and in every
  cell the largest component carries a soma, but 73 cells have more than one
  soma-bearing component (a segmentation artefact or a merge) and the largest holds a
  median 77 percent of the nodes (minimum 25 percent). Five of six verified contact
  endpoints sit on a soma-bearing component (`h01-population-components.md`).
