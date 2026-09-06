# H01 cells: energetic re-characterisation, Isolation first, then Z-dissection

Status: approved by the user on 2026-09-06 (plan-mode approval); execution in progress on feat/h01-braincell. Book references are scan files pNNN.md in the Hartshorne 2020 transcription.

## Context

The 2026-09-06 campaigns (spec `docs/specs/2026-09-06-h01-progressive-search-plan.md`)
closed FAIL for both cells. Reviewed against Hartshorne 2020 (`pages-md/pNNN.md`),
the search broke the book's strategy in six places: it searched on the NOK side
(counts and pass/fail rows, p047, p054); it skipped the Isolation split (inputs vs
function, p144, p179); it ran Dissection between two systems that both fail, so no
swap could reverse (p183, p204); it compressed each group to one RSS number (p199,
p204); it spent reserve on flat Xs (p033); and its explanations are structural labels,
not conditions plus mechanism (p013, p048, p204 margin).

This plan redoes the search in the book's order. User decisions (2026-09-06):
bias forensics first, before any simulation; E cell currents allowed at 35 min per
run, cap 8; I cell keeps 15 min per run, cap 16. No training adapter, no Example 21,
no population work. Worktree `.worktrees/h01-braincell`, branch `feat/h01-braincell`.

Facts from the capability survey that shape the plan:

- **No saved campaign trace contains a current.** Currents require re-runs:
  PV driver `--observe-spike-currents --observe-charge-balance` (capacitive, leak,
  Ih, per-channel, axial via neighbour voltages + `charge_balance_geometry`);
  L2 driver `--record-soma-currents` (per-channel + `soma_icap`, **no axial probe**).
- Phase-plane (V, dV/dt) needs only voltage, so every existing trace and both human
  recordings serve the Isolation and Matryoshka stages at zero run cost.
- Human repeats exist but not at calibration amplitudes: PV 120 pA ×4 (sweeps
  40–43); L2 200 pA ×7 (`C1LSFINEST`, must be extracted from `recording.nwb`).
  PV sweeps 35 and 39 are singletons.
- Reversible channel block: PV `--scale-conductance {NaTg,Kv3_1,SK} --conductance-factor 0
  --conductance-region {soma,axon,all}` plus `somatic_kv3_factor`/`somatic_calva_factor`
  accept 0. L2 `--regional-density MECH:REGION:FACTOR` rejects factor ≤ 0 in
  `h01_l2_regional_density.py` lines 18–19 and 47–48 (relax to `< 0`).
- Segmenters to reuse: `spike_datums` (`h01_pv_human_datums.py`), `event_rows`,
  `recovery_rows`, `dixon_limits` (`h01_contract_score.py`). Layout scaffold:
  `h01_causal_trace_plot.py`. Runner: `h01_campaign.py` manifests. Three sign
  conventions (NEURON outward-positive mA/cm², BrainCell inward-positive) must be
  normalised once at load with the segment-area rule (`h01-pv-charge-balance.md`).
- Cost basis: I evaluation 90–350 s at nseg×9 (4 inputs); E evaluation 1120–1210 s
  (2 sweeps) without probes, up to ~3× with probes on sweep 50.
- Z frame already fixed in `docs/evidence/h01-circuit-z-map.md` §"Dissipation and
  storage, stated once": stored `CV²/2`, dissipation `g(V−E)²`, sources = applied
  current + bias only.

## The characteristic (Y) for every stage

Per spike cycle, minimum to minimum (`recovery_rows` boundaries), the conjugate pair
is membrane voltage (effort) and membrane current (flow); charge is the displacement.

- From voltage only (human and model): the phase-plane loop (V, dV/dt); its
  landmarks: threshold (dV/dt = 10 V/s crossing), max rise rate, max fall rate,
  peak, minimum, cycle length. These are the elemental family.
- From model currents: charge per boundary per cycle in pC: applied + bias,
  capacitive (closes to CΔV), each channel, leak, Ih, axial (PV: from neighbour
  voltages and `resistance_mohm`; L2: after the axial probe is added). Net charge
  per cycle closes to the balance residual, which is the numerical decision floor.
- Never a count. Counts appear only in the final prediction against the contract.

## Stage 0 — Bias forensics (read-only, minutes, before any simulation)

Isolation question (p144, p145 row 2): is the human–model contrast carried in by the
measurement, in the form of the 31.4 pA bias, or does it live in the function?

`docs/evidence/h01_pv_bias_forensics.py` (+ `_test.py`): from the PV NWB via the
loader in `h01_pv_acquisition_audit.py`, for every long-square sweep read the
recorded bias (pre-pulse mean of the current channel), the pre-pulse baseline voltage,
and the steady deflection of the passive sweeps for input resistance. Test: regress
baseline voltage on bias across sweeps (bias spans about 25–32 pA; passive sweeps
give R_in). Prediction if the bias is real membrane current: baseline moves by
R_in × Δbias (order 1–2 mV across the range, above the 0.02 ms/50 kHz noise floor).
Prediction if it is an amplifier offset: baseline is flat within its own repeat range.
Decision limit from the 120 pA ×4 baseline repeats (DLF 1.47). Output
`h01-pv-bias-forensics.{json,md}`; update `h01-pv-input-datum.json` with the verdict.

Consequence for later stages: if offset, the zero-bias source (11 events at 0.19 nA
vs human 12) is the high performer for Dissection and every bias-on residual retires;
if real current, the 2026-09-06 datum stands and the template mismatch moves to the
function branch. Either way the answer is recorded before Stage 1 opens.

Same question for E, cheaper: sweep 43 bias is −3.7 pA and sweep 50's is in
`sweep-50.npz`; record both; no forensics unless either exceeds 10 pA.

## Stage 1 — Isolation with repeatability (zero simulation for the human side)

`docs/evidence/h01_isolation_repeatability.py` (+ `_test.py`):

1. **Human repeatability.** Extract PV sweeps 40–43 (120 pA) and L2's seven 200 pA
   sweeps (extend `h01_l2_recording.py` sweep loader to accept any long-square
   sweep by index; keep 53 closed). Compute the Y landmarks per cycle; per landmark,
   the range across repeats → decision limit R̄ × DLF (1.47 for four, 1.81 for
   three; seven repeats use 1.47 as the conservative floor). This is the measurement
   branch's own spread.
2. **Model repeatability.** Two numerical repeats per cell from existing files where
   present (I: ×9 vs ×27 traces in the contract records; E: atol 1e-10 vs 1e-11
   corner runs), else one pair of runs charged to the caps (I 2, E 2).
3. **Contrast table.** For each landmark and input, human–model contrast against the
   combined limit √(DL_h² + DL_m²). Rows under 5× are parked with the reason
   "inside repeatability" (p033, Fig. 25). Rows above 5× are the search rows.
   Expected: the E late subthreshold return is the first candidate to park if the
   seven human repeats spread by more than 1.2 mV there.

Output `h01-isolation-{i,e}.{json,md}`: a two-branch tree (inputs / function /
interdependency) with the parked rows struck.

## Stage 2 — Matryoshka multivari on the phase plane (zero new runs)

`docs/evidence/h01_spike_cycle_energetics.py` (+ `_test.py`): cycle segmentation,
phase-plane landmarks, charge-per-boundary when currents are present, sign and unit
normalisation at load.

`docs/evidence/h01_multivari_plot.py` (+ `_test.py`): small multiples per cell.
Rows = input level (structural: 0.19 / 0.27 nA; sweep 43 / 50). Columns = early
(cycles 1–3) and late (last 3) train (temporal). Each cell overlays three consecutive
cycles (cyclical) of human and model in the phase plane (elemental). Square, shared
axes. Built from existing campaign traces (source, frozen candidate, matrix cells) and
the human recordings.

Decision (Sparsity of Effects, p033, p034): the family whose stratum carries the
largest share of the human–model range is named. Interdependency (input × early/late,
p036) is trapped as its own item. Pre-registered predictions: I contrast is
cyclical×temporal (cycle length drifts across the train differently at the two
inputs); E contrast is temporal (late cycles only). Rejection: if the elemental loop
itself differs beyond the limit in the first cycle at both inputs, the family is
elemental and Stage 3 starts at the first spike, not the train.

Output `h01-multivari-{i,e}.png`, `h01-matryoshka2-{i,e}.md` replacing the
residual-based Matryoshka tables.

## Stage 3 — Z-dissection of the function (current runs; I cap 16, E cap 8)

Runs only for the cell whose contrast lives in the function branch after Stage 1.
Manifests `h01-i-energetic-manifest.json`, `h01-e-energetic-manifest.json` for
`h01_campaign.py`, with `abort_seconds` 900 (I) and 2100 (E, probed runs only).

1. **Reference charge budgets (I 2, E 2).** Source and frozen candidate with all
   current probes at every calibration input. Per cycle, per boundary charge in pC;
   per-boundary share of the cycle's outward charge. The Z map gains one row per
   boundary with measured charge per cycle.
2. **Reversible removal per boundary (I ≤ 6, E ≤ 4).** Zero one conductance at a time
   (PV: NaTg soma, NaTg axon, Kv3 soma, SK axon, Ca_LVA soma; L2 after relaxing the
   density guard: Kv3, SK, Ih, Ca) and restore. Y = the charge budget of the surviving
   cycles plus the phase-plane loop. Decision: the boundary whose removal changes the
   cycle's charge deficit by the largest share, above the Stage 1 limit, is the Steep
   boundary. This is a function split by reversible removal, not a swap between two
   failing models.
3. **Search Dissection only with a high performer (I ≤ 4).** If Stage 0 makes the
   zero-bias source the high system, swap the Steep boundary's members between it and
   the candidate with restore; DLF limits from the Stage 1 model repeats; combined
   half-split then 2³ per p202. Otherwise skip: no reversal is possible and the book
   says not to run it.
4. **Numerical floor (I 2, E 2 from the caps above if not already spent).** Halve dt
   or tighten atol on the finalist; any landmark moving beyond its limit returns to
   noise.

Driver edits, only when Stage 3 reaches E: add an axial probe to
`h01_l2_neuron_reference.py` mirroring the PV `--observe-charge-balance` block
(neighbour voltages + `resistance_mohm` in the report); relax the two factor guards
in `h01_l2_regional_density.py` from `<= 0` to `< 0` with a test that factor 0 zeroes
`gbar`.

## Stage 4 — Causal explanation and one prediction

Rewrite Y3 and Y4 in `docs/h01-causal-model.md` in the book's form (p013, p048):
necessary and sufficient conditions plus the how-why mechanism in charge terms, for
example "the cycle length is the time for the applied plus bias charge to replace the
charge removed by boundary X after each spike; condition: X's per-cycle charge exceeds
the applied charge over the observed interval". No sub-system names as causes; the
structural finding is stated as where the mechanism lives, not as the explanation.

Prediction test, once, on the closed holdouts (I 0.23 nA; E sweep 53 extracted by
the extended loader): predict the phase-plane landmarks and cycle lengths from the
explanation before the run, then run and score against the contract rows. Pass or
fail recorded in `h01-prediction-{i,e}.md`. No refit.

Update `h01-implementation-status.md`, the Z map, the tutorial's "what its
behaviour represents" section, and the spec outcome section.

## Fail-fast rules (p034, p179)

- Each stage is answerable in one working session; a stalled job is killed at the
  first missed progress line, not after six hours.
- Caps are the stop: I 16, E 8. At cap without an explanation that predicts the
  holdout, report FAIL with the reassessment question and return the decision.
- No stage opens until the previous stage's decision file has a non-empty decision
  (`stage_ready` in `h01_campaign.py`).

## Files

New under `docs/evidence/` (each with sibling `_test.py`):
`h01_pv_bias_forensics.py`, `h01_isolation_repeatability.py`,
`h01_spike_cycle_energetics.py`, `h01_multivari_plot.py`;
manifests `h01-i-energetic-manifest.json`, `h01-e-energetic-manifest.json`;
outputs `h01-pv-bias-forensics.*`, `h01-isolation-{i,e}.*`,
`h01-multivari-{i,e}.png`, `h01-matryoshka2-{i,e}.md`, campaign dirs
`h01-{i,e}-energetic/`, `h01-prediction-{i,e}.md`.

Spec: `docs/specs/2026-09-06-h01-energetic-isolation-plan.md` (this plan, with the
pre-registered predictions per stage).

Modified: `h01_l2_recording.py` (any long-square sweep by index), `h01_l2_neuron_reference.py`
(axial probe), `h01_l2_regional_density.py` (factor 0 allowed), `h01-pv-input-datum.json`,
`docs/h01-causal-model.md`, `docs/evidence/h01-circuit-z-map.md`,
`h01-implementation-status.md`, `docs/tutorials/h01_braincell.rst`.

Not touched: `braintrace/datasets/h01_ei_profiles.py`, `_h01_ei_parameters.py`,
Example 21, anything on `main`.

## Verification

- `pytest docs/evidence -k "h01_pv_bias_forensics or h01_isolation or h01_spike_cycle or h01_multivari" -q`;
  `pytest braintrace/datasets -n auto`; `pytest examples/ -n auto`; `pytest repo_conventions_test.py -q`.
- Unit tests cover: cycle segmentation on a synthetic two-spike trace; sign and unit
  normalisation for all three conventions; charge closure equals the balance residual
  on `h01-pv-neuron-charge-balance.npz`; DLF limits for 3, 4 and 7 repeats; parking
  rule; multivari grid shape; L2 loader rejects sweep 53 until Stage 4 opens it.
- Stage gates: each stage's decision file exists and names its verdict before the
  next manifest opens; run wall times recorded per run; probed E runs under 2100 s.
- Final check: the Stage 4 prediction was written before the holdout run, and the
  causal page states conditions and mechanism with no sub-system name as a cause.
