# H01 human-constrained circuit: progressive topological search plan

Status: approved by the user on 2026-09-06 (plan-mode approval); execution in progress on feat/h01-braincell.

## Context

The H01 worktree (`.worktrees/h01-braincell`, branch `feat/h01-braincell`) has a
two-cell BrainCell circuit on measured anatomy, but neither cell passes the human
acceptance contract the user approved on 2026-09-05 (exact event count, 1 ms
times, 1 mV voltages, 0.05 ms spike phases). The inhibitory (I) cell fires on H01
only under a diagnostic sodium closing-time override (0.15 → 1.0). The
excitatory (E) cell has the right count but its fifth onset is ~64 ms late. Serial
one-parameter tuning was stopped by the validation reassessment spec; the last
bounded campaign (L2 family search) produced a family ranking, not a pass.

Goal: a **reusable, human-constrained measured I→E (and E→I if a measured
contact exists) H01 pair in BrainCell** that other projects can run, with evidence
that states exactly what its behaviour represents. Methods come from Hartshorne
2020 as transcribed in
`C:\Users\J\Documents\Diagnosing Performance and Reliability\pages-md\` (cite scan
files `pNNN.md`; printed page numbers are scrambled).

User decisions (2026-09-05): measured I→E plus E→I if available; both cell
campaigns in parallel; 24 evaluations per cell, ≤15 min per run, two concurrent;
**no training adapter and no Example 21 work** (deferred to a later 104-cell
population effort). The population is not blocked in principle; it is gated by
the partner-mapping audit that Phase 0 advances as a read-only task.

Two findings from the design review change the I campaign and are built in:

- **I bias.** Every existing I residual was computed with zero bias, but the
  recording has a measured 31.4 pA acquisition bias; on the source model it
  doubles the 0.19 nA count (11 → 22) and moves the first peak 15 ms
  (`docs/evidence/h01-pv-bias-audit.md`). That is larger than any candidate
  family. The contract requires the recorded bias. The I input datum must be
  fixed before any I family ranking means anything.
- **I mesh.** At nseg ×9 the 0.27 nA late train is unresolved: ×9 → ×27 changes
  the count 43 → 41 and a late interval by 1.84 ms
  (`docs/evidence/h01-pv-calcium-response.md` §Spatial limit). The frozen I
  profile is qualified only at axon ×2187. Count and late-interval rankings at
  ×9 are inside noise.

## Method rules that govern every phase

1. **Topographic before symptomatic** (`p014.md`): map elements and connections
   (Phase 0) before asking "what's wrong".
2. **Matryoshka first** (`p099.md`, `p034.md`): classify each residual as
   elemental → cyclical → structural → temporal before choosing a split. Here:
   elemental = input datum / which cell / which mechanism family; cyclical =
   repeats across inputs (0.19 vs 0.27 nA; sweep 43 vs 50) and across events;
   structural = region map / mesh / contact path; temporal = early vs late in the
   train.
3. **Only real contrasts** (`p033.md`): search only rows whose residual exceeds
   the approved allowance by ≥10× (unmistakable), and whose source-to-candidate
   contrast exceeds 5× the decision limit. Rows within 2× are parked.
4. **Small-n decision limits** (`p199.md`–`p201.md`): numerical repeats are the
   "repeats"; decision limit per row = R̄ × DLF, DLF 2.95 for two repeats, 1.81
   for three. This replaces the ad-hoc "one fifth of the smallest contrast" rule.
5. **Paired comparison plots** (Youden, `p148.md`–`p150.md`): every candidate is
   plotted model-vs-human on the two calibration inputs with identical square
   axes, separating input-to-input reproducibility from candidate error.
6. **Search Dissection** (`p183.md`, `p202.md`, `p204.md`): swap one sub-system
   between the high and low performer, restore after each; if no single swap
   reverses, combined half-split of the two largest; ≤7 sub-systems per level.
7. **Pre-registration** (`p145.md`, `p194.md`): every stage carries prediction,
   rejection criterion, and unchanged quantities in the manifest before it runs.
8. **Fail fast** (`p034.md`, `p179.md`): each split answerable in hours; the cap
   is the stop; at cap without a pass, report failure and reassess model family
   or data. No extension of the series.
9. **Adequate causal explanation** (`p048.md`): necessary and sufficient
   conditions plus how-why mechanism; not a root-cause label, not a list, not
   structural, not over-complex. Update `docs/h01-causal-model.md` with supported
   conclusions only; unresolved links stay explicit.
10. **Energetic boundaries** (`p206.md`–`p214.md`,
    `comsol-llm-lab/docs/ENERGETIC_FRAMEWORK.md`): observe voltage and signed
    current together at each boundary; conservation closes the charge balance at
    the observation CV; a residual above numerical resolution is bookkeeping, not
    physics.

## Phase 0 — Contract scorecard, input datum, topographic map (no new simulations except two I bias runs)

All under `docs/evidence/`, scripts with sibling `_test.py`.

1. **Commit the contract.** `docs/specs/2026-09-05-h01-recorded-response-acceptance-proposal.md` is uncommitted (`M`). Commit it first; it is the gate.
2. **`h01_contract_score.py` → `h01-contract-scorecard.{md,json}`.** Apply the
   contract row by row to every existing record: I source, I candidate
   (`h01-pv-regional-mesh-axon2187`), axon729 residuals, frozen full response;
   E source, candidate (`h01-l2-kv3-ninety-ca133`), corner, all 14 family-campaign
   runs. Columns: residual, allowance, ratio, numerical limit, verdict
   (pass / fail / unavailable). Reuse `h01_l2_campaign_score.py` residual
   vectors and `h01_pv_human_datums.spike_datums`. Expected from existing data:
   - ≥10× (targets): E i2 +51.6 ms, i4 +20.6 ms, event 3–5 crossings
     +44/+44/+64.5 ms; I 0.19 nA third interval −25.8 ms, late interval −11 ms,
     15 vs 12 events; I 0.27 nA 40 vs 43 events. I minima +7–8 mV and 0.19 onset
     +7.65 ms are 7–8× (real, second tier).
   - ≤2× (parked): E e1 rise/peak, m1/m3/m4, sub 1120/2120; I 0.27 e1 rise,
     duration, peak, peak-to-minimum delay.
3. **I input datum.** Resolve the acquisition bias for each I calibration input
   from the raw records via `h01_pv_acquisition_audit.py` (0.19 nA has 31.4 pA
   on sweep 35; 0.27 nA to be read). Record in
   `h01-pv-input-datum.json`. Then run C at nseg ×9, CVode 1e-10, **with**
   recorded bias at 0.19 and 0.27 nA (2 evaluations, charged to the I cap).
   Decision: if any target row moves by more than 5× its decision limit, the
   zero-bias residual set is retired and all I ranking uses the bias-on input.
   This is the Matryoshka elemental question: is the input correctly specified?
4. **`h01-matryoshka-{i,e}.md`.** Each target row with its family class and
   evidence link. Expected: I early intervals short and count wrong = cyclical +
   temporal; E i2/i4/fifth onset = temporal (adaptation), localised to F1/F2/F4
   on the F3+F5 base.
5. **`h01_youden_plot.py`.** Model-vs-human per row for (0.19, 0.27 nA) and
   (sweep 43, 50).
6. **Edge list among resolved cells.** Extend `h01_index_connectivity_audit.py`
   to complete the export scan (`full_export_scanned` is `false` in
   `h01-measured-connectivity-summary.json`) and emit
   `h01-resolved-edge-list.json`: every directed contact whose endpoints resolve
   to two different proofread cells, with type code, coordinates, placement
   distance. Answers: does a measured E→I contact `4157825456 → 5584343344`
   exist; how many of the 104 have ≥1 resolved partner; which 25 records still
   lack identity. Read-only, one pass, wall time recorded.

Gate to Phase 1: scorecard, input datum decision, Matryoshka tables, edge list
all present.

## Phase 1 — Two bounded campaigns in parallel (24 evaluations each)

Both run in `braintrace-h01-neuron:9.0.2` on donor anatomy. One evaluation =
one parameter set at every calibration input. Manifest-driven runner: generalise
`h01_l2_campaign.py` into `h01_campaign.py` (driver script, input map, required
end time, and per-run abort seconds come from the manifest; keep the cap check,
flag hash, skip-finished, ≤2 concurrent, wall-time record, `STAGE_GATES`
refusal). Keep `h01_l2_campaign.py` as a thin alias. Add `--candidate-json` to
`h01_pv_neuron_reference.py` mirroring the L2 driver. Benchmark one full
evaluation per cell and report total expected wall time before the manifest is
opened.

### I campaign (PV, HL5BN1 donor geometry) — 20 planned, 4 reserve

Inputs: 0.19 and 0.27 nA with the Phase 0 bias datum, 270–1270 ms pulse, run to
1500 ms, CVode 1e-10. 0.23 nA stays closed. Cost basis at ×9: 10 s per 330 ms
(`h01-pv-transfer-isolation.md`); ×2187 cost is unmeasured and must be
benchmarked in Stage 0.

Sub-systems (candidate minus source; `_h01_ei_parameters.py`,
`h01_ei_profiles.py:channel_controls`):

| Sub-system | Members | Predicted group (from Y3 evidence) |
| --- | --- | --- |
| G1a | NaTg h_close 0.15 | first-spike duration and peak |
| G1b | NaTg h_slope 5 | count and late intervals at 0.19 |
| G2 | soma Kv3 m_open 0.5, m_close 0.5 | return and minimum |
| G3 | axon Ca decay 300 ms, gamma 0.004 | 0.19 intervals, onset unchanged |
| G4 | soma NaTg ×1.1, Ca_LVA ×0.5 | onset earlier, first minimum deeper |

G1 is split at Stage A because h_close is the Y1 lever for the H01 spike and
h_slope is the late-count lever; a pooled swap would confound two groups.

| Stage | Evals | Content | Decision |
| --- | --- | --- | --- |
| 0 | 2 (+2 bias from Phase 0) | S at ×9; C at ×9; C at axon ×2187 (the ×2187 run reuses `h01-pv-focused-mesh-2187` if its flags match) | Per-row R̄ = \|C₉ − C₂₁₈₇\|; decision limit = 2.95 R̄. Rows whose S–C contrast < 5× limit are excluded from ranking. If any count or late-interval row is excluded, Stage A runs at ×2187 (cost benchmarked here). |
| A | 10 | Each sub-system into S and out of C | Per observation group, Steep X when one sub-system's normalised RSS exceeds the rest in quadrature. If no group reverses under any single swap → B. |
| B | 2 | Two largest jointly into S / out of C | Not reversed → stop: the partition is wrong; reassess (see fail rule). |
| C | 2 | Within-family split of the minimum-group winner (predicted G2: close-only vs open-only restored on C) | Same direction from both → phase is not the lever; G4 Ca_LVA dose next from reserve. |
| D | 2 | Finalist at ×2187 and CVode 1e-11 | Any ranked row moving > its limit returns to noise. |

Fail rule: no candidate meets the count row at both inputs and every target row
by the cap → FAIL report. Reassessment question recorded: the published template
gives 12/29/46 vs recorded 12/31/43 before any candidate change, so the model
family may not contain the recording; the decision whether to change family or
reference is the user's.

### E campaign (L2, Allen donor geometry) — 12 planned, 12 reserve

Inputs: sweep 50 with recorded bias and sweep 43, nseg ×9, CVode 1e-10, through
2140 ms. Sweep 53 stays closed. Cost basis ≈17 min per evaluation; the
F5-containing base ran 856 s for sweep 50, so pre-register a 900 s per-run abort
and report it as a failed evaluation if hit. New manifest with
`prior_evaluations` reset (runner enforces len + prior ≤ cap).

Base = S + F3 + F5 (5 events, intervals within 10.5 ms, minima 1.6 / 4.4 mV too
negative). Split: 2³ over F1 (NaTs m_open 2.0), F2 (Kv3 m_close 0.9), F4 (soma
NaTs ×1.3) on the base. Cells 000 (`ab-into-F3F5`) and 111 (C) exist → 6 new.

| Stage | Evals | Content | Decision |
| --- | --- | --- | --- |
| A | 6 | Remaining matrix cells | Prediction: F1 and F2 move minima toward zero, F4 away; F2 carries most of i2; no cell passes minima (1 mV) and crossings (1 ms) together. Any cell with ≠5 events is an exact fail and its supersets are skipped. |
| B | 4 | Dose split on the member with the best Δminimum/Δinterval ratio (predicted F1): m_open 1.5 and 3.0 on cell 100; then F2 m_close 0.95 and 0.8 | Rejection: if Δminimum/Δi2 is dose-invariant, no dose of F1/F2/F4 passes both rows and a sub-system outside F1–F5 is required → FAIL report. |
| C | 2 | Finalist tolerance recheck (1e-11) | As above. |

### Outputs per campaign

Manifest, ranking JSON, Youden plots, mermaid search tree with struck branches,
`campaign-result.md` issuing exactly one of PASS (one frozen set meets every row
at every calibration input) or FAIL with the reassessment question. Update
`docs/h01-causal-model.md` Y3 / Y4.

## Phase 2 — Transfer and H01 cell gate (passing candidates only)

1. Freeze; open the holdout (I 0.23 nA / E sweep 53) once; record pass or fail;
   no refit.
2. BrainCell transfer on donor geometry with `--mesh-from` (Y2 rule); numerical
   gate unchanged (0.1 ms / 0.1 mV / 0.01 ms). Cost basis: matched mesh 556 s per
   330 ms at dt 0.000625, so a full input is ≈35–40 min; these runs sit outside
   the campaign cap, each stays under one hour, each is benchmarked first. Repeat
   the human contract on the transferred trace.
3. H01 anatomy: apply the frozen candidate to I `5584343344` and E `4157825456`
   **without** the override. Pre-registered structural split on the region map
   (Y1 unresolved): (a) current map; (b) contact path assigned axon parameters as
   an explicit hypothesis map. Observe soma and contact voltage plus axial and
   channel current along the path (existing all-CV recorder). Decision: (a) no
   contact event and (b) an event → the map is the Steep X, and the causal page
   records that H01 delivery depends on an inferred axon classification, not on
   channel law. Decision limits from dt-halving pairs.
4. If Phase 1 ends in FAIL for a cell, Phase 2 is not entered for it.

## Phase 3 — Circuit gate and reusable package

- Measured I→E with qualified cells: disconnected, edge-removed, connected
  controls; E→I only if Phase 0's edge list contains a measured contact,
  otherwise the illustrative wiring stays opt-in and labelled inferred.
- Numerical: dt-halving and CV-halving pairs; decision limits by DLF.
- `docs/evidence/h01-circuit-z-map.md`: one-page energetic map (mermaid) from
  sources (applied current, ion gradients) through membrane / axial / synapse
  boundaries to observed voltages, each boundary annotated with the evidence that
  characterises it and marked measured / borrowed / inferred.
- Package: add the missing `examples/h01_ei_circuit_test.py`; extend
  `docs/tutorials/h01_braincell.rst` with a "what this circuit represents"
  section; record an installed-wheel run.
- Update causal model Y5 and `h01-implementation-status.md`.

## Files

New (worktree `.worktrees/h01-braincell`):

- `docs/specs/2026-09-05-h01-progressive-search-plan.md` — this plan as the spec
- `docs/evidence/h01_contract_score.py`, `h01_youden_plot.py`,
  `h01_campaign.py` (+ `_test.py` each)
- `docs/evidence/h01-i-campaign-manifest.json`, `h01-e-campaign2-manifest.json`
- `docs/evidence/h01-contract-scorecard.{md,json}`, `h01-pv-input-datum.json`,
  `h01-matryoshka-{i,e}.md`, `h01-resolved-edge-list.json`, campaign result
  pages, `h01-circuit-z-map.md`
- `examples/h01_ei_circuit_test.py`

Modified: `h01_index_connectivity_audit.py` (complete scan),
`h01_l2_campaign.py` (alias to the generalised runner),
`h01_pv_neuron_reference.py` (`--candidate-json`), `docs/h01-causal-model.md`,
`docs/evidence/h01-implementation-status.md`, `docs/tutorials/h01_braincell.rst`.

Not touched: `braintrace/datasets/h01_ei_profiles.py`,
`_h01_ei_parameters.py` (promotion is its own defaults spec after a PASS);
`examples/pp_prop/21-braincell-arc.py`; anything on `main`.

## Verification

- `pytest braintrace/datasets -n auto`, `pytest examples/ -n auto`,
  `pytest repo_conventions_test.py -q` (repo CI commands; no justfile here).
- Each new script's `_test.py` covers contract parsing, ratio/verdict logic,
  DLF arithmetic, manifest cap and stage-gate refusal, per-run abort, edge-list
  rejection of same-cell endpoints.
- A campaign opens only after one benchmarked evaluation and a reported total
  wall time; each phase opens only when its gate files exist.
- Final deliverable check: `examples/h01_ei_circuit.py` runs from an installed
  wheel; its evidence page states, per element, measured / borrowed / inferred.
