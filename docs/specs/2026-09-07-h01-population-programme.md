# H01 population programme (SP0-SP9)

Status: approved by the user on 2026-09-07 (plan mode). This spec binds the sub-project
specs written under `docs/specs/` as each SP opens. Method: Hartshorne 2020. Companion
specs: [usable circuit](2026-09-07-h01-usable-circuit-plan.md), [E continuation](2026-09-07-h01-e-bounded-continuation.md),
[verified network](2026-09-06-h01-verified-network.md), [acceptance contract](2026-09-05-h01-recorded-response-acceptance-proposal.md).


## Context

H01 (`.worktrees/h01-braincell`, branch `feat/h01-braincell`, tip fc3f96d) set out to build
human-constrained active E and I cells on measured H01 anatomy and use them in a small
spiking circuit (`docs/specs/2026-09-04-h01-active-circuit.md`: "the goal is NOT complete").
Every bounded campaign since has closed at its cap without a pass:

- E donor (Allen 541563728 L2 pyramid), frozen B3: 10/10 spikes at 310 pA, 12/13 at 350 pA,
  but 8/5 at 250 pA (rate FAIL); widths fail cycle2@310 and cycle3@350; AHP unresolvable.
  Sweep53 and sweep55 holdouts spent; E currently has NO sealed holdout. Sweep43 not rescored
  on B3. `h01-e-continuation-result.md`: a new bounded causal hypothesis is needed for the
  input-response relation; another dose fitted at 310 pA is not supported.
- I donor (HL5BN1 PV): axon-first at all inputs; post-trough burst and accommodation
  unexplained; one energetic evaluation in reserve; Noise1 sweep48 sealed but its replay
  path (Vector.play) does not exist in `h01_pv_neuron_reference.py`.
- Transfer: PV BrainCell vs NEURON gives 42 vs 40 events, event 7 crossing error -0.193 ms vs
  0.1 ms; halving dt does not repair. Standing directive (`h01-implementation-status.md`):
  no serial physiological tuning while this stands.
- Pair: literature receptor moves the E soma +0.008 mV (attenuation 110), depolarising because
  the model E rests at -84 mV; no E spikes in either control; inhibition unverified.
- Network: 4 cells / 2 projections / 75,605 compartments build in 163 s; init + compile + ONE
  0.005 ms step 157 s (measured); no throughput recorded; `make_h01_network` raises on empty
  contacts (`h01_network.py:97`) so at most 4 cells build today; only `--disconnected` exists.
- Population: 30/104 cells type-match a donor; L4 pyramids (22), L2/L3 interneurons (14),
  L5 pyramids (7), L3 pyramids (6) are the gaps; `full_export_scanned` false (3,000-sample
  rescan checkpointed at 26/104, resumable).
- Stale docs: `docs/h01-network.md` still says execution is blocked; `h01-network-execution.log`
  is the old MemoryError log; adapter spec does not exist.

### User decisions (2026-09-07)

| Decision | Choice |
| --- | --- |
| Done means | Full population deliverable: usable-tier cells via one more bounded split each, transfer gates, functional inhibition, per-type human donors, multi-step population with matched controls, close-out and merge |
| Example 21 adapter | Deferred: interface-contract spec only, no code |
| Compute | Local Docker, detached PowerShell jobs; benchmark throughput first; any job expected over 15 min needs a measured estimate and per-job approval; never quote an unmeasured duration |
| Donors | Human-recorded fits only; donor-less types stay borrowed and labelled |
| Orchestration | Parallel subagents in separate worktrees and the Workflow tool are allowed |

### Method (Hartshorne 2020, `Diagnosing Performance and Reliability/pages-md/`)

Matryoshka classification before any split (p099); only contrasts >=10x the allowance are
searched (p033); decision limit = repeat range x DLF, 2.95 for two repeats (p199-p201);
prediction, rejection and unchanged quantities registered before every run (p145, p194); a cap
is a stop, and at cap without a pass the report is FAIL plus a named family/data reassessment
(p034, p179); tolerance is set for function and repeatability must be under half of it (p130);
the end of a search is a causal explanation (necessary and sufficient conditions plus mechanism),
never a list of root causes (p048, p046). Both tiers (1 mV contract + usable tier) in every
physiological verdict; decision JSON beats prose; a killed run is untested, never negative.

## Sub-projects

Each SP gets its own spec under `docs/specs/` before code (AGENTS.md rule 7), co-located
`*_test.py`, and ends with a `*-decision.json`, a result page derived from it, and the
matching `docs/h01-causal-model.md` update (see the causal-model rule in SP0). Section map:
SP1 -> Measurement function qualification (solver cost and limits); SP2 -> Y2; SP3 -> Y4;
SP4 -> Y3; SP5 and SP8 -> Y5; SP6 -> a new Y6 "per-type donors" section; SP7 -> Y5 edge list.

### SP0. Programme spec, worktrees, run governance (first, one session)

- Write `docs/specs/2026-09-07-h01-population-programme.md` binding SP1-SP9, caps, ordering,
  and the compute rules: detached jobs via `Start-Process`, progress line at least every 60 s,
  watchdog kill at 10 min without progress (killed = untested), measured estimate + approval
  for anything predicted over 15 min, SP1 runs with the machine otherwise idle.
- Add `.gitignore` for `docs/evidence/**/*.npz` raw traces (currently untracked, not ignored).
- **Causal-model rule (every SP, not only close-out).** `docs/h01-causal-model.md` is updated in
  the same commit as the decision JSON whenever a run establishes a direct observation of the
  system's behaviour: a prediction held or missed, a boundary shown to control a row, a
  mechanism excluded, a numerical limit measured. Each entry states the observed behaviour, the
  conditions under which it held, the decision JSON it comes from, and whether it closes,
  narrows, or leaves open the Y-section it belongs to (Y2 transfer, Y3 I cell, Y4 E cell, Y5
  pair/population). Only supported conclusions enter the explanations; unresolved links stay in
  the Unresolved list. A result page without a matching causal-model commit is not mergeable.
- Worktrees, one session each (concurrent sessions in one worktree are forbidden):

| Worktree / branch | Sub-projects | Files it owns |
| --- | --- | --- |
| `h01-braincell` / `feat/h01-braincell` | SP0, SP9, merges | specs, status pages, causal model |
| `h01-registry` / `feat/h01-registry` | SP6a (serial, first) | `h01_cell_types.py`, `h01_ei_profiles.py`, `_h01_ei_parameters.py`, `h01_ei_cell.py`, `h01_network.py:141`, tests |
| `h01-transfer` / `feat/h01-transfer` | SP2 | `h01_pv_braincell_reference.py`, new `h01_l2_braincell_reference.py`, `h01_l2_cell.py` |
| `h01-e-split` / `feat/h01-e-split` | SP3 | `docs/evidence/h01-e-gain/`, `h01_l2_sweep_export.py`, `h01_l2_neuron_reference.py` |
| `h01-i-reserve` / `feat/h01-i-reserve` | SP4 | `h01_pv_neuron_reference.py`, new trough audit |
| `h01-network` / `feat/h01-network` | SP1 then SP8 | `h01_network.py`, `examples/h01_verified_network.py`, new throughput script |
| `h01-connectivity` / `feat/h01-connectivity` | SP7 | `h01_c3_edge_list.py` resume wrapper, `h01_endpoint_recheck.py`, `h01_connectivity_audit.py` |
| `h01-donors` / `feat/h01-donors` | SP6b-d | new donor modules, acquisition audit, survey page |

Merge order into `feat/h01-braincell`: registry, transfer, (e-split, i-reserve), network,
connectivity, donors, close-out. `main` only after user approval (AGENTS.md rule 6).

### SP1. Network throughput benchmark (runs first, machine alone; gates SP8)

- New `docs/evidence/h01_network_throughput.py` (+test): launches
  `examples.h01_verified_network --build --dt-ms 0.005 --duration-ms D` for D in {0.05, 1, 10}
  ms as detached processes, samples child RSS (`psutil`), parses `construction_seconds` and
  `initialization_and_run_seconds` from `*-build.json`. Fit T(steps) = a + b*steps; a = init +
  compile, b = s/step, 200*b = seconds per simulated ms. Repeat the 1 ms point once: decision
  limit on b = range x 2.95. One `--solver staggered` control at 1 ms if it fits 15 min.
- Order: 0.05 ms (must finish near the 157 s anchor), 1 ms, then 10 ms only if a + 2000*b
  predicted from the first two is under 15 min, else stop and report.
- Prediction: init+compile dominates at 0.05 ms. Rejection: three points not linear within the
  1 ms repeat limit, in which case SP8 sizes from the 10 ms point directly.
- Sizing rule for SP8: a and b assumed proportional to CVs (DHS is O(N)); largest-component
  skeleton nodes 2,804,549 vs 245,769 for the 4 built cells gives about 11x (derived, not
  measured); validated at a 12-cell intermediate build before 104. Minimum deliverable duration
  50 ms (10,000 steps), longer only if measured throughput allows.
- Cap 5 runs. Output `h01-network-throughput.json` + page.

### SP2. Transfer isolation on donor geometry (standing directive; gates every promotion)

Reuse `h01_pv_transfer_isolation.py` (`decide()`), `h01_ei_spike_transfer.compare_spike_transfer`
(0.1 ms rise / 0.1 mV peak / 0.01 ms width, equal count), `--mesh-from` in
`h01_pv_braincell_reference.py`, existing NEURON finalist traces `h01-i-energetic/e-kv3-close2-*`.

| Arm | What | Runs |
| --- | --- | --- |
| A0 | BrainCell MaxCVLen 2.5 (current), full 270-1270 ms train at 0.27 nA | 1 |
| A1 | BrainCell mesh copied from the NEURON x9 geometry, full train at 0.19 and 0.27 (0.23 as spent-holdout control) | 3 |
| B1 | NEURON fixed step at the BrainCell dt, 0.19 and 0.27 | 2 |
| Repeat | dt-halving pair in each simulator at matched mesh, 0.27, 330 ms | 2 |

Decision: A1 passes every event at both inputs against CVode reference and B1 -> "mesh"
confirmed for the full train, causal model Y2 closed. A1 fails late events while B1 passes ->
implementation difference at event k (channel-by-channel state comparison named as the next
split; Y2 reassessed). Both fail -> integration. Prediction: A1 passes all events. Rejection:
any A1 event beyond its gate while the halving pair is under half the gate. Cost anchor: matched
BrainCell 556 s per 330 ms at dt 0.000625 (measured) -> full train about 2,500 s (derived;
needs approval). Try dt 0.005 with its own halving pair first.

E side, needed before any E promotion: `make_l2_cell(..., cv_policy=None)`; new
`h01_l2_braincell_reference.py` with `--mesh-from` reading the L2 NEURON report `sections`
(name, nseg) and `--sweep` from `.cache/human-pyramidal-l2/sweep-NN.npz`; generate
`_h01_ei_parameters.E_B3_EXPERIMENTAL` from `h01-e-usable/b3-sk035-ca-decay-sweep53.json`
`applied_genome` with a test asserting density equality with the NEURON report; fix
`_paint_profile` (`h01_ei_cell.py:52-55`) so a region carrying channels but `calcium=None`
(the E axon row) still gets Na/K ions. Gate identical to the I gate over 1020-2020 ms at sweeps
50 and 53; benchmark 100 ms first (E BrainCell cost unmeasured).

Then the H01-anatomy gate for a passing cell: profile on I `5584343344` / E `4157825456` via
`make_h01_ei_cell` without the closing override, with the pre-registered region-map split from
`2026-09-06-h01-progressive-search-plan.md` Phase 2.3, all-CV recorder (`h01_i_all_cv.py`).
Cap: I 8 BrainCell + 2 NEURON runs; E 4 BrainCell runs. Stop: halving pair exceeds half the
gate (report "time level open").

### SP3. E cell: one bounded causal split on the input-response relation (cap 4 evaluations)

Matryoshka reading of `b3-all-inputs-usable.md` (no run): late cycles are uniform within each
input, so the error is a gain (f-I slope) error repeating across cycles: model 0.033 spikes/pA
from 250 to 310 pA vs human 0.083. SK doses shift the curve; they cannot rotate it, which is
why B3 fixed 310 and overfired 250. Rest -84 vs literature -72 mV is parked, not searched: the
donor's own pre-pulse samples match the model within 1 mV (`h01-matryoshka-e.md`, rows
1019-1120 ms). Controlling property: a current present between spikes at low input and absent
at high input, i.e. Ih (distributed x75 in the candidate) or the linear leak.

Before any run: seal a new E holdout (an unopened higher-amplitude sweep from
`long-square-index.json`) via `SEALED` in `h01_l2_sweep_export.py`; add sweep 56 (200 pA,
repeats 56/59/60/61/62 already used for AHP spread, so it is calibration, not a holdout) to the
driver choices at `h01_l2_neuron_reference.py:79` and the exporter, tested together per the
continuation spec's prevention rule.

- Stage 0 (evaluation 1): B3 at sweep 43 and sweep 56. Prediction: sweep-43 late-return rows
  within 1 mV (F5 untouched by B3); 200 pA count reported against the repeat band, giving the
  E rate row its only repeat-based decision limit. Rejection: sweep 43 fails 1 mV -> B3's Stage B
  levers broke F5 and every arm must be scored on 43 too (they are regardless).
- Stage G (evaluations 2-4), every arm at sweeps 43, 50, 53, 56:

| Arm | Flag | Prediction | Rejection |
| --- | --- | --- | --- |
| G1 `g-ih-half` | `ih_density_factor 37.5` (from 75) | 250 pA count 5-7, late cycles >= 200 ms; 310 pA count 9-10, rate within 1.5 Hz; baseline -1 to -3 mV; sweep-43 return moves <= 1 mV | 310 count < 9, or 250 count still >= 8 |
| G2 `g-leak-150` | `leak_factor 1.5`, reversal unchanged | both counts fall, 250 proportionally more (rheobase shifts right) | 250 and 310 fall by the same spike count (pure shift) |
| G3 reserve | dose of the better arm by delta(250 count)/delta(310 rate) | 250 count 5 +/- 1, 310 rate within 1.5 Hz, sweep 43 within 1 mV | as above |

Unchanged in every arm: all B3 flags, nseg factor 9, CVode 1e-10, stop 2100 ms, axon-first
initiation (`h01_initiation_score.py`). Manifest `docs/evidence/h01-e-gain-manifest.json`
(runner `h01_campaign.py`, cap 4, `prior_evaluations 0`, abort 1500 s, max 2 concurrent, stage
gates `stage-g0-decision.json` -> `stage-g-decision.json`). Scoring: `h01_usable_tier.py`
(new run stems), `h01_contract_score.py`, both tiers per input. PASS = one profile with rate
and adaptation pass at 250/310/350 (and 200 if resolvable) and no width row newly failing.
FAIL at cap names the reassessment: gain set outside the fit's passive family (human slow Na
inactivation or Kv7/M kinetics absent from the Allen genome, or a second human L2/3 donor with a
recorded f-I curve). Cost anchor 649-665 s per 250/310 run (measured); 43 and 56 unmeasured;
total reported before the manifest opens.

### SP4. I cell: the single reserve evaluation; Noise1 stays sealed

- Stage 0, no evaluation: `docs/evidence/h01_i_trough_audit.py` (+test) on the retained
  `h01-i-energetic/e-kv3-close2-{019,027}.npz` (NaTg h/m, Kv3 m, axon voltage, axial
  neighbours): soma h and axial current into the soma at each trough, trough+6 and trough+10 ms,
  finalist vs source. Pre-registered rule: h >= 0.7 at trough+6 -> availability is not the
  limit, the reserve tests post-trough axial drive; h < 0.5 -> it tests recovery.
- Driver: add a repeatable `--scale MECH:REGION:FACTOR` to `h01_pv_neuron_reference.py`
  (mirror `parse_regional_density` in `h01_l2_regional_density.py`), keep the old single slot
  for hash stability, sibling test, actual-image `--help` check.
- Evaluation (0.19 and 0.27; 0.23 is spent), manifest `h01-i-reserve-manifest.json`, cap 1:
  drive arm `--scale NaTg:axon:1.5` with somatic NaTg x1.1 retained, or recovery arm
  `sodium_h_recovery_factor 0.5`. Prediction: cycle 2 at 0.27 shortens from 18 to <= 12 ms,
  count 37 -> 40-43, width 0.221 +/- 0.006 unchanged, trough within 1 mV. Rejection: cycle 2
  unchanged within its DLF limit or the loop breaks (peak > +22 mV or width outside 20 %).
- Noise1 sweep48: not built now (about 20x a long-square run per the seal, unmeasured). If
  the reserve passes rate and adaptation at both inputs, amend the seal before any export to a
  prediction on the first 2 s only, build Vector.play, open once. Otherwise the I cell stays
  "experimental, borrowed, labelled".

### SP5. Functional inhibition at the measured pair

Extend `h01_ie_pair_response.py`: E driven so the disconnected control fires >= 3 spikes in the
window (stimulus choice registered, not fitted); I pulse train through the closing-restored
wrapper (`h01_i_source_closing_circuit.py`) landing between E spikes. Arms: disconnected;
literature receptor at the measured site `cable_location [2805, 0.93]`; literature receptor at
the E soma (explicit perisomatic hypothesis, labelled inferred); dt-halving pair on arm 2.
Retained per E cycle: cycle length, spike-time shift, IPSP at site and soma. Prediction: arm 2
soma IPSP < 0.05 mV and no spike moves beyond the halving limit; arm 3 soma IPSP >= 0.3 mV
(from the pinned unitary current and E input resistance in `h01-ie-synapse-literature.json`)
and at least one E spike delayed beyond the limit. Rejection of the placement hypothesis: arm 3
also below 0.05 mV. Gate: functional inhibition = one E spike shifted or one cycle lengthened
beyond the DLF limit; both human tiers "not applicable" and stated so. Cost anchor 170 s per
40 ms run (measured); benchmark 100 ms before a 300 ms window. Cap 6 + 2 halving runs; run
stage 2 only if SP2/SP3 promote a profile.

### SP6. Per-type human donors

- SP6a registry re-key (serial, small): `h01_cell_types.DONORS` keyed by donor key with layer,
  class, modifiers, polarity, profile, source, channel_prefix; `donor_for(kind)` precedence
  (layer, class, modifiers) -> (layer, class) -> class -> polarity default; keep
  `DONORS_BY_POLARITY` and `get_ei_profile(polarity, mode)` as aliases so existing tests hold;
  `EIProfile.channel_prefix` replaces the `"H01L2" if E else "H01PV"` literal at
  `h01_ei_cell.py:40`; `make_h01_ei_cell(..., donor=None)`; `h01_network.py:141` resolves the
  donor from tags and records it per cell in build evidence; regenerate
  `h01-population-types.{json,md}` with a `donor_key` column.
- SP6b survey (Workflow: N searchers + judge, read-only): criteria = human species; layer and
  class match a gap; published fit with obtainable mechanism files and licence; the recordings
  behind the fit obtainable (so a contract can be scored); protocol readable. Leads to verify,
  not assert: ModelDB 267587 HL5PN1/HL5PN2 (named in `.cache/human-pv/source-tree.json`, same
  mechanism library as HL5BN1), Allen human biophysical models by layer, Eyal et al. human L2/3,
  Kalmbach/Berg supragranular. Output `docs/evidence/h01-donor-survey.{md,json}`; judge picks
  at most two imports.
- SP6c import per donor (cheapest first): acquisition audit (`h01_pv_acquisition_audit.py`
  pattern), NEURON reproduction (`--template/--biophys` flags on the PV driver if the HL5 family
  shares the template), `_h01_ei_parameters.<KEY>_SOURCE`, geometry reference JSON, channel
  module only if the mechanism set differs, cache under `.cache/human-<donor>/`.
- SP6d gate to enter the population: fit reproduced at >= 2 recorded inputs within a
  dt/tolerance pair; usable tier scored and printed (a pass is not required to enter; the donor
  enters labelled "type-matched, usable verdicts: ..."); BrainCell transfer at matched mesh
  passes the SP2 gate. Types without a surviving candidate stay on the polarity donor, labelled.
  Cap two donors.

### SP7. Connectivity scan completion

- SP7a: resume `h01_c3_edge_list.py --limit 3000` from the 26/104 checkpoint under a PowerShell
  watchdog (kill at 10 min without a cell line, at most 3 relaunches). Anchor 98-110 s per cell
  (measured) -> about 2.1-2.4 h for 78 cells (derived; needs approval).
- SP7b: `full_export_scanned` lives in `h01-measured-connectivity-summary.json`; benchmark one
  of the nine local avro files with `h01_connectivity_audit.py`, then finish or record "N of 9".
- SP7c: merge check on the 71-candidate pair (shared C3 labels, same proofread label).
- SP7d: 5-voxel recheck (`h01_endpoint_recheck.py`, 17-19 s each measured) on any newly
  endpoint-verified edge, then `prepare_connectivity` -> topology for SP8.
Gate: either `full_export_scanned true` or an explicit coverage statement; never "absence
established".

### SP8. Population simulation with matched controls (after SP1, SP6a, SP7)

- `make_h01_network`: `include_isolated` (build isolated cells from their soma-bearing largest
  component per `h01-population-components.json`, soma output site, no projections; never join
  fragments); `control in {ei, e_only, i_only, disconnected}` mirroring
  `braintrace/datasets/h01_ei_circuit.py:113-115`, enabled edges recorded per control.
- `examples/h01_verified_network.py`: `--control`, `--include-isolated`, `--cells N` (incident
  cells first, then smallest components), init/compile time separated from stepping if the API
  allows.
- Staged detached builds, each approved from the previous measurement: 4 (SP1) -> 12 -> 40 ->
  104 or the largest N that fits. Build-only first, then run. Four control runs at final N with
  identical cells, inputs, dt, duration. dt-halving pair on `ei` over the first 10 ms for
  decision limits.
- Gate: all traces finite; per-cell spike table; conductance arrival only where the pre cell
  fires; `disconnected` shows none; `e_only`/`i_only` differ from `ei` only at removed edges'
  receivers. Page `docs/evidence/h01-population-run.md` states N of 104, enabled edges,
  assumed inputs, profile provenance per cell, "construction and delivery, not physiology".
- Stop: RSS at 12 cells projects past available memory -> reduce N.

### SP9. Close-out

1. Docs truth pass (Workflow: one agent per page, judge merges a claim/JSON/value/match table
   into `docs/evidence/h01-docs-truth-pass.md`; prose fixed, JSON never): `docs/h01-network.md`,
   `h01-network-execution.log` (git mv to a dated memory-failure name), `h01-population-status.md`,
   `h01-implementation-status.md` (transfer row from SP2's JSON), `h01-verified-network-validation.md`,
   `h01-usable-tier-{i,e}.md`, `docs/tutorials/h01_braincell.rst` representation table,
   `docs/h01-causal-model.md` (Y2 closed or reassessed; Y3/Y4/Y5 appended; Unresolved rewritten).
2. `docs/specs/2026-09-08-h01-example21-adapter-contract.md` (contract only): inputs
   (`make_h01_network` topology + evidence; the `_TOPOLOGY_ARRAYS` vocabulary at
   `examples/pp_prop/example21_arc_adapter.py:29-40`), state ownership vs Example 21's
   single-compartment HH, injection and readout hooks, what is not promised (learning-rule
   claims need the finite-window oracle per AGENTS.md), qualification required before any
   training claim.
3. Final "what a user gets" table: anatomy / types / physiology per type (measured,
   type-matched borrowed, polarity-borrowed) / E and I tiers per input / transfer / connectivity
   / synapses / pair / population run / holdouts / Example 21 (spec only); every cell cites its
   decision JSON.
4. Merge readiness: full suites with the Abseil preload workaround from
   `h01-verified-network-validation.md`, coverage >= 90 % on every changed production module
   (numbers recorded), `git merge --no-ff` per sub-branch, user approval before `main`.

## Dependency graph and critical path

```
SP0 -> SP6a registry (serial, small)
 |         |
 |-> SP1 bench ----------------------> SP8 population -> SP9
 |                                     ^
 |-> SP7 scan -------------------------' (new edges -> topology)
 |
 |-> SP2 transfer (I full train; L2 runner + E_B3 profile)
 |        |                    |
 |        v                    v
 |   SP3 E split -> E transfer gate -> H01-anatomy gate -> SP5 stage 2
 |   SP4 I reserve -> (pass only) Noise1 path
 |
 |-> SP5 stage 1 (diagnostic pair, current profiles)
 '-> SP6b survey -> SP6c import -> SP6d gate -> SP8 profiles and SP9 table
```

- Compute critical path: SP1 (machine alone) -> SP8 staged builds -> SP9.
- Physiology critical path: SP2 -> SP3 -> E transfer gate -> H01 gate -> SP5 stage 2.
- Parallel in separate worktrees: SP2, SP3, SP4, SP7, SP6b are mutually independent. SP1 needs
  the machine idle, so it is scheduled in a window with no containers or BrainCell runs.
- Serial: SP6a before SP2's E work and SP8's builder changes; SP1 before SP8; SP7 before the
  final SP8 topology; everything before SP9. `h01_campaign.py` caps two containers, so SP3 and
  SP4 together fill both slots and SP2's NEURON runs are scheduled around them.

## Where the Workflow tool fits

| Use | Fit | Shape |
| --- | --- | --- |
| Docs truth pass (SP9.1) | yes | one agent per page -> judge merges the claim table |
| Donor survey (SP6b) | yes | N searchers by lead -> judge applies the criteria |
| Adversarial verification of each SP result page against its decision JSON before merge | yes | one verifier per (page, JSON) pair |
| Test-suite fan-out per worktree before merge | optional | one runner per worktree, summariser |
| NEURON/BrainCell runs, campaign stages, benchmarks, GCS scans | no | detached `Start-Process` jobs with log watchdogs; `h01_campaign.py` owns caps, gates, abort |

## Risks and stop rules

| SP | Risk | Stop / mitigation |
| --- | --- | --- |
| SP1 | nonlinear step cost, RSS growth | stop at first nonfinite trace or an unapproved >15 min point; size SP8 from the largest measured point |
| SP2 | matched mesh fails late events at 0.19; dt 0.005 fails its halving gate | report implementation difference at event k; dt 0.000625 only with a measured estimate; no promotion |
| SP3 | Ih/leak arms break sweep-43 rows; no rotation | cap 4 is the stop; FAIL names the reassessment; exporter+driver joint test before any run |
| SP4 | new driver flag alters old candidate hashes | keep old flag; new repeatable flag only; image startup check |
| SP5 | E does not fire in the disconnected control | one drive change, then stop; placement stays the user's decision |
| SP6 | no human fit with obtainable recordings for L4 pyramids | type stays borrowed and labelled; no non-human donor |
| SP7 | scan stalls again | watchdog, 3 relaunches, checkpoint; report partial coverage |
| SP8 | memory or time at 104 cells | staged 12 -> 40 -> 104; largest constructible N is the deliverable |
| SP9 | prose/JSON disagreement | JSON wins; truth pass is a merge gate |

## Verification

- Each SP: its `*-decision.json` exists before its result page; both tiers printed for any
  physiological verdict; predictions scored against registered bands with misses recorded
  against the prediction.
- Code: co-located `*_test.py` for every new or changed module, >= 90 % line coverage on
  `h01_network`, `h01_cell_types`, `h01_ei_profiles`, `h01_ei_cell`, `h01_l2_cell`; existing
  tests untouched by the registry aliases.
- Runs: every detached job leaves a log with progress lines, a wall-clock record, and finite,
  time-ordered traces; over-cap outputs quarantined under `aborted/`, never scored.
- End-to-end: `examples.h01_verified_network --control ei --include-isolated --cells N` runs
  from an installed wheel for the final N and D; the four control traces load and differ only
  where the plan says; the representation table cites a JSON for every cell; full suites pass
  with the Abseil preload; `feat/h01-braincell` merges cleanly and `main` waits for approval.
