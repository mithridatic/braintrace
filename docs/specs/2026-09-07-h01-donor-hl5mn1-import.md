# H01 donor import: HL5MN1 (Yao 2022 HL23SST), L3 aspiny interneuron (SP6c, first donor)

Programme: [population programme](2026-09-07-h01-population-programme.md) SP6c and SP6d.
Survey: [h01-donor-survey.md](../evidence/h01-donor-survey.md) (import 1). Registry:
[re-key spec](2026-09-07-h01-donor-registry-rekey.md). Worktree `h01-donors` /
`feat/h01-donors`. Written before code (AGENTS.md rule 7). Compute: no run starts until
`.worktrees/h01-network/docs/evidence/h01-network-throughput.json` exists (SP1 owns the machine).

## Donor

| Field | Value |
| --- | --- |
| Model | `biophys_HL5MN1.hoc` in `agmccrei/HumanL5Circuit_AGM2022` (= ModelDB 267587), a re-badged copy of Yao et al. 2022 `biophys_HL23SST.hoc` (ModelDB 267595) |
| Licence | GPL-3.0 (`LICENSE` at the repository root, 35,149 bytes, git blob `f288702d`) |
| Fitted cell | Allen Cell Types human specimen 571700636 (`H17.06.006.11.09.05`), MTG, layer 3, aspiny dendrites, male 35 y, epilepsy resection |
| Recordings | NWB `http://api.brain-map.org/api/v2/well_known_file_download/618228061` (`571700399_ephys.nwb`, 16,422,444 bytes), IVSCC pipeline 1.0, 50 kHz, Long Square steps 1020-2020 ms |
| Subtype | unknown in H01 tags; Yao 2022 calls the cell "putative SST"; scored subtype-unknown |
| Mechanisms | CaDynamics, Ca_HVA, Ca_LVA, Ih, Im, K_P, K_T, Kv3_1, NaTg, Nap, SK: byte-identical to the eleven `.mod` files of the imported HL5BN1 (`.cache/human-pv/neuron-reference/mod`) |
| Template | `NeuronTemplate.hoc` byte-identical to the HL5BN1 one (sha256 `e4a044e7...`); for `HL5MN1` its `init` calls `delete_axon(3,1.75,1,1)` (axon[0] 20 um 3->1.75 um, axon[1] 30 um 1.75->1 um, myelin 1000 um x 1 um, cm 0.02) instead of `delete_axon_BPO` |

## Acquisition (step 2)

Cache `.cache/human-sst-l3/` in this worktree (gitignored). Every file, URL and sha256 goes to
the versioned `docs/evidence/h01-sst-acquisition.json`, written by
`docs/evidence/h01_sst_acquisition_audit.py` (+ `_test.py`) in the pattern of
`h01_pv_acquisition_audit.py`:

- pins: repository commit `dd472f19a0d1bfbbba59677cfd82c6e9f8a80590` (2022-08-22) for
  `L5Circuit/default_circuit/{models/biophys_HL5MN1.hoc, models/NeuronTemplate.hoc,
  morphologies/HL5MN1.swc, mod/*.mod}` and `LICENSE`; KantYao commit
  `4b970fb5881929d192691e68a2a146c2e97766f6` for `L23Net/models/biophys_HL23SST.hoc`;
  Allen `571700636_fit.json` (perisomatic model 626170439, well-known file 626185239) as a
  cross-check of the fit source, not as a model;
- NWB checks: `generated_by == [pipeline, IVSCC, version, 1.0]` (SI units already stored),
  `aibs_specimen_id == 571700636`, every exported sweep at 50,000 Hz, stimulus name
  `Long Square`, step from 1020 to 2020 ms at the sweep's `aibs_stimulus_amplitude_pa`;
- exports: crop from 750 ms, junction correction -14 mV (Allen reports the LJP uncorrected;
  the export applies it as the PV export did), time from 0 in 0.02 ms steps, so the step
  runs 270-1270 ms as in the PV driver; per-sweep `bias_current` read separately from the
  stimulus waveform. Exported: `active-0.npz` = sweep 44 (100 pA), `active-1.npz` = sweep 35
  (150 pA), `passive-0.npz` = sweep 23 (-90 pA). Repeats 44-47 (100 pA) and 40-43 (60 pA)
  are read from the NWB by the scorer and are not exported.

## The three verifications the survey demands (step 1, outcomes recorded in the JSON)

1. `diff biophys_HL5MN1.hoc biophys_HL23SST.hoc`: the only difference is the procedure name
   (`biophys_HL5MN1` vs `biophys_HL23SST`); every density, kinetic shift, passive value and
   `distribute_channels` call is identical. Recorded as `identical_parameters: true` with the
   diff lines.
2. GET of the NWB: HTTP 200, `Content-Disposition: 571700399_ephys.nwb`, sha256
   `218aa144e2e3b63dbcb3fe57072572a7e305a89a57b69ddb36618a4ed9d5c45b`; sweep table read
   (57 sweeps; Long Square 23-37 at -90..190 pA in 20 pA steps, repeats 38-47, noise 48-51).
3. Yao et al. 2022 methods (bioRxiv 10.1101/2021.02.17.431698 v5, JATS source): "putative
   SST (Neuron ID: 571700636), PV (Neuron ID: 529807751) and VIP (Neuron ID: 525018757)
   interneurons available from the Allen Brain Atlas"; "five hyperpolarizing and depolarizing
   current steps ... Three depolarizing supra-threshold current steps ... A small
   hyperpolarizing step ... a large hyperpolarizing current step". The SWC header names the
   same specimen (`H17.06.006.11.09.05_578627812_p_DendriteAxon.swc`). Which three
   depolarising sweeps the fit used is not stated; Allen's own perisomatic fit used sweep 32.

## Parameters and registry (step 3)

`docs/evidence/h01_sst_parameters.py` parses the hoc (`forsec` blocks, `distribute_channels`
with distribution type 0 and unit scale, `insert` lists, Ih default `gbar 1e-5` where inserted
but unset) into the `_h01_ei_parameters` region tuple format
`(family, cm, g_pas, ((mechanism, density), ...), (decay, gamma) | None)` for soma, axon,
dend (`basal`), apic. Its test asserts density-by-density equality between the parse of the
cached hoc and the committed constant `SST_L3_HL5MN1_SOURCE` (skipped when the cache is absent).
Facts the parse must reproduce: `cm 1` everywhere (myelin 0.02 is outside the four regions),
`g_pas 2.32e-5`, `e_pas -81.5`, `Ra 100`; soma NaTg 0.127, Kv3_1 0.871, K_P 0.0111, Im
1.58e-4, Ca_HVA 3.55e-3, Ca_LVA 3.14e-3, SK 0, K_T 0, Ih 4.31e-5, calcium (465, 5e-4); axon
NaTg 0.343, Nap 4.44e-4, Kv3_1 0.984, K_P 0.0295, K_T 0.023, Im 3.17e-4, SK 1.13e-3, Ca_HVA
1.45e-3, Ca_LVA 0.0627, Ih 1e-5 (default), calcium (469, 5e-4); dend Ih 9.49e-5; apic Ih 1e-5
(the SWC has no apical section; the row is what the hoc would paint if it had).

Registry record `l3-sst-interneuron-hl5mn1`: layer L3, class interneuron, modifiers (),
polarity I, subtype `unknown (putative SST)`, profile `h01-sst-l3-hl5mn1`, channel_prefix
`H01PV` (mechanism set identical), ena 50, ek -85. Physiology: initial -81.5 mV (the hoc's
`e_pas`; the hoc names no `v_init`), leak reversal -81.5 mV in both modes, Ra 100 ohm cm,
digest = sha256 of `biophys_HL5MN1.hoc`; `DONOR_REGIONS[key] = {candidate: SOURCE, source:
SOURCE}` (no candidate exists). `channel_controls` returns the PV candidate controls only for
the PV donor; the new donor gets none in either mode. `donor_match`'s note names the resolved
donor's subtype. Expected `donor_for` change: L3 interneurons without modifiers (7 cells)
resolve to the new key; L2/L1/L4/L5 interneurons still to HL5BN1; the population match count
rises from 30 to 37.

Recorded limitation for the SP6d BrainCell gate: the mechanism *set* is identical but the
kinetic parameters are not. HL5MN1's soma NaTg uses `vshiftm 13, vshifth 15, slopem 7`
(HL5BN1: 0, 10, 9), its Ih uses the mod-file default shifts (HL5BN1 sets six fitted shifts
that `h01_pv_rates` hardcodes), and its myelin has no leak. `H01PV_*` cannot reproduce HL5MN1
until these become constructor parameters; that is a channel-module change outside SP6c.

## NEURON reproduction (step 4; this is the simulation)

Driver: `h01_pv_neuron_reference.py` gains `--template`, `--biophys`, `--morphology`
(in-container paths, defaults = today's `/work/source/{NeuronTemplate.hoc, biophys_HL5BN1.hoc,
HL5BN1.swc}`); the biophysics procedure name is the hoc file stem; probes that HL5MN1 lacks
(soma Nap) are recorded only when present; the report records `donor`, the three paths and the
morphology sha256; myelin joins the geometry list when connected. Defaults leave every retained
candidate report unchanged. Tests: parse defaults and overrides, procedure-name derivation,
missing file rejected; actual-image `--help` check recorded in the result page.

Container layout `.cache/human-sst-l3/neuron-reference/{source/NeuronTemplate.hoc,
source/biophys_HL5MN1.hoc, morphologies/HL5MN1.swc, mod/}` (the template finds the cell name
after `morphologies/`, so the SWC must live under that folder). Mod files compiled in the
container (`nrnivmodl mod`) before the first run.

Manifest `docs/evidence/h01-sst-reproduction-manifest.json` (runner `h01_campaign.py`, cap 4,
abort 900 s, max 2 concurrent, image `braintrace-h01-neuron:9.0.2`): candidate `source` at
inputs `100` (0.100 nA, sweep 44) and `150` (0.150 nA, sweep 35), fixed step dt 0.025 ms,
`--initial-mv -81.5`, 1500 ms, bias 0 (command-only input; the PV acquisition audit showed the
holding current is absorbed by the fit); repeat candidate `source-dt-half` at dt 0.0125 ms
for the numerical pair. Decision limit per row = |pair difference| x 2.95.

Pre-registered (before any run):

- Prediction: the published fit reproduces the Allen spike counts exactly at both inputs:
  14 at sweep 44 (100 pA; repeats 44-47 give 14, 14, 13, 12) and 34 at sweep 35 (150 pA);
  first-spike latency within the pair limit of 24.16 ms and 13.0 ms after step onset.
- Rejection: a count differing by more than 1 at either input (sweep 44 also has a
  repeat band 12-14; a model count outside it fails the repeat row).
- Unchanged: template, mechanisms, every hoc value, 34 C, 270-1270 ms step, no bias.
- A killed or aborted run is untested, never negative. Wall clock is recorded by the runner;
  no duration is quoted before it is measured.

## Scoring plan (SP6d, both tiers printed)

`h01_usable_tier.py` gains cell `SST-L3`: pulse 270-1270 ms, inputs 100 pA -> `active-0.npz`
vs `h01-sst-reproduction/source-100`, 150 pA -> `active-1.npz` vs `source-150`; repeats
sweeps 44-47 (1020-2020 ms) with `repeat_inputs {"100 pA": (44, 45, 46, 47)}` so count and
first-spike rows get a human repeat limit. `h01_contract_score.py` conventions (1 mV, 0.05 ms,
exact count) are printed beside the usable rows. A pass is not required to enter the
population: the donor enters labelled "type-matched, usable verdicts: ..." with the printed
verdicts. Outputs `h01-sst-reproduction.{json,md}` and
`h01-donors/stage-hl5mn1-decision.json`; `docs/h01-causal-model.md` gains section Y6 in the
same commit.

## Out of scope

BrainCell transfer of HL5MN1 (needs the kinetic-shift parameters above); the second import
(Allen L4 527952884); any parameter change to the published fit.
