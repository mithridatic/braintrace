# H01 donor import: Allen human specimen 527952884, L4 pyramidal (SP6c, second donor)

Programme: [population programme](2026-09-07-h01-population-programme.md) SP6c and SP6d.
Survey: [h01-donor-survey.md](../evidence/h01-donor-survey.md) (import 2). Registry:
[re-key spec](2026-09-07-h01-donor-registry-rekey.md). Pattern: [HL5MN1 import](2026-09-07-h01-donor-hl5mn1-import.md);
the existing L2 donor's acquisition and reproduction: [L2 source reproduction](2026-09-05-human-l2-source-reproduction.md).
Worktree `h01-donors` / `feat/h01-donors`. Written before code (AGENTS.md rule 7). Compute: this
task runs no NEURON simulation (the machine is shared by measured runs); acquisition, verification,
registry, driver readiness and manifest only.

## Donor

| Field | Value |
| --- | --- |
| Model | Allen Cell Types perisomatic model 626170709 (`Biophysical - perisomatic_H16.06.008.01.31.06`, template 329230710, the same template as the imported 626170538), bundle `http://api.brain-map.org/neuronal_model/download/626170709` |
| Licence | Allen Institute Terms of Use (https://alleninstitute.org/terms-of-use/): research and non-commercial use, citation required ("Allen Cell Types Database, Allen Institute for Brain Science, celltypes.brain-map.org"), no commercial redistribution without written permission |
| Fitted cell | Allen human specimen 527952884 (`H16.06.008.01.31.06`), MTG layer 4, spiny dendrites, apical truncated, female 24 y, epilepsy resection, left hemisphere |
| Recordings | NWB `http://api.brain-map.org/api/v2/well_known_file_download/618205555` (`527952752_ephys.nwb`, ephys result 527952752), IVSCC pipeline 1.0, 50 kHz |
| Fit sweeps | 69, 70, 71, 72 (`fit_parameters.json` `fitting[0].sweeps`): four repeats of the `Square - 2s Suprathreshold` protocol at 100 pA, step 1020-3020 ms, Allen spike counts 20, 17, 19, 20 |
| Mechanisms | The genome uses Im, Ih, NaTs, Nap, K_P, K_T, SK, Kv3_1, Ca_HVA, Ca_LVA, CaDynamics (soma only) plus `g_pas` per region: exactly the eleven mechanisms of the imported 626170538, and the eleven `.mod` files in the bundle are byte-identical to `.cache/human-pyramidal-l2/source-model/*.mod`. The bundle also ships Kd, Kv2like, NaTa, NaV, Im_v2 that the genome does not reference; they are not compiled |
| Passive | `Ra 14.997`, `cm` 1.0 (soma, axon) and 1.5826 (dend, apic), `e_pas -80.818`, `v_init -80.818`, 34 C, `ena 53`, `ek -107`, junction potential -14 mV |

## Acquisition (step 2)

Cache `.cache/human-pyramidal-l4/` in this worktree (gitignored). Every file, URL, size and sha256
goes to the versioned `docs/evidence/h01-l4-acquisition.json`, written by
`docs/evidence/h01_l4_acquisition_audit.py` (+ `_test.py`) in the pattern of
`h01_pv_acquisition_audit.py` / `h01_sst_acquisition_audit.py`:

- fetched: the model bundle zip (unpacked to `source-model/`: `fit_parameters.json`,
  `reconstruction.swc`, `ephys_sweeps.json`, `manifest.json`, `model_metadata.json`,
  `modfiles/*.mod`), the NWB, the fit JSON by its own well-known file (626185209; must equal
  `fit_parameters.json` byte for byte), `allen-cell-detail.json`
  (`ApiCellTypesSpecimenDetail` row) and `neuronal-models.json` (the specimen's `NeuronalModel`
  rows) as the L2 acquisition did;
- NWB checks: `generated_by == [pipeline, IVSCC, version, 1.0]`, `aibs_specimen_id == 527952884`,
  `aibs_specimen_name == H16.06.008.01.31.06`, every exported sweep at 50,000 Hz, stimulus name
  as listed, step window read from the stimulus waveform (1020-2020 ms for `Long Square`,
  1020-3020 ms for the 2-s squares), per-sweep `bias_current` read from the acquisition group,
  and `leak_pa` / `bridge_balance_mohm` read from `ephys_sweeps.json` for the same sweep;
- exports in the L2 driver's waveform format (`h01_l2_sweep_export.export_sweep`: `time_ms` from
  0 in 0.02 ms steps, `reported_voltage_mv`, `corrected_voltage_mv` = reported - 14 mV,
  `command_current_na`, scalar `bias_current_na`, `total_current_na`), not cropped, so the
  released L2 driver plays them unchanged: `sweep-69.npz` (fit sweep, 100 pA, 2 s),
  `sweep-39.npz` (`Long Square` 90 pA, 1 s, the lower recorded input), `sweep-66.npz`
  (2-s square 60 pA, near rheobase; repeats 66-68 give 1, 2, 1). The fit sweeps share one
  amplitude, so "highest-amplitude fit sweep and one lower" resolves to sweep 69 and a lower
  input from the same recording session;
- LJP note: Allen stores voltages uncorrected for the measured -14 mV liquid junction potential;
  the export applies it once in `corrected_voltage_mv`; the model's internal voltage is
  compared to the corrected trace (the L2 rule: never apply the correction twice);
- driver layout `.cache/human-pyramidal-l4/neuron-reference/`: the eleven genome `.mod` files
  copied from `source-model/modfiles/`, `527952884_fit.json` (copy of `fit_parameters.json`),
  `morphology.swc` (copy of `reconstruction.swc`), the three `sweep-*.npz`, and `donor.json`
  (fit and morphology names, their sha256, model and specimen id, the exported sweep list).

## The verifications the survey demands (step 1; outcomes recorded in the JSON)

1. Mechanism set: the set of `mechanism` values in the genome equals
   `{Im, Ih, NaTs, Nap, K_P, K_T, SK, Kv3_1, Ca_HVA, Ca_LVA, CaDynamics}` and each of those
   eleven `.mod` files is byte-identical (sha256) to the L2 import's copy. Recorded as
   `mechanism_set_identical: true` with the per-file hashes; the five unused bundle files are
   listed as `unused_bundle_mod_files`.
2. Fit-sweep protocol: sweeps 69-72 are `Square - 2s Suprathreshold`, 100 pA, 1020-3020 ms,
   50 kHz, with `leak_pa` -16.73 / -19.63 / -19.63 / -17.30 pA and bridge balance 9.16 Mohm
   from `ephys_sweeps.json`; the NWB `bias_current` of each equals its `leak_pa` within 0.01 pA.
   The -20 mV crossing counts read from the NWB equal Allen's `num_spikes` (20, 17, 19, 20).
3. Per-sweep bias and bridge: recorded for every exported sweep and for the whole sweep table
   (76 sweeps); `Long Square` 40-43 carry bias 0, sweep 39 carries +2.33 pA, the 2-s squares
   -16.7 to -19.6 pA. The bias is the holding current and is not injected by the driver
   (command-only input, the PV and L2 convention; `--include-recorded-bias` stays available).

## Parameters and registry (step 3)

`docs/evidence/h01_l4_parameters.py` turns `fit_parameters.json` into the `_h01_ei_parameters`
region tuple format `(family, cm, g_pas, ((mechanism, density), ...), (decay, gamma) | None)`
for soma, axon, dend, apic: `cm` from `passive[0].cm`, `g_pas` from the genome rows with an
empty mechanism, channels in genome order, calcium from `decay_CaDynamics` and
`gamma_CaDynamics`. Its test asserts density-by-density equality between the generated tuple
and the committed constant `L4_ALLEN_527952884_SOURCE` (skipped when the cache is absent) and,
as a check of the generator itself, that the same function applied to the cached L2 fit
`541563728_fit.json` reproduces the committed `E_SOURCE` (skipped when that sibling cache is
absent). Facts the generator must reproduce: soma `g_pas 4.325e-4`, NaTs 2.0538, Nap 1.98e-3,
K_P 8.37e-5, K_T 5.16e-4, SK 0.4842, Kv3_1 0.4492, Im 3.41e-4, Ih 5.766e-3, Ca_HVA 9.61e-4,
Ca_LVA 9.65e-3, calcium (175.17, 7.43e-5); axon `g_pas 2.68e-4`, no channels; dend cm 1.5826,
`g_pas 1.63e-5`; apic cm 1.5826, `g_pas 1.04e-7`; no channel outside the soma (perisomatic fit).

Registry record `l4-pyramidal-allen-527952884`: layer L4, class pyramidal, modifiers (),
polarity E, subtype `regular-spiking pyramidal (spiny, apical truncated)`, profile
`h01-l4-allen-527952884`, channel_prefix `H01L2` (mechanism set identical), ena 53, ek -107.
Physiology: initial -80.81838607788086 mV (`v_init`), leak reversal -80.81838607788086 mV in
both modes (`e_pas`), Ra 14.9970627156 ohm cm, digest = sha256 of `fit_parameters.json`
(`1aa0e2c5...`); `DONOR_REGIONS[key] = {candidate: SOURCE, source: SOURCE}` (a fresh donor has
no candidate). `channel_controls` returns nothing for it in either mode. Expected `donor_for`
change: L4 pyramids without modifiers (18 cells) resolve to the new key by rule 1; the four
`sparsely-spiny` L4 pyramids resolve to it by rule 2 (match `modifier`); L2 pyramids stay on
541563728; every other E type still falls to the polarity default. The population match count
rises from 37 to 55; `h01-population-types.{json,md}` are regenerated.

## Driver readiness (step 4; the simulation is registered, not run)

`h01_l2_neuron_reference.py` gains `--donor-json PATH`, default None = today's behaviour
(`--cache/541563728_fit.json`, the pinned L2 fit hash, `source-model/morphology.swc`,
`DRIVER_SWEEPS` choices and the holdout seal). With a donor file, the fit and morphology are
the paths it names relative to its own directory, the fit hash must equal its `fit_sha256`,
the sweep must be one of its `sweeps` (the L2 seal does not apply; the donor's list is its
own registration), the waveform is `<dir>/sweep-<n>.npz`, and the report records `donor`
(key, model id, specimen id, morphology sha256, donor-file path) instead of the L2 ids. The
axon replacement (two 30 um sections of 1 um) and every other setup step are unchanged.
Defaults leave every retained L2 report byte-identical in its inputs. Tests: default path
unchanged (existing tests), donor sweep outside the donor list rejected before setup, wrong fit
hash rejected, `--help` lists the option; the actual-image `--help` check is recorded in the
decision JSON.

Manifest `docs/evidence/h01-l4-reproduction-manifest.json` (runner `h01_campaign.py`, cap 4,
abort 900 s, max 2 concurrent, image `braintrace-h01-neuron:9.0.2`, `cache_dir
.cache/human-pyramidal-l4`, `library neuron-reference`): candidate `source` at inputs `69`
(fit sweep, 100 pA, `--stop-ms 3520`) and `39` (90 pA, `--stop-ms 2520`), fixed step
dt 0.005 ms (the L2 source reproduction's step: the LCM of 50 kHz input and 40 kHz target);
repeat candidate `source-dt-half` at dt 0.0025 ms for the numerical pair. Decision limit per
row = |pair difference| x 2.95. Mod files compiled in the container (`nrnivmodl` in
`neuron-reference/`) before the first run. Stage decision
`docs/evidence/h01-donors/stage-allen-l4-decision.json` with status `untested` and the ready
commands.

Pre-registered (before any run):

- Prediction: the published fit reproduces the Allen spike counts exactly at both inputs:
  20 at sweep 69 (100 pA, 2 s; repeats 69-72 give 20, 17, 19, 20) and 12 at sweep 39 (90 pA,
  1 s); first-spike latency within the dt-pair limit of 32.18 ms and 33.88 ms after step onset.
- Rejection: a count differing from the Allen sweep by more than 1 at either input (17-20 is
  the human repeat band at 100 pA; a model count outside it also fails the repeat row), or a
  dt pair whose count differs (numerics not converged: "time level open", no verdict).
- Unchanged: the eleven mod files, every value of `fit_parameters.json`, 34 C, initial
  -80.818 mV, the recorded command waveform, bias 0, nseg factor 1, no CVode.
- A killed or aborted run is untested, never negative. Wall clock is recorded by the runner;
  no duration is quoted before it is measured.

## Usable-tier plan (SP6d, both tiers printed)

`h01_usable_tier.py` gains cell `L4-PYR`: pulse 1020-3020 ms for input 100 pA -> `sweep-69.npz`
vs `h01-l4-reproduction/source-69`, pulse 1020-2020 ms for 90 pA -> `sweep-39.npz` vs
`source-39`; repeats 69-72 as `repeat_inputs {"100 pA": (69, 70, 71, 72)}` so count and
first-spike rows get a human repeat limit; sweep 39 has no repeat and its rows carry only the
dt-pair limit. A pass is not required to enter the population: the donor enters labelled
"type-matched, usable verdicts: ..." with the printed verdicts. Outputs
`h01-l4-reproduction.{json,md}`; `docs/h01-causal-model.md` Y6 gains the second-donor entry in
the same commit as this import.

## Out of scope

Running the reproduction (machine shared); the usable-tier code change (with the run);
BrainCell transfer of the new donor (`H01L2_*` already carry the mechanisms; the transfer needs
the SP2 matched-mesh gate on its own recordings); any parameter change to the published fit;
further donors (cap two).
