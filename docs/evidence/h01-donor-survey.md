# H01 per-type donor survey (SP6b, 2026-09-07)

Five parallel read-only searches (ModelDB human L5 family, Allen human biophysical models,
human L2/3 pyramidal models, human interneuron models, human L4 pyramidal) returned 63
candidates; one judge applied the SP6b criteria (human species; layer and class match an H01
gap; published fit with obtainable mechanisms and licence; the fitted recordings obtainable;
protocol readable). Full rows, URLs, dead ends and the judge's ranking:
[h01-donor-survey.json](h01-donor-survey.json). Nothing here has been imported or run.

## Imports selected (at most two, cheapest first)

| Order | Donor | Closes | Why | Verify before import |
| --- | --- | --- | --- | --- |
| 1 | HL5MN1 (= Yao 2022 HL23SST), ModelDB 267587 / agmccrei/HumanL5Circuit_AGM2022, GPL-3.0; Allen specimen 571700636 (MTG L3 aspiny), NWB well_known_file 618228061 | L2/L3 interneurons (14), scored subtype-unknown | Same mod/ library and NeuronTemplate.hoc as the imported HL5BN1: no new mechanisms | diff biophys_HL5MN1.hoc against 267595 biophys_HL23SST.hoc; GET the NWB and read its sweep table; confirm the fit source cell in the Yao methods |
| 2 | Allen human specimen 527952884, perisomatic model 626170709 (MTG L4 spiny), NWB 618205555 (HEAD 200), fit sweeps 69-72 | L4 pyramidal (22, largest gap) | Same Allen template (329230710) and import path as the existing 541563728/626170538 donor | confirm the zip's mod set equals the imported E mechanism set; confirm sweeps 69-72 protocol |

## Types with no human-recorded fit

- L1 interneuron (4): no published human L1 biophysical fit found; recordings exist without
  models (DANDI 000630, 000636). Stays borrowed and labelled.
- Interneuron subtype: Allen does not annotate PV/SST/VIP for human cells; the only labels are
  Yao et al.'s putative ones. Every interneuron donor is scored subtype-unknown.
- Active-dendrite human fits: none in the Allen style; only the Yao/Guet-McCreight hoc models.

## Corrections to the leads

- HL5PN2 does not exist in the 267587 tree; the source-tree note was wrong.
- HL5MN1/HL5VN1 are re-badged HL23SST/HL23VIP fitted to Allen L3 (571700636, MTG) and L2
  (525018757, frontal) cells, so they address the L2/L3 gap, not L5.
- Krembil-derived fits (HL5PN1, HL23PYR) are population fits against DANDI 000293; criterion (d)
  holds only at the population level and needs a different contract type. Deferred.
- Only NWB 618205555 was HEAD-verified; every other recording link was read from API listings and
  must be fetched before import.
- Licences: Allen Terms of Use are non-commercial; ModelDB 267595 carries no LICENSE file (the
  Zenodo 5771000 copy is CC-BY-4.0), so the GPL-3.0 267587 copy of the SST model is the one to
  import.
