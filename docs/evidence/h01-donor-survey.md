# H01 per-type donor survey (SP6b, 2026-09-07)

Follow-up on 2026-09-14: the previously identified DANDI 000630 lead now has a
verified downloaded human L1 recording and matching public morphology, with
36 metadata-eligible specimens joined to published recordings. This supplies
inputs for fitting; no new donor has passed reproduction or human-only mechanism
qualification. See the [acquisition receipt and limits](h01-human-unity-20260914/human-data/README.md).

Five parallel read-only searches (ModelDB human L5 family, Allen human biophysical models,
human L2/3 pyramidal models, human interneuron models, human L4 pyramidal) returned 63
candidates; one judge applied the SP6b criteria (human species; layer and class match an H01
gap; published fit with obtainable mechanisms and licence; the fitted recordings obtainable;
protocol readable). Full rows, URLs, dead ends and the judge's ranking:
[h01-donor-survey.json](h01-donor-survey.json). Both selected donors were imported and their
published fits run (4 of cap 4 each, all rc 0) on 2026-09-07 under SP6d; both reproductions were
REJECTED by the registered count rule (HL5MN1: 16 for 14 at 100 pA, 30 for 34 at 150 pA; Allen L4:
19 for 20 at sweep 69, 8 for 12 at sweep 39). See
[h01-donors/stage-hl5mn1-decision.json](h01-donors/stage-hl5mn1-decision.json) and
[stage-allen-l4-decision.json](h01-donors/stage-allen-l4-decision.json).

## Imports executed (at most two, cheapest first)

| Order | Donor | Closes | Why | Verification (done, all pass) |
| --- | --- | --- | --- | --- |
| 1 | HL5MN1 (= Yao 2022 HL23SST), ModelDB 267587 / agmccrei/HumanL5Circuit_AGM2022, GPL-3.0; Allen specimen 571700636 (MTG L3 aspiny), NWB well_known_file 618228061 | 7 L3 interneurons without modifiers (population_match 30 -> 37), scored subtype-unknown; the gap was 14, and the 5 L2 interneurons and 2 L2 sparsely-spiny interneurons remain borrowed from HL5BN1 | Same mod/ library and NeuronTemplate.hoc as the imported HL5BN1: no new mechanisms | hoc identity with 267595 biophys_HL23SST.hoc: pass; NWB 618228061 fetched (HTTP 200, 16,422,444 bytes) and sweep table read: pass; fit source cell in the Yao 2022 methods: pass (`stage-hl5mn1-decision.json#/verifications`) |
| 2 | Allen human specimen 527952884, perisomatic model 626170709 (MTG L4 spiny), NWB 618205555 (HEAD 200), fit sweeps 69-72 | L4 pyramidal (22, largest gap) | Same Allen template (329230710) and import path as the existing 541563728/626170538 donor | mechanism set byte-identical to the imported 626170538 set: pass; sweeps 69-72 protocol confirmed (Square 2s Suprathreshold, 100 pA): pass; NWB 618205555 fetched in full (HTTP 200, 17,976,764 bytes): pass (`stage-allen-l4-decision.json#/verifications`) |

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
- At survey time only NWB 618205555 was HEAD-verified and every other recording link was read from
  API listings; both import NWBs have since been fetched and hashed (618205555: HTTP 200,
  17,976,764 bytes; 618228061: HTTP 200, 16,422,444 bytes; `h01-donors/stage-*-decision.json#/verifications`).
- Licences: Allen Terms of Use are non-commercial; ModelDB 267595 carries no LICENSE file (the
  Zenodo 5771000 copy is CC-BY-4.0), so the GPL-3.0 267587 copy of the SST model is the one to
  import.
