# H01 population: what a user gets today (2026-09-07)

One table, no pass claimed. Sources are the pages linked in each row.

| Layer | What exists | What it is not | Evidence |
| --- | --- | --- | --- |
| Anatomy | 104 proofread cells, 3,327 components, all with a soma-bearing component; largest component holds a median 77 % of a cell's nodes (minimum 25 %); 73 cells have more than one soma-bearing component | Joined cells: the builder loads one component per cell and never joins fragments | [components](h01-population-components.md) |
| Cell types | Layer and class for every cell from the released tags (28 L2 pyramids, 18 L4 pyramids, 25 interneurons across L1 to L5, 27 tag combinations) | Interneuron subtypes: none in the tags | [types](h01-population-types.md) |
| Physiology per cell | Two frozen donor profiles applied by Dale sign: an L2 pyramidal fit (Allen 541563728) and a PV basket fit (HL5BN1). 30 of 104 cells match their donor in layer and class; 74 are borrowed across layer, class or morphology | Human-qualified cells: neither donor passes the 1 mV contract; under the usable tier the I cell fails rate at two of three inputs, adaptation everywhere and width (20 % narrow); the E cell fails rate at 310 pA and adaptation | [usable I](h01-usable-tier-i.md), [usable E](h01-usable-tier-e.md), [prediction I](h01-prediction-i.md), [prediction E](h01-prediction-e.md) |
| E cell repair | B2 (somatic sodium 0.9, axonal NaTs 3.814, SK 0.5, calcium decay 1.0) has 9 of 10 spikes at 310 pA; usable rate and cycle-2 width fail, AHP is unresolvable. The 5-of-5 result at 250 pA belongs to the earlier A2 profile | A finished E cell: bounded dose/calibration/prediction continuation registered | [E result](h01-e-usable-result.md), [B2 audit](h01-e-usable/b2-audit.md) |
| Connectivity | 123 candidate contacts on 31 pairs from a 1,500-sample scan; 3 endpoint-verified (EE, EI, IE); 2 constructible; E-to-I 54906016 unverified; 71 candidates on one pair suspected merge; 3,000-sample rescan stopped after 26 of 104 cells following a progress stall | A measured microcircuit: full_export_scanned remains false; absence of a contact is never established | [progress](h01-measured-connectivity-progress.md), [network](../h01-network.md) |
| Synapses | One model (single-exponential conductance). Numbers were assumed (20 nS, −80 mV, 5 ms, 0.5 ms). Literature pins from human PV basket to pyramidal pairs now recorded: 3.1 nS (1.4 to 3.9), decay 4.2 ms, delay 2.3 ms, reversal about −75 mV | Measured H01 synaptic strength: none exists | [literature](h01-ie-synapse-literature.json) |
| Pair behaviour | Delivery verified under the closing-restored diagnostic I wrapper; with the human-sized receptor one I spike moves the receptor site +0.86 mV and soma +0.008 mV in the tested drifting state | Functional inhibition remains unverified: neither control has E spikes, and the voltage response is depolarising; these runs do not establish impossibility at other states | [pair result](h01-ie-pair-result.md) |
| Network construction | 4 cells, 2 synapses, 75,605 compartments built in 163 s; a compiled run of one 0.005 ms step took 157 s (other session, commits 9184269, dff10c7) | A simulated population: no run longer than one step exists; 98 cells have no accepted edge and are not simulated | [validation](h01-verified-network-validation.md) |
| Holdouts | New seals: PV Noise 1 sweep 48 (needs a waveform-replay path in the PV driver), L2 sweep 55 | Closed spike counts: the Allen counts are on disk and were seen | [I datum](h01-pv-input-datum.json), [L2 reservation](h01-l2-reserved-input.json) |

## What remains, in order of what a user would feel

1. E cell: one dose step (SK 0.35) and the sweep 55 prediction, two evaluations beyond the cap (returned).
2. Functional inhibition at the pair with the literature conductance, and the placement
   question the 0.003 mV result raises (the receptor site's electrotonic distance).
3. I cell post-trough drive and accommodation (reserve held; the existing finalist already has axon-first crossing at all three inputs: [audit](h01-i-initiation-audit.md)).
4. Per-type donors: 74 cells have no type-matched donor; L4 pyramids (18) and
   L3 interneurons (7) are the largest gaps.
5. Population simulation beyond one step, and the matched E-only / I-only controls
   (other session's builder).
