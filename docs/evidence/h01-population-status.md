# H01 population: what a user gets today (2026-09-07)

One table, no pass claimed. Sources are the pages linked in each row.

| Layer | What exists | What it is not | Evidence |
| --- | --- | --- | --- |
| Anatomy | 104 proofread cells, 3,327 components, all with a soma-bearing component; largest component holds a median 77 % of a cell's nodes (minimum 25 %); 73 cells have more than one soma-bearing component | Joined cells: the builder loads one component per cell and never joins fragments | [components](h01-population-components.md) |
| Cell types | Layer and class for every cell from the released tags (28 L2 pyramids, 18 L4 pyramids, 25 interneurons across L1 to L5, 27 tag combinations) | Interneuron subtypes: none in the tags | [types](h01-population-types.md) |
| Physiology per cell | Two frozen donor profiles applied by Dale sign: an L2 pyramidal fit (Allen 541563728) and a PV basket fit (HL5BN1). 30 of 104 cells match their donor in layer and class; 74 are borrowed across layer, class or morphology | Human-qualified cells: neither donor passes the 1 mV contract; under the usable tier the I cell fails rate at two of three inputs, adaptation everywhere and width (20 % narrow); the E cell fails rate at 310 pA and adaptation | [usable I](h01-usable-tier-i.md), [usable E](h01-usable-tier-e.md), [prediction I](h01-prediction-i.md), [prediction E](h01-prediction-e.md) |
| E cell repair in progress | Axonal initiation site established (Stage A): threshold and two-stage upstroke now match the human; peak, main rise and count under search (Stage A2, B) | A finished E cell | [Stage A decision](h01-e-usable/stage-a-decision.json) |
| Connectivity | 123 candidate contacts on 31 pairs from a 1,500-sample scan; 3 endpoint-verified (EE, EI, IE); 2 constructible; E-to-I 54906016 unverified; 71 candidates on one pair suspected merge; a 3,000-sample rescan running | A measured microcircuit: absence of a contact is never established | [progress](h01-measured-connectivity-progress.md), [network](../h01-network.md) |
| Synapses | One model (single-exponential conductance). Numbers were assumed (20 nS, −80 mV, 5 ms, 0.5 ms). Literature pins from human PV basket to pyramidal pairs now recorded: 3.1 nS (1.4 to 3.9), decay 4.2 ms, delay 2.3 ms, reversal about −75 mV | Measured H01 synaptic strength: none exists | [literature](h01-ie-synapse-literature.json) |
| Pair behaviour | Event delivery from the I axon endpoint to the E receptor site verified; the delivered inhibition moved the E soma by 0.003 mV with the assumed 20 nS | Demonstrated functional inhibition (pair test in progress, W4) | [paired audit](h01-measured-delivery20-paired-audit.json) |
| Network construction | 4 cells, 2 synapses, 75,605 compartments built in 163 s; a compiled run of one 0.005 ms step took 157 s (other session, commits 9184269, dff10c7) | A simulated population: no run longer than one step exists; 98 cells have no accepted edge and are not simulated | [validation](h01-verified-network-validation.md) |
| Holdouts | New seals: PV Noise 1 sweep 48 (needs a waveform-replay path in the PV driver), L2 sweep 55 | Closed spike counts: the Allen counts are on disk and were seen | [I datum](h01-pv-input-datum.json), [L2 reservation](h01-l2-reserved-input.json) |

## What remains, in order of what a user would feel

1. E cell count and gain (Stage B, up to 5 evaluations), then the sweep 55 prediction.
2. Functional inhibition at the pair with the literature conductance, and the placement
   question the 0.003 mV result raises (the receptor site's electrotonic distance).
3. I cell post-trough drive and accommodation (reserve held; the E result points at
   an axonal initiation site as the mechanism to try).
4. Per-type donors: 74 cells have no type-matched donor; L4 pyramids (18) and
   L3 interneurons (7) are the largest gaps.
5. Population simulation beyond one step, and the matched E-only / I-only controls
   (other session's builder).
