# H01 population: what a user gets today (2026-09-07)

One table, no pass claimed. Sources are the pages linked in each row.

| Layer | What exists | What it is not | Evidence |
| --- | --- | --- | --- |
| Anatomy | 104 proofread cells, 3,327 components, all with a soma-bearing component; largest component holds a median 77 % of a cell's nodes (minimum 25 %); 73 cells have more than one soma-bearing component | Joined cells: the builder loads one component per cell and never joins fragments | [components](h01-population-components.md) |
| Cell types | Layer and class for every cell from the released tags (28 L2 pyramids, 18 L4 pyramids, 25 interneurons across L1 to L5, 27 tag combinations) | Interneuron subtypes: none in the tags | [types](h01-population-types.md) |
| Physiology per cell | Two frozen donor profiles applied by Dale sign: an L2 pyramidal fit (Allen 541563728) and a PV basket fit (HL5BN1). 30 of 104 cells match their donor in layer and class; 74 are borrowed across layer, class or morphology | Human-qualified cells: neither donor passes the 1 mV contract; under the usable tier the I cell fails rate at two of three inputs, adaptation everywhere and width (20 % narrow); the E cell fails rate at 310 pA and adaptation | [usable I](h01-usable-tier-i.md), [usable E](h01-usable-tier-e.md), [prediction I](h01-prediction-i.md), [prediction E](h01-prediction-e.md) |
| E cell repair | Frozen B3 gives 8/5, 10/10 and 12/13 spikes at 250/310/350 pA; rate passes at 310/350, fails at 250; adaptation passes; width fails at cycle2 of 310 and cycle3 of 350; AHP unresolvable. Gain split Stage 0 (g0-b3, 1 of 4 evaluations): sweep-43 onset rows within 1 mV, late-return rows miss by +1.4 to +1.7 mV (registered rejection met); sweeps 50/53/56 killed at the 1500 s abort on a shared host (untested); model f-I slope 0.033 vs human 0.083 spikes/pA (250-310, retained B3), 200 pA point unmeasured; Stage G closed | A usable shared profile or a BrainCell/H01 transfer pass: no promotion; a completed Stage 0 evaluation | [continuation](h01-e-continuation-result.md), [rows](h01-e-usable/b3-all-inputs-usable.md), [stage 0](h01-e-gain-stage0.md), [decision](h01-e-gain/stage-g0-decision.json) |
| Connectivity | 123 candidate contacts on 31 pairs from a 1,500-sample scan; 3 endpoint-verified (EE, EI, IE); 2 constructible; E-to-I 54906016 unverified; 71 candidates on one pair suspected merge; 3,000-sample rescan stopped after 26 of 104 cells following a progress stall | A measured microcircuit: full_export_scanned remains false; absence of a contact is never established | [progress](h01-measured-connectivity-progress.md), [network](../h01-network.md) |
| Synapses | One model (single-exponential conductance). Numbers were assumed (20 nS, −80 mV, 5 ms, 0.5 ms). Literature pins from human PV basket to pyramidal pairs now recorded: 3.1 nS (1.4 to 3.9), decay 4.2 ms, delay 2.3 ms, reversal about −75 mV | Measured H01 synaptic strength: none exists | [literature](h01-ie-synapse-literature.json) |
| Pair behaviour | Delivery verified under the closing-restored diagnostic I wrapper; with the human-sized receptor one I spike moves the receptor site +0.86 mV and soma +0.008 mV in the tested drifting state | Functional inhibition remains unverified: neither control has E spikes, and the voltage response is depolarising; these runs do not establish impossibility at other states | [pair result](h01-ie-pair-result.md) |
| Network construction | 4 cells, 2 synapses, 75,605 compartments built in 163 s; a compiled run of one 0.005 ms step took 157 s (other session, commits 9184269, dff10c7) | A simulated population: no run longer than one step exists; 98 cells have no accepted edge and are not simulated | [validation](h01-verified-network-validation.md) |
| Holdouts | PV Noise1 sweep48 remains sealed and needs waveform replay; L2 sweep55 evaluated once after frozen prediction commit306459e, now spent | Independent donor validation: the prediction is on the same donor; Allen count metadata had already been seen | [I datum](h01-pv-input-datum.json), [L2 reservation](h01-l2-reserved-input.json), [P2 decision](h01-e-usable/stage-p2-decision.json) |

## What remains, in order of what a user would feel

1. E cell: resolve the input-response and early-spike waveform failures exposed by the completed [bounded continuation](h01-e-continuation-result.md), then qualify one shared profile and its transfer.
2. Functional inhibition at the pair with the literature conductance, and the placement
   question the 0.003 mV result raises (the receptor site's electrotonic distance).
3. I cell post-trough drive and accommodation (reserve held; the existing finalist already has axon-first crossing at all three inputs: [audit](h01-i-initiation-audit.md)).
4. Per-type donors: 74 cells have no type-matched donor; L4 pyramids (18) and
   L3 interneurons (7) are the largest gaps.
5. Population simulation beyond one step, and the matched E-only / I-only controls
   (other session's builder).
