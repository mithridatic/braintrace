# C3 partners of the 17 kept cells (workstream A, kept-partner graph)

Graph source `kept-partner`; rows 378; partners with >= 1 AXON-pre synapse 91 (any pre class: 167); >= 2 AXON-pre synapses to kept: 2; either direction: 14; kept-kept rows 0 (AXON-pre pairs 0, same-cell rows 121); eligible E 61, I 28. Prediction A (registered wording, >= 2 to kept): **refuted**; to/from reading: **refuted** (every kept cell has an AXON-pre partner: False).

Why this graph replaces the C3-id-only ranking. The first ranking (`h01-c3-candidates-c3id-only.json`, from `h01-c3-neuron-graph.json.gz`) kept a synapse only when both `neuron_id` fields were tabulated C3 neurons. A C3 soma-bearing segment carries almost none of its own axon (axons are separate C3 fragments), so a kept cell's outgoing synapses never met that rule and its incoming synapses were seen only where the kept cell's dendrite happened to sit in the tabulated segment: 85 partners, 4 with >= 2 synapses. The kept 17 are proofread cells whose reconstructions are known through base-segment membership (`.cache/h01/proofread-base-membership.json`), and every export row carries `base_neuron_id` for both sites, so `h01_c3_kept_partner_graph.py` rescanned the 166 shards keeping a row when a kept cell owns either base segment (or the row's `neuron_id` is a kept cell's C3 id) and the other side is a tabulated neuron or another kept cell. Only rows with `pre_class == AXON` count toward partners and ranks; the other presynaptic classes are recorded as `dendrite_pre_rows_ignored`. The tabulated partners themselves are still identified by their C3 segment, so a tabulated neuron's own axon output onto a kept cell is seen only where its axon sits in its soma-bearing segment; the progress file's `dropped_untabulated_partner` counts the kept-side synapses whose other end is an untabulated fragment.

| kept C3 id | released id | partners (AXON-pre) | partners (any) | in | in AXON | out | out AXON |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1931553044 | 1669770671 | 27 | 34 | 5 | 1 | 39 | 33 |
| 1684504313 | 1684504313 | 11 | 20 | 9 | 3 | 14 | 8 |
| 2757428565 | 2001418787 | 0 | 3 | 2 | 0 | 5 | 0 |
| 2103991145 | 2103991145 | 6 | 11 | 10 | 6 | 2 | 0 |
| 2252715458 | 2252715458 | 5 | 10 | 3 | 1 | 18 | 12 |
| 2863125070 | 2848552900 | 6 | 13 | 5 | 4 | 9 | 2 |
| 3111823918 | 3111823553 | 11 | 13 | 1 | 0 | 25 | 21 |
| 3571083397 | 3571083397 | 0 | 5 | 4 | 0 | 3 | 0 |
| 3761379470 | 3761379470 | 1 | 4 | 4 | 3 | 4 | 0 |
| 4010150634 | 4010150634 | 7 | 10 | 3 | 0 | 9 | 8 |
| 4445420826 | 4197933517 | 3 | 11 | 5 | 1 | 9 | 2 |
| 4466811772 | 4437316933 | 3 | 8 | 21 | 0 | 5 | 4 |
| 5013648003 | 5013648003 | 1 | 6 | 6 | 1 | 2 | 0 |
| 5421628756 | 5173982155 | 4 | 4 | 1 | 1 | 6 | 6 |
| 5147072947 | 5439194879 | 2 | 3 | 2 | 1 | 1 | 1 |
| 6833911543 | 6833911543 | 1 | 5 | 2 | 1 | 3 | 0 |
| 751294744 | 751294744 | 7 | 14 | 7 | 4 | 13 | 5 |

Kept -> kept AXON-pre rows (pre C3 id, post C3 id, count): none

| # | C3 id | pol | layer | class | modifiers | donor | match | to kept AXON | from kept AXON | ignored | NSI |
| ---: | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | 3790464726 | E | L4 | pyramidal | - | l4-pyramidal-allen-527952884 | matched | 4 | 0 | 0 | 1804 |
| 2 | 2062261179 | E | L2 | pyramidal | lot-of-axon | l2-pyramidal-allen-541563728 | modifier | 1 | 2 | 0 | 3778 |
| 3 | 36052250628 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 2 | 0 | 0 | 3270 |
| 4 | 693197378 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 0 | 2 | 0 | 1096 |
| 5 | 4083404518 | E | L4 | pyramidal | - | l4-pyramidal-allen-527952884 | matched | 0 | 2 | 0 | 519 |
| 6 | 3445514816 | E | L3 | pyramidal | - | l2-pyramidal-allen-541563728 | layer | 0 | 1 | 0 | 10677 |
| 7 | 2468824745 | E | L3 | pyramidal | - | l2-pyramidal-allen-541563728 | layer | 0 | 1 | 0 | 9867 |
| 8 | 1913504984 | E | L3 | pyramidal | - | l2-pyramidal-allen-541563728 | layer | 1 | 0 | 0 | 9365 |
| 9 | 3373239870 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 0 | 1 | 0 | 8666 |
| 10 | 1669011958 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 0 | 1 | 0 | 8090 |
| 11 | 2136317545 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 0 | 1 | 0 | 7718 |
| 12 | 3126060193 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 0 | 1 | 0 | 6680 |
| 13 | 2950643092 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 0 | 1 | 0 | 6079 |
| 14 | 30580832829 | E | L3 | pyramidal | - | l2-pyramidal-allen-541563728 | layer | 1 | 0 | 0 | 5974 |
| 15 | 927076559 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 0 | 1 | 0 | 5864 |
| 16 | 3825023980 | E | L3 | pyramidal | - | l2-pyramidal-allen-541563728 | layer | 0 | 1 | 0 | 5843 |
| 17 | 2136098827 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 0 | 1 | 0 | 5580 |
| 18 | 2091491230 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 0 | 1 | 0 | 5506 |
| 19 | 1886902875 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 0 | 1 | 0 | 5469 |
| 20 | 2251927076 | E | L2 | pyramidal | - | l2-pyramidal-allen-541563728 | matched | 0 | 1 | 0 | 5400 |
| 21 | 3504577656 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 11 | 0 | 2952 |
| 22 | 3518943222 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 1 | 10 | 0 | 2330 |
| 23 | 3617981810 | I | L5 | interneuron | - | l5-pv-basket-hl5bn1 | matched | 0 | 3 | 1 | 2646 |
| 24 | 2702762854 | I | L3 | interneuron | - | l3-sst-interneuron-hl5mn1 | matched | 0 | 3 | 0 | 2437 |
| 25 | 2440908874 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 3 | 0 | 1061 |
| 26 | 4524762575 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 2 | 0 | 697 |
| 27 | 2193655685 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 2 | 0 | 611 |
| 28 | 28949696072 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 2 | 0 | 455 |
| 29 | 4896547576 | I | L5 | interneuron | - | l5-pv-basket-hl5bn1 | matched | 1 | 0 | 0 | 5342 |
| 30 | 3426840498 | I | L5 | interneuron | - | l5-pv-basket-hl5bn1 | matched | 0 | 1 | 0 | 3659 |
| 31 | 30406000355 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 1 | 0 | 2773 |
| 32 | 4475221179 | I | L5 | interneuron | - | l5-pv-basket-hl5bn1 | matched | 0 | 1 | 0 | 2486 |
| 33 | 1539076840 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 1 | 0 | 2376 |
| 34 | 29938695074 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 1 | 0 | 2349 |
| 35 | 37218168369 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 1 | 0 | 2223 |
| 36 | 37421824682 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 1 | 0 | 1833 |
| 37 | 2207657768 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 1 | 0 | 1537 |
| 38 | 925441648 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 1 | 0 | 1469 |
| 39 | 28629556145 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 1 | 0 | 1124 |
| 40 | 2776001292 | I | L2 | interneuron | - | l5-pv-basket-hl5bn1 | layer | 0 | 1 | 0 | 1052 |
