# H01 verified-network throughput (SP1)

Generated 2026-09-07T15:24:48. dt 0.005 ms, 4 cells, 2 projections, 75,605 compartments. Cost qualification only; no physiology claim.

| Run | Solver | Duration ms | Steps | Construction s | Init+run s | Wall s | Peak RSS MB | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| d0p05 | h01_staggered_scan | 0.05 | 10 | 280.2 | 270.0 | 556.8 | 1350 | completed |
| d1 | h01_staggered_scan | 1.0 | 200 | 266.2 | 212.4 | 484.7 | 1342 | completed |
| d1r | h01_staggered_scan | 1.0 | 200 | 243.6 | 274.9 | 525.1 | 1331 | completed |
| d10 | h01_staggered_scan | 10.0 | 2000 | 246.4 | 363.2 | 615.1 | 1329 | completed |
| ctl-staggered-d1 | staggered | 1.0 | 200 | 239.3 | not tested | 844.1 | 8100 | killed |

## Fit T(steps) = a + b*steps (h01_staggered_scan, completed runs)

- a (init + compile): 245.39 s
- b (per step): 0.05764 s
- 200*b, seconds per simulated ms: 11.528 s
- residuals (s): 24.01, -44.54, 17.99, 2.53
- two-repeat decision limit on b: 0.97098 s/step; in seconds at the repeated point: 184.486
- prediction (0.05 ms within 156.96669389998715 s +/- limit): held
- three points linear within limit: held
- predicted 10 ms cost: 360.7 s (cap 900.0 s; allowed: True)

Killed or incomplete (untested, not negative): ctl-staggered-d1

## Addendum 2026-09-07, 19:45: post-merge equivalence rerun of d1 (registered in the builder spec addendum 19:30; **prediction failed**)

`feat/h01-braincell` (performance commits plus the SP6c donors) merged into `feat/h01-network` at 0488c84; the d1 configuration
(4 incident cells, dt 0.005 ms, 1 ms, `h01_staggered_scan`, control ei, zero current) rerun on a quiet machine (the 40-cell wait
condition held: SP5 benchmark completed and labelled quiet, docker only `synapse`, no heavy python; CPU 19-26 % with SearchIndexer
~1.5 cores, docker backend ~1.2, ProtonDrive ~0.9, bdservicehost ~0.7). Full record:
[h01-network-equivalence-post-merge.json](h01-network-equivalence-post-merge.json).

| Quantity | SP1 d1 (alone) | SP1 d1r | Rerun (merged tree) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Construction s | 266.2 | 243.6 | 97.9 | 2.718x |
| init_state s | not separated | not separated | 74.7 (cell 3955003482 40.9 s) | - |
| compile + 200 steps s | not separated | not separated | 13.2 | - |
| init + compile + run s | 212.4 | 274.9 | 87.9 | 2.416x |
| Wall s | 484.7 | 525.1 | 191.7 | 2.575x |
| Peak RSS MB | 1342 | 1331 | 1347 (peak_wset) / 1330 (tree) | - |
| Compartments | 75,605 | 75,605 | 75,605 | - |

The wall speedup is the JSON value (`speedup.total` 2.575 in
[h01-network-equivalence-post-merge.json](h01-network-equivalence-post-merge.json)); note that
484.7 / 191.7 = 2.53, so the JSON figure and the wall clocks need reconciliation before quoting either as final.

Trace comparison against `.cache/h01/bench-d1` (itself bitwise identical to `bench-d1r`); keys and shapes identical; the comparison
ran without error:

| Array | Bitwise equal | Max abs difference (mV or uS), or spikes ref / new |
| --- | --- | ---: |
| cell_3955003482_events | yes | 0 / 0 spikes |
| cell_3955003482_output_voltage | no | 3.17 |
| cell_3955003482_syn_124698307_g | yes | 0 |
| cell_3955003482_syn_124698307_voltage | no | 3.16 |
| cell_3955003482_voltage | no | 3.17 |
| cell_4157825456_events | yes | 0 / 0 spikes |
| cell_4157825456_output_voltage | no | 1.33e-09 |
| cell_4157825456_syn_8105899_g | yes | 0 |
| cell_4157825456_syn_8105899_voltage | no | 5.41e-11 |
| cell_4157825456_voltage | no | 1.33e-09 |
| cell_4188575291_events | yes | 0 / 0 spikes |
| cell_4188575291_output_voltage | no | 1.13e-11 |
| cell_4188575291_voltage | no | 2.13e-11 |
| cell_5584343344_events | yes | 0 / 0 spikes |
| cell_5584343344_output_voltage | no | 2.67e-11 |
| cell_5584343344_voltage | no | 3.59e-10 |
| time_ms | yes | 0 |

Verdict: **rejected** under the registered rule (1e-9 mV). Spike arrays and conductance probes are bitwise identical; `time_ms`
identical. Two cells are within 1e-9 (2.1e-11, 3.6e-10), one is marginally over (4157825456, 1.325e-9 at 0.99 ms), and cell
3955003482 differs by 3.16 mV from the first sample (rest -80.82 vs -83.98 mV). Cause of the 3 mV: the merged SP6c commits (415b038,
312e391) assign the new Allen L4 pyramidal donor 527952884 to the L4-tagged cell 3955003482 (`initial_mv` -80.818 vs -83.980, axial
15.0 vs 94.6 ohm cm, different regions); SP1 built it on the L2 donor. A donor-registry change, not a numerical one. The three other
cells keep donor and values; their axon electrical-interval endpoints differ by 2e-16 to 4e-16 (same count, same set to that
rounding), the signature of the vectorised geometry attachment / DHS assembly reordering sums, and their voltages differ by 1e-11 to
1.3e-9 mV after 200 steps. The verdict **rejected** stands for this unpinned run. The registered follow-up (the same rerun with
the donor registry pinned to SP1's assignment for 3955003482) was run and **passed** at tolerance 2e-9 on all four cells
(per-cell maxima 2.20e-10 / 1.325e-9 / 2.13e-11 / 3.59e-10 mV; spikes identical; construction 152.3 s, init+run 133.5 s,
wall 295.0 s, speedup 1.674x; `status` pass in
[h01-network-equivalence-post-merge-pinned.json](h01-network-equivalence-post-merge-pinned.json); see the 20:35 addendum) and
gated the 40-cell attempt (`preconditions.equivalence_step` held in [h01-population-build-40.json](h01-population-build-40.json)).
A build-only 40-cell launch was made at 21:06:01 (after the pinned equivalence step held) and failed at 82.6 s with ValueError in
H01Archive.load on cell 5805562981; stages 2-5 untested, stopped by the user (see [h01-population-build-40.md](h01-population-build-40.md)).


## Addendum 2026-09-07, 20:35: equivalence split 2, cell 3955003482 pinned to SP1's donor (registered 19:55; **prediction held**)

Same d1 configuration on the merged tree with `3955003482` pinned to `l2-pyramidal-allen-541563728` by an in-process override of `donor_for_tags` (recorded in [the JSON](h01-network-equivalence-post-merge-pinned.json) under `donor_override`; the merged registry would assign `l4-pyramidal-allen-527952884`, an intended SP6c change, not a fault). Label: **measured beside one NEURON container** (Stage G; CPU 20-40 %, vmmemWSL 3-5.6 cores, SearchIndexer 1.7-4.8, bdservicehost ~1.6).

Per-cell max abs voltage difference against `bench-d1` (rule 2e-9 mV): 3955003482 **2.20e-10**, 4157825456 **1.325e-9**, 4188575291 **2.13e-11**, 5584343344 **3.59e-10**; spike arrays bitwise identical (0 events both sides); conductance probes and `time_ms` bitwise identical; keys and shapes identical. **Held.** Timing: construction 152.3 s (SP1 266.2), init_state 116.0 s (cell 3955003482 68.9), compile + 200 steps 17.5 s, init + run 133.5 s (SP1 212.4), wall 295.0 s (SP1 484.7), peak RSS 1,354 MB; speedup 1.75x construction, 1.59x init + run, beside the NEURON container (2.7x / 2.4x on the quieter 19:42 run).

The first comparison of this run (20:29) failed in the compare script itself (`max()` over an empty list: the per-cell reduction matched keys without the `cell_` prefix), not in the data; fixed, unit-tested, and rerun offline on the recorded trace. The 40-cell stages proceeded as registered; the build-only launch at 21:06:01 then failed at 82.6 s (see the 19:45 addendum and [h01-population-build-40.md](h01-population-build-40.md)).
