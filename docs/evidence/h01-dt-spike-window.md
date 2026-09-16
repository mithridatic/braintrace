# H01 timestep across a spiking window (2026-09-16)

Spec: [2026-09-16-h01-fused-population.md](../specs/2026-09-16-h01-fused-population.md),
lever 0. The 17-cell Example 21 ARC model (`build_network` -> `init_h01_network_states` ->
`H01ArcModel`, the manifest's numerical path) is driven for 40 ms with each cell's donor
primary current ([kept-currents.json](h01-keep-drop/kept-currents.json), the keep-test
probe) held on the soma clamp route from 2 ms; soma voltage is recorded at every cable
substep and spikes are upward crossings of 0 mV (the output-site event count agreed with
the soma crossings on every cell and arm). Per-arm receipts: [h01-dt-spike-window/](h01-dt-spike-window/).
Executor: Vast 50616476, `/workspace/braintrace-fused`, `taskset -c 160-223`, two keep-test
chains sharing the GPU (the s/event column is contended: 0.528 s at dt 0.005 against 0.180 s
uncontended in the profile campaign).

Reading: every coarser timestep reproduces every spike (30 of 30, the same 1-3 per cell as
the keep test) and moves each spike by at most 0.064 / 0.029 / 0.0087 ms at dt 0.005 /
0.0025 / 0.00125, i.e. first order in dt. The pointwise 1 mV band is a spike-time band of
about 3 us on a spike upstroke, so the ladder fails it at every rung and the edge lies
below 0.00125 ms: the pinned 0.000625 stays the session default and `numerical_settings()`
is unchanged. Off the spikes the arms agree to well under 1 mV (the 2 ms subthreshold
window of the profile campaign gave 0.0013 mV at dt 0.005). If J accepts spike counts plus a
spike-time band as the contract (the keep tests' own dt-half repeats did), dt 0.00125 buys
2x and dt 0.005 buys 8x on every substep-bound cost; that is a decision, not a measurement.

Reference dt 0.000625 ms; 17 cells, 107537 compartments; window 40 ms. Script: `h01_dt_spike_window.py`; JSON: `h01-dt-spike-window.json`.

| dt (ms) | s/event | max abs dV (mV) | cells spiking (ref) | spikes ref / other | counts equal | max spike shift (ms) | within 1 mV |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 0.005 | 0.528 | 17.8005 | 17 | 30 / 30 | True | 0.0637 | False |
| 0.0025 | 1.391 | 7.8697 | 17 | 30 / 30 | True | 0.0288 | False |
| 0.00125 | 2.777 | 2.6484 | 17 | 30 / 30 | True | 0.0087 | False |

Verdict: {"passing_dt_ms": [], "failing_dt_ms": [0.005, 0.0025, 0.00125], "coarsest_passing_dt_ms": null, "contract": "max |dV| <= 1 mV on every cell and identical spike counts per cell"}

## Per cell

### dt 0.005 ms

| cell | max abs dV (mV) | spikes ref | spikes other | shifts (ms) |
| --- | ---: | ---: | ---: | --- |
| 3761379470 | 0.5570 | 1 | 1 | -0.0012 |
| 5173982155 | 3.7865 | 2 | 2 | +0.0031, +0.0044 |
| 5013648003 | 3.7128 | 2 | 2 | -0.0013, +0.0069 |
| 2001418787 | 0.7679 | 1 | 1 | -0.0012 |
| 3571083397 | 5.9573 | 2 | 2 | +0.0000, +0.0113 |
| 4010150634 | 0.4940 | 1 | 1 | -0.0006 |
| 4197933517 | 5.1062 | 2 | 2 | +0.0019, +0.0075 |
| 6833911543 | 4.7136 | 2 | 2 | -0.0006, +0.0087 |
| 5439194879 | 0.9152 | 1 | 1 | +0.0019 |
| 4437316933 | 17.8005 | 3 | 3 | +0.0038, +0.0162, +0.0637 |
| 3111823553 | 4.2790 | 2 | 2 | +0.0031, +0.0075 |
| 1669770671 | 9.4626 | 3 | 3 | +0.0012, +0.0125, +0.0250 |
| 751294744 | 3.5924 | 2 | 2 | +0.0012, +0.0119 |
| 2252715458 | 6.8361 | 2 | 2 | +0.0019, +0.0156 |
| 1684504313 | 1.8543 | 1 | 1 | +0.0050 |
| 2848552900 | 6.2115 | 2 | 2 | +0.0012, +0.0150 |
| 2103991145 | 1.8553 | 1 | 1 | +0.0006 |

### dt 0.0025 ms

| cell | max abs dV (mV) | spikes ref | spikes other | shifts (ms) |
| --- | ---: | ---: | ---: | --- |
| 3761379470 | 0.2349 | 1 | 1 | -0.0012 |
| 5173982155 | 1.6187 | 2 | 2 | +0.0006, +0.0019 |
| 5013648003 | 1.5965 | 2 | 2 | +0.0012, +0.0044 |
| 2001418787 | 0.3243 | 1 | 1 | -0.0012 |
| 3571083397 | 2.5668 | 2 | 2 | +0.0000, +0.0063 |
| 4010150634 | 0.2075 | 1 | 1 | -0.0006 |
| 4197933517 | 2.1931 | 2 | 2 | -0.0006, +0.0050 |
| 6833911543 | 2.0274 | 2 | 2 | -0.0006, +0.0038 |
| 5439194879 | 0.3878 | 1 | 1 | -0.0006 |
| 4437316933 | 7.8697 | 3 | 3 | +0.0012, +0.0062, +0.0288 |
| 3111823553 | 1.8373 | 2 | 2 | +0.0006, +0.0050 |
| 1669770671 | 4.0953 | 3 | 3 | +0.0012, +0.0050, +0.0100 |
| 751294744 | 1.5515 | 2 | 2 | +0.0012, +0.0044 |
| 2252715458 | 2.9509 | 2 | 2 | +0.0019, +0.0056 |
| 1684504313 | 0.7937 | 1 | 1 | +0.0025 |
| 2848552900 | 2.6771 | 2 | 2 | +0.0012, +0.0050 |
| 2103991145 | 0.7933 | 1 | 1 | +0.0006 |

### dt 0.00125 ms

| cell | max abs dV (mV) | spikes ref | spikes other | shifts (ms) |
| --- | ---: | ---: | ---: | --- |
| 3761379470 | 0.0777 | 1 | 1 | +0.0000 |
| 5173982155 | 0.5386 | 2 | 2 | +0.0006, +0.0006 |
| 5013648003 | 0.5329 | 2 | 2 | +0.0000, +0.0006 |
| 2001418787 | 0.1073 | 1 | 1 | +0.0000 |
| 3571083397 | 0.8576 | 2 | 2 | +0.0000, +0.0025 |
| 4010150634 | 0.0685 | 1 | 1 | -0.0006 |
| 4197933517 | 0.7316 | 2 | 2 | -0.0006, +0.0012 |
| 6833911543 | 0.6768 | 2 | 2 | -0.0006, +0.0012 |
| 5439194879 | 0.1284 | 1 | 1 | +0.0006 |
| 4437316933 | 2.6484 | 3 | 3 | +0.0000, +0.0025, +0.0087 |
| 3111823553 | 0.6128 | 2 | 2 | +0.0006, +0.0012 |
| 1669770671 | 1.3701 | 3 | 3 | +0.0000, +0.0012, +0.0038 |
| 751294744 | 0.5190 | 2 | 2 | +0.0000, +0.0019 |
| 2252715458 | 0.9866 | 2 | 2 | +0.0006, +0.0019 |
| 1684504313 | 0.2644 | 1 | 1 | +0.0000 |
| 2848552900 | 0.8945 | 2 | 2 | +0.0012, +0.0025 |
| 2103991145 | 0.2642 | 1 | 1 | +0.0006 |

