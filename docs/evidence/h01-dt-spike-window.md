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

## Every state variable, in lockstep (J's amendment; 2026-09-17)

Script [h01_dt_state_check.py](h01_dt_state_check.py) (`run --mode ladder`); receipts in
[h01-dt-spike-window/state-check/](h01-dt-spike-window/state-check/) (`ladder-report.json`,
`plots/`); the full traces stay on the box under `/workspace/braintrace-fused/var/dtstate/ladder/`
(`frames.npz` sha256 `f107cf1b1ccf6cec...`, per-compartment differences every 0.5 ms for every
variable and pair, float32, 2.2 GB; `sites.npz` per-site traces every 0.005 ms; `end_values.npz`;
the exact digests are in `ladder-report.json` under `files`). Four copies of the fused 17-cell
network (kept manifest plus the same 3 synthetic contacts as the parity check, so synaptic
conductances and currents are exercised) run side by side through the same 40 ms window,
the reference at dt 0.000625 and one copy at 0.005 / 0.0025 / 0.00125; every 0.005 ms all
four are at the same simulated time and every floating-point state of the forest is compared
element by element: membrane voltage, every gating variable of every mechanism (NaTs m/h,
Nap h, K_P m/h, K_T m/h, Kv3.1 m, Im m, Ih m, Ca_HVA m/h, Ca_LVA m/h, SK z), intracellular
calcium, the three synaptic conductances, plus the derived synaptic currents and the axial
(cable) term of dV/dt. Spikes: 33 on every timestep (same times as the parity arm).

**The traces are the deliverable.** Per cell, per pair, one figure with every variable at
the soma, the axon initial segment and the most distal dendritic compartment, reference
and coarse overlaid in the variable's own units and the raw difference (coarse minus
reference) underneath on its own axis:
[ladder-4437316933-0.005-vs-0.000625.png](h01-dt-spike-window/state-check/plots/ladder-4437316933-0.005-vs-0.000625.png)
(L2, three spikes), [ladder-4437316933-0.00125-vs-0.000625.png](h01-dt-spike-window/state-check/plots/ladder-4437316933-0.00125-vs-0.000625.png),
[ladder-4437316933-0.005-vs-0.0025.png](h01-dt-spike-window/state-check/plots/ladder-4437316933-0.005-vs-0.0025.png)
(the standing dt-half rule), [ladder-3761379470-0.005-vs-0.000625.png](h01-dt-spike-window/state-check/plots/ladder-3761379470-0.005-vs-0.000625.png)
(one spike), [ladder-2001418787-0.005-vs-0.000625.png](h01-dt-spike-window/state-check/plots/ladder-2001418787-0.005-vs-0.000625.png)
(three spikes, receives a contact), [ladder-contacts.png](h01-dt-spike-window/state-check/plots/ladder-contacts.png)
(synaptic conductances and currents at all four timesteps); every cell is in `sites.npz`
on the box (`plot --cells ...` draws any of them).

What the traces show: at every site and for every variable the overlaid curves coincide at
the resolution of the plot; the difference trace is flat between spikes and carries one
bipolar pulse per spike whose width is the spike itself and whose height is the spike-time
shift times the slope (13.8 mV on V at dt 0.005 for a 0.064 ms shift on a 200 mV/ms upstroke;
0.17 on NaTs m, 0.10 on Ca_HVA m). After each spike the difference decays over the AHP
(several ms) rather than within +-1 ms, so the "between spikes" column of the index below
(the +-1 ms mask) still contains the tail of the shifted repolarisation: the 2.98 mV on V at
dt 0.005 is at 36.5 ms on cell 4437316933, 1.2 ms after its third spike (35.295 ms), and the
1.71 mV of the dt-half pair is 1.4 ms after a spike of 2001418787. The end-of-window column
(40 ms) is within 2 ms of the last spikes of 1669770671 and 4010150634 (38.04 ms) for the same
reason. Slow variables that do not reset on a spike show the genuine accumulated drift:
calcium 7.7e-6 mM (0.75 % of its range) at dt 0.005, 2.5e-6 at 0.0025, 5.3e-7 at 0.00125; SK z
1.0e-2 / 3.3e-3 / 7.1e-4; Ca_HVA h 2.4e-4 / 7.8e-5 / 1.6e-5; Nap h 1.3e-4 / 4.1e-5 / 8.6e-6:
first order in dt, and at every rung of the same order as the dt-half pair (0.005 vs 0.0025:
calcium 5.2e-6, SK z 6.6e-3), so no variable's drift disappears at a rung above the pinned
one; it halves per halving. Synaptic conductances agree to 2e-7 uS (the pre spike's shift
moves the arrival by at most one coarse step), synaptic currents to 1.6e-3 nA between spikes.

Index (max |coarse - reference| in the variable's own units; "between" = outside +-1 ms of
any spike of that cell, with the time of the maximum; "near" = inside; "end" = at 40 ms):

| variable | unit | dt 0.005: between (at ms) / near / end | dt 0.0025 | dt 0.00125 | dt 0.005 vs 0.0025 |
|---|---|---|---|---|---|
| V | mV | 2.98e+00 (36.5) / 1.38e+01 / 7.98e-01 | 1.27e+00 (36.5) / 4.54e+00 / 3.40e-01 | 4.20e-01 (36.5) / 1.36e+00 / 1.13e-01 | 1.71e+00 (21.5) / 9.26e+00 / 4.58e-01 |
| calcium.Ci | mM | 7.68e-06 (40.0) / 4.42e-06 / 7.68e-06 | 2.50e-06 (40.0) / 1.44e-06 / 2.50e-06 | 5.31e-07 (40.0) / 3.69e-07 / 5.31e-07 | 5.18e-06 (40.0) / 2.98e-06 / 5.18e-06 |
| calcium.pv_Ca_HVA.h | 1 | 2.45e-04 (34.5) / 2.25e-04 / 2.42e-04 | 7.82e-05 (34.5) / 9.73e-05 / 7.72e-05 | 1.62e-05 (34.5) / 3.25e-05 / 1.60e-05 | 1.67e-04 (34.5) / 1.44e-04 / 1.65e-04 |
| calcium.pv_Ca_HVA.m | 1 | 1.95e-02 (36.5) / 1.05e-01 / 9.42e-03 | 8.32e-03 (36.5) / 4.37e-02 / 3.06e-03 | 2.77e-03 (36.5) / 1.44e-02 / 9.89e-04 | 1.15e-02 (28.5) / 6.15e-02 / 6.36e-03 |
| calcium.pv_Ca_LVA.h | 1 | 8.68e-04 (27.0) / 9.56e-04 / 4.82e-04 | 2.74e-04 (27.0) / 3.10e-04 / 1.53e-04 | 5.71e-05 (27.0) / 6.47e-05 / 3.17e-05 | 5.94e-04 (27.0) / 6.47e-04 / 3.28e-04 |
| calcium.pv_Ca_LVA.m | 1 | 1.74e-02 (27.5) / 2.50e-02 / 8.81e-03 | 7.04e-03 (36.5) / 1.08e-02 / 2.84e-03 | 2.36e-03 (36.5) / 3.59e-03 / 6.71e-04 | 1.20e-02 (27.5) / 1.43e-02 / 5.98e-03 |
| syn_synthetic-contact-0.g | uS | 2.18e-07 (32.0) / 2.51e-05 / 3.99e-09 | 2.18e-07 (32.0) / 2.51e-05 / 3.99e-09 | 2.28e-18 (32.5) / 3.47e-17 / 1.19e-19 | 2.17e-19 (33.0) / 2.26e-17 / 2.03e-20 |
| syn_synthetic-contact-1.g | uS | 5.93e-07 (32.0) / 4.16e-05 / 1.09e-08 | 5.89e-07 (32.0) / 4.13e-05 / 1.08e-08 | 1.98e-07 (32.0) / 1.39e-05 / 3.62e-09 | 3.68e-09 (32.0) / 4.92e-05 / 6.74e-11 |
| syn_synthetic-contact-2.g | uS | 2.17e-05 (32.0) / 7.17e-04 / 3.97e-07 | 6.59e-06 (32.0) / 2.18e-04 / 1.21e-07 | 1.64e-06 (32.0) / 5.45e-05 / 3.01e-08 | 1.51e-05 (32.0) / 5.02e-04 / 2.76e-07 |
| pv_Ih.m | 1 | 2.08e-04 (28.5) / 4.53e-04 / 1.58e-04 | 6.55e-05 (28.5) / 1.77e-04 / 5.02e-05 | 1.74e-05 (13.0) / 5.87e-05 / 1.05e-05 | 1.43e-04 (28.5) / 3.19e-04 / 1.08e-04 |
| potassium.pv_Im.m | 1 | 1.49e-02 (13.0) / 9.10e-02 / 4.98e-03 | 6.31e-03 (13.0) / 3.70e-02 / 2.15e-03 | 2.09e-03 (13.0) / 1.18e-02 / 7.21e-04 | 8.60e-03 (13.0) / 5.40e-02 / 2.83e-03 |
| potassium.pv_K_P.h | 1 | 3.25e-04 (35.5) / 3.96e-04 / 3.23e-04 | 1.04e-04 (35.5) / 1.71e-04 / 1.03e-04 | 2.34e-05 (36.5) / 5.73e-05 / 2.18e-05 | 2.21e-04 (35.5) / 2.24e-04 / 2.19e-04 |
| potassium.pv_K_P.m | 1 | 5.10e-03 (25.5) / 1.73e-02 / 2.04e-03 | 1.70e-03 (25.5) / 7.46e-03 / 6.59e-04 | 3.68e-04 (25.5) / 2.48e-03 / 1.38e-04 | 3.40e-03 (25.5) / 9.84e-03 / 1.38e-03 |
| potassium.pv_K_T.h | 1 | 5.57e-03 (30.0) / 6.33e-03 / 4.03e-03 | 1.76e-03 (29.5) / 2.08e-03 / 1.29e-03 | 5.21e-04 (36.5) / 6.95e-04 / 2.89e-04 | 3.81e-03 (30.0) / 4.25e-03 / 2.74e-03 |
| potassium.pv_K_T.m | 1 | 1.64e-02 (36.5) / 6.27e-02 / 6.05e-03 | 7.04e-03 (36.5) / 2.60e-02 / 2.59e-03 | 2.35e-03 (36.5) / 8.57e-03 / 8.63e-04 | 9.40e-03 (36.5) / 3.71e-02 / 3.46e-03 |
| potassium.pv_Kv3_1.m | 1 | 5.55e-03 (25.5) / 1.25e-02 / 8.31e-04 | 1.76e-03 (25.5) / 5.41e-03 / 3.56e-04 | 3.71e-04 (25.5) / 1.80e-03 / 1.19e-04 | 3.78e-03 (25.5) / 7.10e-03 / 4.75e-04 |
| potassium.pv_SK.z | 1 | 9.95e-03 (40.0) / 4.21e-03 / 9.95e-03 | 3.33e-03 (40.0) / 1.31e-03 / 3.33e-03 | 7.12e-04 (40.0) / 3.09e-04 / 7.12e-04 | 6.62e-03 (40.0) / 2.90e-03 / 6.62e-03 |
| sodium.pv_NaTs.h | 1 | 8.86e-03 (23.5) / 3.29e-02 / 2.83e-03 | 2.76e-03 (29.0) / 9.90e-03 / 1.21e-03 | 9.08e-04 (36.5) / 2.50e-03 / 4.04e-04 | 6.21e-03 (23.5) / 2.30e-02 / 1.73e-03 |
| sodium.pv_NaTs.m | 1 | 5.04e-02 (36.5) / 1.66e-01 / 2.57e-02 | 2.15e-02 (36.5) / 6.52e-02 / 1.10e-02 | 7.13e-03 (36.5) / 2.14e-02 / 3.66e-03 | 3.35e-02 (26.5) / 1.18e-01 / 1.47e-02 |
| sodium.pv_Nap.h | 1 | 1.29e-04 (36.5) / 1.10e-04 / 1.29e-04 | 4.11e-05 (36.5) / 4.78e-05 / 4.11e-05 | 8.59e-06 (36.5) / 1.60e-05 / 8.57e-06 | 8.76e-05 (40.0) / 7.35e-05 / 8.76e-05 |
| syn_current | nA | 1.59e-03 (32.0) / 2.33e-01 / 3.13e-05 | 4.84e-04 (32.0) / 7.97e-02 / 9.49e-06 | 1.21e-04 (32.0) / 2.15e-02 / 2.36e-06 | 1.10e-03 (32.0) / 1.53e-01 / 2.18e-05 |

The axial rate is in `ladder-report.json` too (`axial_rate`, mV/ms per CV): its range is set
by the thinnest compartments (coefficients of 1e8-1e9 per ms), so its absolute differences
(4.4e9 mV/ms between spikes at dt 0.005) read only as a fraction of its range (1.1e-2, first
order in dt like V); the per-site traces are the physical reading.

Under the coordinator's proposed contract (identical spike counts and spike-time shift <=
0.1 ms against dt-half) dt 0.005 passes on the spikes (33 / 33, shift <= 0.064 ms against
0.000625 and <= 0.035 ms against 0.0025) and carries the first-order state drift above; J's
call.

## Decision against the dt-half rule (2026-09-17)

The rule put to this ladder: dt 0.005 / 20 substeps becomes the session default only if no
state variable drifts between spikes beyond its own dt-half difference (dt 0.005 against
0.0025). Read from `ladder-report.json` (`index`), between-spike and end-of-window maxima in
the variable's own units, and their ratio:

| variable | unit | between: dt 0.005 vs 0.000625 | dt 0.005 vs 0.0025 (dt-half) | ratio | end: dt 0.005 vs 0.000625 / dt-half | ratio |
|---|---|---:|---:|---:|---:|---:|
| V | mV | 2.98e+00 | 1.71e+00 | 1.74 | 7.98e-01 / 4.58e-01 | 1.74 |
| calcium.Ci | mM | 7.68e-06 | 5.18e-06 | 1.48 | 7.68e-06 / 5.18e-06 | 1.48 |
| calcium.pv_Ca_HVA.h | 1 | 2.45e-04 | 1.67e-04 | 1.47 | 2.42e-04 / 1.65e-04 | 1.47 |
| calcium.pv_Ca_HVA.m | 1 | 1.95e-02 | 1.15e-02 | 1.69 | 9.42e-03 / 6.36e-03 | 1.48 |
| calcium.pv_Ca_LVA.h | 1 | 8.68e-04 | 5.94e-04 | 1.46 | 4.82e-04 / 3.28e-04 | 1.47 |
| calcium.pv_Ca_LVA.m | 1 | 1.74e-02 | 1.20e-02 | 1.45 | 8.81e-03 / 5.98e-03 | 1.47 |
| syn_synthetic-contact-0.g | uS | 2.18e-07 | 2.17e-19 | (a) | 3.99e-09 / 2.03e-20 | (a) |
| syn_synthetic-contact-1.g | uS | 5.93e-07 | 3.68e-09 | (a) | 1.09e-08 / 6.74e-11 | (a) |
| syn_synthetic-contact-2.g | uS | 2.17e-05 | 1.51e-05 | 1.44 | 3.97e-07 / 2.76e-07 | 1.44 |
| pv_Ih.m | 1 | 2.08e-04 | 1.43e-04 | 1.46 | 1.58e-04 / 1.08e-04 | 1.46 |
| potassium.pv_Im.m | 1 | 1.49e-02 | 8.60e-03 | 1.73 | 4.98e-03 / 2.83e-03 | 1.76 |
| potassium.pv_K_P.h | 1 | 3.25e-04 | 2.21e-04 | 1.47 | 3.23e-04 / 2.19e-04 | 1.47 |
| potassium.pv_K_P.m | 1 | 5.10e-03 | 3.40e-03 | 1.50 | 2.04e-03 / 1.38e-03 | 1.48 |
| potassium.pv_K_T.h | 1 | 5.57e-03 | 3.81e-03 | 1.46 | 4.03e-03 / 2.74e-03 | 1.47 |
| potassium.pv_K_T.m | 1 | 1.64e-02 | 9.40e-03 | 1.75 | 6.05e-03 / 3.46e-03 | 1.75 |
| potassium.pv_Kv3_1.m | 1 | 5.55e-03 | 3.78e-03 | 1.47 | 8.31e-04 / 4.75e-04 | 1.75 |
| potassium.pv_SK.z | 1 | 9.95e-03 | 6.62e-03 | 1.50 | 9.95e-03 / 6.62e-03 | 1.50 |
| sodium.pv_NaTs.h | 1 | 8.86e-03 | 6.21e-03 | 1.43 | 2.83e-03 / 1.73e-03 | 1.64 |
| sodium.pv_NaTs.m | 1 | 5.04e-02 | 3.35e-02 | 1.50 | 2.57e-02 / 1.47e-02 | 1.75 |
| sodium.pv_Nap.h | 1 | 1.29e-04 | 8.76e-05 | 1.47 | 1.29e-04 / 8.76e-05 | 1.47 |
| axial_rate | mV/ms | 4.37e+09 | 3.09e+09 | 1.42 | 2.13e+08 / 1.19e+08 | 1.78 |
| syn_current | nA | 1.59e-03 | 1.10e-03 | 1.44 | 3.13e-05 / 2.18e-05 | 1.44 |

(a) The dt 0.005 and 0.0025 copies deliver contact 0's and 1's pre spike on the same coarse
step, so their conductances agree to 1e-19 uS; the ratio there reads the coarse arrival
grid, not convergence, and contact 2 (whose pre spike straddles a step) shows the 1.44.

Every variable's difference from the pinned step exceeds its dt-half difference, by 1.42-1.76x,
the ratio of first-order convergence (0.875 / 0.5 = 1.75 in the limit; below it where the
coarse-step error is not yet in the asymptotic regime). The slow states that do not reset on
a spike (calcium, SK z, Ca_HVA h, Nap h) carry their maximum at 40 ms, a drift that halves per
halving of dt and vanishes at no rung above the pinned one. So: `dt_ms` stays 0.000625 /
160 substeps, `fused_population` stays `False` by default, the kept manifest stands.
Recorded in the spec (section 6) and in `docs/h01-causal-model.md` ("Population execution
cost and the cable timestep"). The caveat stated above holds: the +-1 ms mask leaves the AHP
tail of a shifted spike in the between-spike column, which bounds V's column from above and
does not move the ratio.
