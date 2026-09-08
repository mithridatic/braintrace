# SP2 transfer isolation, I side: result (2026-09-07, closed)

Spec: [SP2 transfer isolation](../specs/2026-09-07-h01-transfer-isolation.md) with its four 2026-09-07 amendments
(interpolated peak metric; peak row 0.5 mV + Richardson column; close on simulator identity + NEURON dt
qualification + one confirming BrainCell train). Steps 1 and 1b: [halving pair](h01-i-transfer-step1.md),
`h01-i-transfer/step1-decision.json`, `step1b-decision.json`. Full-train arms under the amended gate:
`h01-i-transfer/sp2-i-decision.json`. This page is rendered from `h01-i-transfer/sp2-close-decision.json`
(runner `--score --close --peak-method interpolated --reference-root <sibling worktree>`; every per-event row,
wall clock and hash is in it).

**Identity gate: CLOSED.** Basis arm `r-braincell-matched-027` (BrainCell, copied x9 mesh, dt 0.005)
scored against `b1-fixed-027` (NEURON fixed step, same dt and mesh) over 270-329.5 ms at 0.27 nA:
counts 3 / 3, max |rise| 1.2e-09 ms, max |interpolated peak| 5.42e-06 mV,
max |width| 1.91e-11 ms, raw voltage 7.47e-07 mV. Closing rule: user decision (a): the basis arm reproduces the quoted identity values;
decision values {'rise_crossing_ms': 1.2e-09, 'peak_interpolated_voltage_mv': 5.4e-06, 'time_above_threshold_ms': 1.9e-11}; measured {'rise_crossing_ms': 1.2048531061736867e-09, 'peak_interpolated_voltage_mv': 5.424022674560547e-06, 'time_above_threshold_ms': 1.9099388737231493e-11}.
Y2's question, does BrainCell reproduce the NEURON model, is answered yes at equal dt and mesh (user decision, spec amendment).

**Confirming full-train prediction (c), identity gate {'rise_crossing_ms': 1e-08, 'peak_sample_voltage_mv': 1e-06, 'time_above_threshold_ms': 1e-08} at every event over 270-1270 ms: held = False** (rows below).

**dt qualification (NEURON fixed step vs CVode finalist, 1 ms crossing tolerance at every event, both inputs): not reached within cap.**
Rule: {'rise_crossing_ms': 1.0} at every paired event and equal nonzero count, at both 0.27 and 0.19 nA. Cap 6 NEURON runs, 900 s abort.

**Full-train decision literal (`decide()`, amended gate): `untested`**; gate valid: True.
A1 passed: {'a1-matched-019': False, 'a1-matched-027': False}. B1 passed: {'b1-fixed-019': False, 'b1-fixed-027': False}.

## Wall clocks (measured; container or interpreter start included)

| Arm | Simulator | dt (ms) | Simulated (ms) | Wall clock (s) | Exit | Killed | Other containers at start |
| --- | --- | --- | --- | --- | --- | --- | --- |
| a1-matched-027.killed | braincell | 0.005 | 1500 | 602.0 | None | yes | h01-e-gain-g0-b3-sweep50,synapse |
| a1-matched-027 | braincell | 0.005 | 1500 | 616.7 | 0 | no | synapse |
| b1-fixed-019 | neuron | 0.005 | 1500 | 152.8 | 0 | no | h01-e-gain-g0-b3-sweep53,synapse |
| b1-fixed-027 | neuron | 0.005 | 1500 | 232.4 | 0 | no | h01-e-gain-g0-b3-sweep53,synapse |
| mislaunch-maxcv-as-a1-027-killed | braincell | 0.005 | 1500 | 601.5 | None | yes | h01-e-gain-g0-b3-sweep50,synapse |
| q-neuron-fixed-019-dt000625 | neuron | 0.000625 | 1500 | 351.1 | 0 | no | synapse |
| q-neuron-fixed-019-dt00125 | neuron | 0.00125 | 1500 | 421.5 | 0 | no | h01-e-gain-g0-b3-sweep56,synapse |
| q-neuron-fixed-019-dt0025 | neuron | 0.0025 | 1500 | 194.7 | 0 | no | h01-e-gain-g0-b3-sweep56,synapse |
| q-neuron-fixed-027-dt000625 | neuron | 0.000625 | 1500 | 666.6 | 0 | no | h01-e-gain-g0-b3-sweep56,synapse |
| q-neuron-fixed-027-dt00125 | neuron | 0.00125 | 1500 | 428.4 | 0 | no | h01-e-gain-g0-b3-sweep56,synapse |
| q-neuron-fixed-027-dt0025 | neuron | 0.0025 | 1500 | 274.5 | 0 | no | h01-e-gain-g0-b3-sweep53,synapse |
| r-braincell-matched-027-halfdt | braincell | 0.0025 | 330 | 246.0 | 0 | no |  |
| r-braincell-matched-027 | braincell | 0.005 | 330 | 115.6 | 0 | no |  |
| r-neuron-fixed-027-halfdt | neuron | 0.0025 | 330 | 121.8 | 0 | no | h01-e-gain-g0-b3-sweep53,synapse |

Other containers or sessions were running on the host during most runs (listed per arm; the last two ran beside `synapse` only), so the wall clocks are upper bounds on an idle host.
No duration on this page is derived; each is the launcher's stopwatch.

## dt series: NEURON fixed step vs CVode finalist, 270-1270 ms

Prediction registered: The late-event rise error scales first order with dt: 7.42 ms at dt 0.005 at 0.27 nA -> about 3.7, 1.9, 0.9 ms at 0.0025, 0.00125, 0.000625 (14.75 ms at 0.19 nA -> about 7.4, 3.7, 1.8 ms); 1 ms is met at dt 0.000625 or finer at 0.27 nA. Adjacent-dt pairs give ratios near 2.

Rejection registered: The error does not halve with dt (adjacent-pair ratio outside 1.5-2.5); then the CVode reference (atol 1e-10) must be questioned before any dt is named. A run whose derived cost (twice the measured adjacent-dt wall clock) exceeds the 900 s abort is not launched and recorded as such; a killed run is untested.

| Rung | Input (nA) | dt (ms) | Wall clock (s) | Events ref / actual | Last paired event rise err (ms) | Max abs rise err (ms) | 1 ms met |
| --- | --- | --- | --- | --- | --- | --- | --- |
| b1-fixed-027 | 0.27 | 0.005 | 232.4 | 37 / 36 | +7.420 | 7.420 | no |
| b1-fixed-019 | 0.19 | 0.005 | 152.8 | 14 / 14 | +14.755 | 14.755 | no |
| q-neuron-fixed-027-dt0025 | 0.27 | 0.0025 | 274.5 | 37 / 37 | +3.878 | 3.878 | no |
| q-neuron-fixed-019-dt0025 | 0.19 | 0.0025 | 194.7 | 14 / 14 | +7.423 | 7.423 | no |
| q-neuron-fixed-027-dt00125 | 0.27 | 0.00125 | 428.4 | 37 / 37 | +1.955 | 1.955 | no |
| q-neuron-fixed-019-dt00125 | 0.19 | 0.00125 | 421.5 | 14 / 14 | +3.722 | 3.722 | no |
| q-neuron-fixed-027-dt000625 | 0.27 | 0.000625 | 666.6 | 37 / 37 | +0.981 | 0.981 | yes |
| q-neuron-fixed-019-dt000625 | 0.19 | 0.000625 | 351.1 | 14 / 14 | +1.864 | 1.864 | no |

Adjacent-dt decision limits (error at dt divided by error at dt/2; first order predicts 2; the event-matched column pairs the same event index):

| Input (nA) | dt -> dt/2 (ms) | Last-event ratio | Max-abs ratio | Common event | Rise errors at common event (ms) | Event-matched ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 0.19 | 0.005 -> 0.0025 | 1.988 | 1.988 | 14 | +14.755 / +7.423 | 1.988 |
| 0.19 | 0.0025 -> 0.00125 | 1.994 | 1.994 | 14 | +7.423 / +3.722 | 1.994 |
| 0.19 | 0.00125 -> 0.000625 | 1.997 | 1.997 | 14 | +3.722 / +1.864 | 1.997 |
| 0.27 | 0.005 -> 0.0025 | 1.914 | 1.914 | 36 | +7.420 / +3.770 | 1.968 |
| 0.27 | 0.0025 -> 0.00125 | 1.984 | 1.984 | 37 | +3.878 / +1.955 | 1.984 |
| 0.27 | 0.00125 -> 0.000625 | 1.992 | 1.992 | 37 | +1.955 / +0.981 | 1.992 |

Met at both inputs by dt: 0.005: no, 0.0025: no, 0.00125: no, 0.000625: no.

## Confirming BrainCell full train: identity against NEURON fixed step

`a1-matched-027` (BrainCell, copied mesh, dt 0.005) vs `b1-fixed-027` (NEURON fixed step, dt 0.005, x9), window 270-1270 ms, identity gate {'rise_crossing_ms': 1e-08, 'peak_sample_voltage_mv': 1e-06, 'time_above_threshold_ms': 1e-08}:
counts 36 / 36, passed **False**, first failed event 1, max |rise| 1.42e-07 ms, max |interpolated peak| 0.000407 mV, max |sampled peak| 2.73e-06 mV, max |width| 3.25e-09 ms, max raw |dV| 8.33e-05 mV.

| Event | NEURON rise (ms) | Rise diff (ms) | Peak sampled diff (mV) | Peak interp diff (mV) | Width diff (ms) | In identity gate |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 281.206 | -4.36e-10 | -1.36e-08 | +5.42e-06 | -1.91e-11 | NO: peak_interpolated_voltage_mv |
| 2 | 297.543 | -7.37e-10 | +3.50e-09 | +3.64e-06 | +6.71e-12 | NO: peak_interpolated_voltage_mv |
| 3 | 314.577 | -1.20e-09 | -5.60e-09 | +5.19e-06 | -1.20e-11 | NO: peak_interpolated_voltage_mv |
| 4 | 332.446 | -1.77e-09 | -4.19e-08 | -1.91e-05 | -7.38e-11 | NO: peak_interpolated_voltage_mv |
| 5 | 351.189 | -2.64e-09 | +5.05e-08 | -3.31e-05 | +7.39e-11 | NO: peak_interpolated_voltage_mv |
| 6 | 370.924 | -3.09e-09 | +7.23e-08 | +3.81e-06 | +1.13e-10 | NO: peak_interpolated_voltage_mv |
| 7 | 391.801 | -3.97e-09 | -2.80e-08 | -1.49e-05 | -5.93e-11 | NO: peak_interpolated_voltage_mv |
| 8 | 413.961 | -4.60e-09 | -7.12e-08 | +1.57e-05 | -1.29e-10 | NO: peak_interpolated_voltage_mv |
| 9 | 437.493 | -5.64e-09 | +6.63e-08 | -3.98e-05 | +9.74e-11 | NO: peak_interpolated_voltage_mv |
| 10 | 462.410 | -7.54e-09 | -1.67e-07 | +4.05e-06 | +1.97e-10 | NO: peak_interpolated_voltage_mv |
| 11 | 488.627 | -8.29e-09 | -7.45e-09 | -3.70e-05 | -3.99e-11 | NO: peak_interpolated_voltage_mv |
| 12 | 515.976 | -9.75e-09 | -1.15e-07 | -4.05e-05 | -2.20e-10 | NO: peak_interpolated_voltage_mv |
| 13 | 544.241 | -1.29e-08 | -1.18e-07 | -3.48e-05 | -2.34e-10 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 14 | 573.202 | -1.68e-08 | +1.62e-08 | -6.51e-05 | -3.41e-11 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 15 | 602.666 | -2.18e-08 | -2.66e-07 | -1.00e-05 | -4.94e-10 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 16 | 632.480 | -2.96e-08 | -6.08e-07 | +6.44e-06 | +5.92e-10 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 17 | 662.532 | -3.41e-08 | +1.13e-07 | +3.72e-05 | +6.46e-11 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 18 | 692.745 | -4.06e-08 | -5.19e-07 | +2.81e-05 | -9.52e-10 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 19 | 723.067 | -5.04e-08 | -1.13e-07 | +1.43e-05 | -3.51e-10 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 20 | 753.460 | -6.03e-08 | -9.86e-07 | +2.05e-05 | -1.75e-09 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 21 | 783.903 | -7.26e-08 | +6.76e-07 | +2.91e-05 | +9.45e-10 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 22 | 814.379 | -7.68e-08 | +1.77e-06 | -7.87e-05 | +1.15e-09 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 23 | 844.879 | -7.32e-08 | -1.60e-06 | +1.66e-04 | +1.22e-09 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 24 | 875.396 | -6.67e-08 | -2.74e-07 | -7.10e-05 | -6.52e-10 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 25 | 905.926 | -6.73e-08 | -5.83e-07 | -1.76e-05 | -1.14e-09 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 26 | 936.465 | -7.77e-08 | -1.26e-06 | +1.20e-04 | +1.99e-09 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 27 | 967.012 | -8.29e-08 | -1.39e-07 | -2.77e-05 | -4.90e-10 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 28 | 997.565 | -9.81e-08 | -1.85e-06 | -1.67e-04 | +2.09e-09 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 29 | 1028.123 | -1.09e-07 | +1.46e-06 | +4.07e-04 | -1.68e-10 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 30 | 1058.687 | -1.08e-07 | +1.33e-09 | +1.81e-04 | -3.49e-10 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 31 | 1089.255 | -1.10e-07 | -1.71e-06 | -1.34e-04 | -3.04e-09 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 32 | 1119.828 | -1.26e-07 | +1.35e-06 | +1.62e-04 | -7.54e-10 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 33 | 1150.405 | -1.24e-07 | -1.82e-06 | -2.06e-04 | -3.25e-09 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 34 | 1180.987 | -1.22e-07 | +1.28e-07 | -2.29e-05 | -1.86e-10 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 35 | 1211.573 | -1.33e-07 | +1.59e-06 | -1.36e-04 | -4.77e-10 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |
| 36 | 1242.164 | -1.42e-07 | +2.73e-06 | +1.72e-05 | +1.28e-09 | NO: rise_crossing_ms, peak_interpolated_voltage_mv |

## Full-train arms against the CVode finalist (amended gate: rise 0.1 ms, interpolated peak 0.5 mV, width 0.01 ms)

`a1-matched-019`: **untested** (trace absent).

`a1-matched-027`: counts 37 / 36, passed False, first failed event 6, max |rise| 7.4203 ms.

| Event | Ref rise (ms) | Rise err (ms) | Width err (ms) | Peak sampled (mV) | Peak interp (mV) | Richardson peak err (mV) | In gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 281.198 | +0.0085 | +0.0029 | -0.4537 | -0.4349 | -0.0080 | yes |
| 2 | 297.558 | -0.0151 | +0.0030 | -0.4336 | -0.4343 | -0.0120 | yes |
| 3 | 314.595 | -0.0176 | +0.0030 | -0.4317 | -0.4323 | -0.0104 | yes |
| 4 | 332.436 | +0.0092 | +0.0029 | -0.4496 | -0.4310 | n/a | yes |
| 5 | 351.118 | +0.0709 | +0.0030 | -0.4465 | -0.4313 | n/a | yes |
| 6 | 370.748 | +0.1759 | +0.0030 | -0.4561 | -0.4341 | n/a | NO: rise_crossing_ms |
| 7 | 391.469 | +0.3328 | +0.0030 | -0.4380 | -0.4367 | n/a | NO: rise_crossing_ms |
| 8 | 413.413 | +0.5477 | +0.0030 | -0.4468 | -0.4410 | n/a | NO: rise_crossing_ms |
| 9 | 436.673 | +0.8198 | +0.0030 | -0.4433 | -0.4333 | n/a | NO: rise_crossing_ms |
| 10 | 461.270 | +1.1396 | +0.0030 | -0.4601 | -0.4403 | n/a | NO: rise_crossing_ms |
| 11 | 487.138 | +1.4889 | +0.0030 | -0.4350 | -0.4365 | n/a | NO: rise_crossing_ms |
| 12 | 514.130 | +1.8460 | +0.0030 | -0.4395 | -0.4325 | n/a | NO: rise_crossing_ms |
| 13 | 542.049 | +2.1920 | +0.0030 | -0.4354 | -0.4375 | n/a | NO: rise_crossing_ms |
| 14 | 570.686 | +2.5159 | +0.0030 | -0.4299 | -0.4299 | n/a | NO: rise_crossing_ms |
| 15 | 599.851 | +2.8143 | +0.0029 | -0.4365 | -0.4263 | n/a | NO: rise_crossing_ms |
| 16 | 629.390 | +3.0891 | +0.0029 | -0.4529 | -0.4277 | n/a | NO: rise_crossing_ms |
| 17 | 659.188 | +3.3442 | +0.0030 | -0.4271 | -0.4221 | n/a | NO: rise_crossing_ms |
| 18 | 689.161 | +3.5848 | +0.0029 | -0.4353 | -0.4309 | n/a | NO: rise_crossing_ms |
| 19 | 719.252 | +3.8147 | +0.0030 | -0.4260 | -0.4267 | n/a | NO: rise_crossing_ms |
| 20 | 749.423 | +4.0372 | +0.0029 | -0.4402 | -0.4368 | n/a | NO: rise_crossing_ms |
| 21 | 779.648 | +4.2548 | +0.0029 | -0.4301 | -0.4233 | n/a | NO: rise_crossing_ms |
| 22 | 809.910 | +4.4692 | +0.0029 | -0.4563 | -0.4222 | n/a | NO: rise_crossing_ms |
| 23 | 840.198 | +4.6816 | +0.0029 | -0.4555 | -0.4259 | n/a | NO: rise_crossing_ms |
| 24 | 870.504 | +4.8927 | +0.0029 | -0.4261 | -0.4115 | n/a | NO: rise_crossing_ms |
| 25 | 900.823 | +5.1031 | +0.0029 | -0.4296 | -0.4328 | n/a | NO: rise_crossing_ms |
| 26 | 931.152 | +5.3133 | +0.0029 | -0.4422 | -0.4162 | n/a | NO: rise_crossing_ms |
| 27 | 961.488 | +5.5232 | +0.0030 | -0.4250 | -0.4223 | n/a | NO: rise_crossing_ms |
| 28 | 991.831 | +5.7332 | +0.0029 | -0.4483 | -0.4236 | n/a | NO: rise_crossing_ms |
| 29 | 1022.180 | +5.9433 | +0.0029 | -0.4353 | -0.4161 | n/a | NO: rise_crossing_ms |
| 30 | 1052.533 | +6.1536 | +0.0030 | -0.4248 | -0.4223 | n/a | NO: rise_crossing_ms |
| 31 | 1082.891 | +6.3642 | +0.0029 | -0.4389 | -0.4134 | n/a | NO: rise_crossing_ms |
| 32 | 1113.253 | +6.5749 | +0.0029 | -0.4319 | -0.4604 | n/a | NO: rise_crossing_ms |
| 33 | 1143.619 | +6.7860 | +0.0029 | -0.4375 | -0.4248 | n/a | NO: rise_crossing_ms |
| 34 | 1173.990 | +6.9971 | +0.0030 | -0.4251 | -0.4187 | n/a | NO: rise_crossing_ms |
| 35 | 1204.365 | +7.2086 | +0.0029 | -0.4337 | -0.4280 | n/a | NO: rise_crossing_ms |
| 36 | 1234.744 | +7.4203 | +0.0029 | -0.4480 | -0.4330 | n/a | NO: rise_crossing_ms |

`b1-fixed-019`: counts 14 / 14, passed False, first failed event 3, max |rise| 14.7546 ms.

| Event | Ref rise (ms) | Rise err (ms) | Width err (ms) | Peak sampled (mV) | Peak interp (mV) | Richardson peak err (mV) | In gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 298.446 | +0.0169 | +0.0029 | -0.4314 | -0.4300 | n/a | yes |
| 2 | 333.166 | +0.0343 | +0.0029 | -0.4626 | -0.4318 | n/a | yes |
| 3 | 372.293 | +0.2171 | +0.0029 | -0.4604 | -0.4278 | n/a | NO: rise_crossing_ms |
| 4 | 417.334 | +0.7191 | +0.0029 | -0.4331 | -0.4270 | n/a | NO: rise_crossing_ms |
| 5 | 470.934 | +1.7964 | +0.0029 | -0.4354 | -0.4247 | n/a | NO: rise_crossing_ms |
| 6 | 535.303 | +3.5433 | +0.0029 | -0.4284 | -0.4178 | n/a | NO: rise_crossing_ms |
| 7 | 609.229 | +5.4549 | +0.0029 | -0.4398 | -0.4114 | n/a | NO: rise_crossing_ms |
| 8 | 688.519 | +7.0728 | +0.0029 | -0.4223 | -0.4225 | n/a | NO: rise_crossing_ms |
| 9 | 770.034 | +8.4515 | +0.0029 | -0.4284 | -0.4104 | n/a | NO: rise_crossing_ms |
| 10 | 852.510 | +9.7327 | +0.0029 | -0.4200 | -0.4298 | n/a | NO: rise_crossing_ms |
| 11 | 935.546 | +10.9853 | +0.0029 | -0.4233 | -0.4079 | n/a | NO: rise_crossing_ms |
| 12 | 1019.024 | +12.2351 | +0.0029 | -0.4391 | -0.4128 | n/a | NO: rise_crossing_ms |
| 13 | 1102.902 | +13.4912 | +0.0029 | -0.4283 | -0.4376 | n/a | NO: rise_crossing_ms |
| 14 | 1187.160 | +14.7546 | +0.0029 | -0.4505 | -0.4255 | n/a | NO: rise_crossing_ms |

`b1-fixed-027`: counts 37 / 36, passed False, first failed event 6, max |rise| 7.4203 ms.

| Event | Ref rise (ms) | Rise err (ms) | Width err (ms) | Peak sampled (mV) | Peak interp (mV) | Richardson peak err (mV) | In gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 281.198 | +0.0085 | +0.0029 | -0.4537 | -0.4349 | -0.0081 | yes |
| 2 | 297.558 | -0.0151 | +0.0030 | -0.4336 | -0.4343 | -0.0120 | yes |
| 3 | 314.595 | -0.0176 | +0.0030 | -0.4317 | -0.4323 | -0.0106 | yes |
| 4 | 332.436 | +0.0092 | +0.0029 | -0.4496 | -0.4309 | n/a | yes |
| 5 | 351.118 | +0.0709 | +0.0030 | -0.4465 | -0.4313 | n/a | yes |
| 6 | 370.748 | +0.1759 | +0.0030 | -0.4561 | -0.4341 | n/a | NO: rise_crossing_ms |
| 7 | 391.469 | +0.3328 | +0.0030 | -0.4380 | -0.4366 | n/a | NO: rise_crossing_ms |
| 8 | 413.413 | +0.5477 | +0.0030 | -0.4468 | -0.4410 | n/a | NO: rise_crossing_ms |
| 9 | 436.673 | +0.8198 | +0.0030 | -0.4433 | -0.4333 | n/a | NO: rise_crossing_ms |
| 10 | 461.270 | +1.1396 | +0.0030 | -0.4601 | -0.4403 | n/a | NO: rise_crossing_ms |
| 11 | 487.138 | +1.4889 | +0.0030 | -0.4350 | -0.4364 | n/a | NO: rise_crossing_ms |
| 12 | 514.130 | +1.8460 | +0.0030 | -0.4395 | -0.4325 | n/a | NO: rise_crossing_ms |
| 13 | 542.049 | +2.1920 | +0.0030 | -0.4354 | -0.4375 | n/a | NO: rise_crossing_ms |
| 14 | 570.686 | +2.5159 | +0.0030 | -0.4299 | -0.4298 | n/a | NO: rise_crossing_ms |
| 15 | 599.851 | +2.8143 | +0.0029 | -0.4365 | -0.4263 | n/a | NO: rise_crossing_ms |
| 16 | 629.390 | +3.0891 | +0.0029 | -0.4529 | -0.4277 | n/a | NO: rise_crossing_ms |
| 17 | 659.188 | +3.3442 | +0.0030 | -0.4271 | -0.4222 | n/a | NO: rise_crossing_ms |
| 18 | 689.161 | +3.5848 | +0.0029 | -0.4353 | -0.4310 | n/a | NO: rise_crossing_ms |
| 19 | 719.252 | +3.8147 | +0.0030 | -0.4260 | -0.4267 | n/a | NO: rise_crossing_ms |
| 20 | 749.423 | +4.0372 | +0.0029 | -0.4402 | -0.4368 | n/a | NO: rise_crossing_ms |
| 21 | 779.648 | +4.2548 | +0.0029 | -0.4301 | -0.4234 | n/a | NO: rise_crossing_ms |
| 22 | 809.910 | +4.4692 | +0.0029 | -0.4563 | -0.4221 | n/a | NO: rise_crossing_ms |
| 23 | 840.198 | +4.6816 | +0.0029 | -0.4554 | -0.4261 | n/a | NO: rise_crossing_ms |
| 24 | 870.504 | +4.8927 | +0.0029 | -0.4261 | -0.4114 | n/a | NO: rise_crossing_ms |
| 25 | 900.823 | +5.1031 | +0.0029 | -0.4296 | -0.4328 | n/a | NO: rise_crossing_ms |
| 26 | 931.152 | +5.3133 | +0.0029 | -0.4422 | -0.4163 | n/a | NO: rise_crossing_ms |
| 27 | 961.488 | +5.5232 | +0.0030 | -0.4250 | -0.4223 | n/a | NO: rise_crossing_ms |
| 28 | 991.831 | +5.7332 | +0.0029 | -0.4483 | -0.4234 | n/a | NO: rise_crossing_ms |
| 29 | 1022.180 | +5.9433 | +0.0029 | -0.4353 | -0.4165 | n/a | NO: rise_crossing_ms |
| 30 | 1052.533 | +6.1536 | +0.0030 | -0.4248 | -0.4225 | n/a | NO: rise_crossing_ms |
| 31 | 1082.891 | +6.3642 | +0.0029 | -0.4389 | -0.4133 | n/a | NO: rise_crossing_ms |
| 32 | 1113.253 | +6.5749 | +0.0029 | -0.4319 | -0.4605 | n/a | NO: rise_crossing_ms |
| 33 | 1143.619 | +6.7860 | +0.0029 | -0.4375 | -0.4246 | n/a | NO: rise_crossing_ms |
| 34 | 1173.990 | +6.9971 | +0.0030 | -0.4251 | -0.4187 | n/a | NO: rise_crossing_ms |
| 35 | 1204.365 | +7.2086 | +0.0029 | -0.4337 | -0.4278 | n/a | NO: rise_crossing_ms |
| 36 | 1234.744 | +7.4203 | +0.0029 | -0.4480 | -0.4331 | n/a | NO: rise_crossing_ms |

## Halving pairs (gate validity, half gate 0.05 ms / 0.25 mV / 0.005 ms)

braincell_halving: passed True, max |rise| 0.00881 ms (rows in `sp2-i-decision.json`).
neuron_halving: passed True, max |rise| 0.00881 ms (rows in `sp2-i-decision.json`).

## Prediction outcomes

- `a_identity_basis_reproduced`: held
- `b_first_order_halving_held`: held
- `b_one_ms_met_at_dt_000625_at_027`: held
- `b_one_ms_met_at_both_inputs_within_cap`: NOT held
- `c_full_train_identity_within_1e-6_mV_1e-8_ms`: NOT held
- `c_same_failing_events_as_b1`: held

## Runs

Order: q-neuron-fixed-027-dt0025, q-neuron-fixed-019-dt0025, q-neuron-fixed-027-dt00125, q-neuron-fixed-019-dt00125, q-neuron-fixed-027-dt000625, q-neuron-fixed-019-dt000625, a1-matched-027. NEURON runs used 6 of cap 6; not launched: none.
Launcher: PowerShell Start-Process per arm (stdout/stderr persisted, abort by docker kill or process-tree stop, timing JSON written by the launcher); polled with sleeps under 10 min. Another agent's NEURON container (h01-e-gain-g0-b3-sweep53/56) ran during the first five NEURON runs; it had finished before q-neuron-fixed-019-dt000625 and a1-matched-027. Other python processes from other sessions were resident during the BrainCell run (one at 1.4 GB working set). Wall clocks are upper bounds on an idle host. Cost rule: each rung launched only when twice the measured adjacent-dt wall clock was under the 900 s abort; derived values were 549, 389, 857 and 843 s for the dt 0.00125 and 0.000625 rungs (never quoted as measured).
BrainCell A1 launch: abort 1500 s, silence kill False, measured 616.7 s (over the earlier 600 s watchdog by 16.7 s; the two earlier kills were at 601.5 and 602.0 s).

## Reading

Identity gate closed (user decision a): the basis arm reproduces the quoted identity values exactly (rise 1.2e-9 ms, interpolated peak 5.4e-6 mV, width 1.9e-11 ms over 270-329.5 ms; raw voltage within 7.5e-7 mV). The confirming BrainCell full train (a1-matched-027, 616.7 s, exit 0) holds 36 events like NEURON fixed step at the same dt and mesh and agrees with it over 270-1270 ms to 1.4e-7 ms rise, 2.7e-6 mV sampled peak, 3.2e-9 ms width and 8.3e-5 mV raw voltage, the difference growing smoothly along the train (floating-point accumulation: 4e-10 ms at event 1, 1.4e-7 ms at event 36). The registered (c) literal, identity within 1e-6 mV / 1e-8 ms at every event, is REFUTED as written: the interpolated peak is already 5.4e-6 mV at event 1 (above the 1e-6 mV literal; first failed event 1), the rise difference passes 1e-8 ms at event 13, and the interpolated-peak column reaches 4.1e-4 mV because the parabola vertex amplifies 1e-6 mV sample differences on a flat peak (the sampled peak stays under 2.7e-6 mV). Against the CVode finalist a1-matched-027 fails at exactly B1's events: count 36 vs 37, first failure at event 6, max rise 7.4203056 ms (B1 7.4203057 ms); the (c) prediction 'same failing events as B1' held. dt qualification: the late-event rise error against CVode halves with every dt halving at both inputs (event-matched ratios 1.97/1.98/1.99 at 0.27 nA, 1.99/1.99/2.00 at 0.19 nA; prediction held, rejection not met, the CVode reference is not questioned). At 0.27 nA the 1 ms tolerance is met at dt 0.000625 (37/37 events, max 0.981 ms at event 37; 0.954 ms at event 36). At 0.19 nA the error is 1.864 ms at dt 0.000625 (14/14), so the both-inputs rule is 'not reached within cap' (six NEURON runs used); first order projects about 0.93 ms at dt 0.0003125 for 0.19 nA, which is one more halving beyond the registered series and is not a measurement. The fixed-step drift is per-interval, first order, and identical in both simulators; the transfer is qualified as a numerical statement about dt, not a simulator or mesh fault.

## Hashes

Inputs (sha256): `r-braincell-matched-027` 168ab9d36ee46855...; `r-braincell-matched-027-halfdt` 7b4fdc4b5fe0f2e3...; `r-neuron-fixed-027-halfdt` 891d251683e665e0...; `a1-matched-027` 0692450105c41870...; `b1-fixed-027` f1ac69c3640cd28d...; `b1-fixed-019` 50461e970ee59ca7...; `q-neuron-fixed-027-dt0025` ba9b91a7dd28e592...; `q-neuron-fixed-019-dt0025` 89cc5c127b6779f9...; `q-neuron-fixed-027-dt00125` 6c20d74f21e62d1a...; `q-neuron-fixed-019-dt00125` 4ac5bf54b02f7a57...; `q-neuron-fixed-027-dt000625` 18c2f7c26e821505...; `q-neuron-fixed-019-dt000625` 4fbc16a0e069c501....
References (sha256, sibling worktree `C:\Users\J\Documents\Projects\connectome-agent\braintrace-source\.worktrees\h01-braincell`): `027` 85a9b734b909fc4a...; `019` be3bda0567d603f3....

Qualification: Numerical transfer only on donor anatomy; identifies no channel cause and no human waveform validity.
