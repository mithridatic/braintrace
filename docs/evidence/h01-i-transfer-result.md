# SP2 transfer isolation, I side: result (2026-09-07)

Spec: [SP2 transfer isolation](../specs/2026-09-07-h01-transfer-isolation.md) with the amendments of
2026-09-07 (interpolated peak metric; user decision: full-train peak row 0.5 mV, Richardson column).
Steps 1 and 1b: [halving pair](h01-i-transfer-step1.md), `h01-i-transfer/step1-decision.json`,
`h01-i-transfer/step1b-decision.json`. Runner output for this page: `h01-i-transfer/sp2-i-decision.json`
(`--score --peak-method interpolated --reference-root <sibling worktree>`; every per-event row, both
peak metrics, the Richardson column, timing gates, wall clocks and hashes are in it; this page is
rendered from it).

**Decision literal: `untested`.** Gate valid (both halving pairs inside the half gate):
True. A1 passed: {'a1-matched-019': False, 'a1-matched-027': False}. B1 passed: {'b1-fixed-019': False, 'b1-fixed-027': False}.
A0 passed: {'a0-maxcv-027': False}. First failed A1 event: {'a1-matched-019': None, 'a1-matched-027': None}.
Untested arms: `a0-maxcv-027`, `a1-matched-027`, `a1-matched-019`, `a1-matched-023`.

## Amended gate (user decision, registered before any run)

Count equal; rise crossing 0.1 ms; time above -20 mV 0.01 ms; peak row: dt-0.005 **interpolated** peak
error under 0.5 mV (half the human contract's 1 mV). Halving pairs: half the same gate (0.05 ms /
0.25 mV / 0.005 ms). The first-order Richardson peak, 2 x peak(dt 0.0025) - peak(dt 0.005), is a
derived column filled where a dt-0.0025 partner trace exists (330 ms partners: events inside
270-329.5 ms), never gated. Rationale: step 1b located the peak deficit in the integrator's first-order
convergence, the usable tier does not score the peak, and the contract tolerates 1 mV.

Registered prediction: A1 passes timing and the 0.5 mV peak row at every event at 0.27 and 0.19 nA over
270-1270 ms; A0 fails at least one late rise crossing; B1 matches CVode inside the timing gate.
Held: {'a1_passes_every_event_at_019_and_027': False, 'a0_fails_a_late_rise_crossing': None, 'b1_matches_cvode_inside_timing_gate': False}.

## Wall clocks (measured; compile and interpreter start included)

| Arm | Simulator | dt (ms) | Simulated (ms) | Wall clock (s) | Exit | Killed | Other containers at start |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r-braincell-matched-027 | braincell | 0.005 | 330 | 115.6 | 0 | no | none |
| r-braincell-matched-027-halfdt | braincell | 0.0025 | 330 | 246.0 | 0 | no | none |
| r-neuron-fixed-027-halfdt | neuron | 0.0025 | 330 | 121.8 | 0 | no | h01-e-gain-g0-b3-sweep53,synapse |
| a0-maxcv-027 | braincell | 0.005 | 1500 | not run | | | |
| a1-matched-027 | braincell | 0.005 | 1500 | 601.5 | None | yes | h01-e-gain-g0-b3-sweep50,synapse |
| a1-matched-019 | braincell | 0.005 | 1500 | not run | | | |
| b1-fixed-027 | neuron | 0.005 | 1500 | 232.4 | 0 | no | h01-e-gain-g0-b3-sweep53,synapse |
| b1-fixed-019 | neuron | 0.005 | 1500 | 152.8 | 0 | no | h01-e-gain-g0-b3-sweep53,synapse |
| a1-matched-023 | braincell | 0.005 | 1500 | not run | | | |

The derived 525 s full-train figure was an upper-bound estimate, not a measurement; the measured values
above supersede it. Other containers were running on the host during these runs (listed per arm), so the
wall clocks are upper bounds on an idle host.

## Per-input verdicts

| Input (nA) | Arm | Verdict under the amended gate |
| --- | --- | --- |
| 0.27 | `a0-maxcv-027` | untested |
| 0.27 | `a1-matched-027` | untested |
| 0.27 | `b1-fixed-027` | fails at event 6 (rise_crossing_ms) |
| 0.19 | `a1-matched-019` | untested |
| 0.19 | `b1-fixed-019` | fails at event 3 (rise_crossing_ms) |
| 0.23 (spent-holdout control) | `a1-matched-023` | untested |

## Halving pairs (gate validity)

`r-braincell-matched-027` (dt 0.005) vs `r-braincell-matched-027-halfdt` (dt 0.0025), 270-329.5 ms, half gate 0.05 ms / 0.25 mV / 0.005 ms: **inside**.

| Event | Rise diff (ms) | Width diff (ms) | Peak sampled diff (mV) | Peak interp diff (mV) | In half gate |
| --- | --- | --- | --- | --- | --- |
| 1 | -0.0043 | -0.0015 | +0.2305 | +0.2134 | yes |
| 2 | +0.0076 | -0.0015 | +0.2107 | +0.2111 | yes |
| 3 | +0.0088 | -0.0015 | +0.2113 | +0.2109 | yes |

`b1-fixed-027` (dt 0.005) vs `r-neuron-fixed-027-halfdt` (dt 0.0025), 270-329.5 ms, half gate 0.05 ms / 0.25 mV / 0.005 ms: **inside**.

| Event | Rise diff (ms) | Width diff (ms) | Peak sampled diff (mV) | Peak interp diff (mV) | In half gate |
| --- | --- | --- | --- | --- | --- |
| 1 | -0.0043 | -0.0015 | +0.2305 | +0.2134 | yes |
| 2 | +0.0076 | -0.0015 | +0.2107 | +0.2111 | yes |
| 3 | +0.0088 | -0.0015 | +0.2113 | +0.2109 | yes |

## Per-event rows, full train 270-1270 ms (errors are actual minus reference)

`a1-matched-027`: **untested** (trace absent).

`a1-matched-019`: **untested** (trace absent).

`b1-fixed-027` vs `e-kv3-close2-027`, window 270-1270.0 ms, counts 37 / 36, gate rise 0.1 ms / peak 0.5 mV / width 0.01 ms. **failed at event 6**; max |rise| 7.4203 ms.

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

`b1-fixed-019` vs `e-kv3-close2-019`, window 270-1270.0 ms, counts 14 / 14, gate rise 0.1 ms / peak 0.5 mV / width 0.01 ms. **failed at event 3**; max |rise| 14.7546 ms.

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

`a0-maxcv-027`: **untested** (trace absent).

`a1-matched-023`: **untested** (trace absent).

## Killed and not-launched BrainCell arms

| Record | Intended arm | Actual command mesh | Wall clock (s) | Outcome |
| --- | --- | --- | --- | --- |
| `a1-matched-027.killed.timing.json` | a1-matched-027 | --mesh-from (copied x9) | 602.0 | killed at the 600 s watchdog, no trace, untested |
| `mislaunch-maxcv-as-a1-027-killed.timing.json` | a1-matched-027 | --max-cv-um 2.5 | 601.5 | killed at the 600 s watchdog, no trace, untested |

The first launch of `a1-matched-027` was mis-armed by the launcher (a PowerShell case-insensitive variable
clash turned the mesh mode into the geometry path), so it ran the MaxCVLen 2.5 um full train; it is recorded
as a killed mislaunch and not as an A0 test. The corrected relaunch with `--mesh-from` was also killed at 600 s
with no trace. The derived 525 s figure therefore under-estimated the matched-mesh full train on this host
(other containers were running; listed above). Not launched, reason recorded in the JSON (`not_launched`):
`a1-matched-019`, `a1-matched-023` (identical mesh, dt and duration; cost does not depend on the current
amplitude) and `a0-maxcv-027` (its configuration was the mislaunch, already over the watchdog). Both BrainCell
logs were empty at the kill: the driver prints nothing before the end of the run.

## Supplementary: BrainCell vs NEURON fixed step at the same dt and mesh, 270-329.5 ms

`r-braincell-matched-027` scored against `b1-fixed-027` as the reference (full gate): counts
3 / 3, passed True, max |rise|
1.20e-09 ms, max |interpolated peak| 5.42e-06 mV,
max |width| 1.91e-11 ms. The two simulators at the same fixed step and mesh
produce the same train over the window available.

## Reading

B1 (NEURON fixed step, dt 0.005, x9) fails the 0.1 ms rise gate against the CVode finalist from event 6 (0.27 nA) and event 3 (0.19 nA), with a rise error growing monotonically along the train (7.4 ms and 14.8 ms at the last event) and one event fewer at 0.27 nA; width and interpolated peak (-0.43 mV) stay inside their gates. BrainCell at the copied mesh and the same dt reproduces the NEURON fixed-step trace over the 330 ms available (see supplementary). Decision-table row selected by the B1 outcome: integration (time level, not mesh); the literal stays untested because no A1 trace exists.

## What it means for Y2

See the dated Y2 entry in `docs/h01-causal-model.md` for the reading; the decision literal above is the
runner's, evaluated literally by `decide()` on the amended gate.

Qualification: Numerical transfer only on donor anatomy; identifies no channel cause and no human waveform validity.
