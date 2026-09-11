# SP10. I cell: isolate the axonal SK effect on the finalist

Status: spec written 2026-09-11 before any run. Branch
`campaign/h01-currents-2026-09-11`. Manifest
`docs/evidence/h01-i-sk-manifest.json`, output `docs/evidence/h01-i-sk/`.
Executor: the Vast.ai box (`ssh braintrace-gpu`), NEURON 9.0.2 in a Python 3.12
venv, mechanisms compiled from the same mod files as the reference image. The
campaign runner gained an `executor: local` mode for this (no Docker on the box).

## Question

The causal model's Y3 section leaves the late trough-to-threshold delay of the PV
finalist (`e-kv3-close2`) without an isolated cause. Along the train the axonal
calcium rises (0.000120 to 0.000199 mM at 0.23 nA), the axonal SK gate rises
(0.0022 to 0.024) and the trough-to-threshold delay grows (17.6 to 40.3 ms). The
earlier "no axonal SK" arm (57/138 spikes) was not a single-change comparison: it
also returned somatic NaTg to 1.0 and used Kv3 closing factor 0.5
([correction](../evidence/h01-causal-current-observations-2026-09-09.md)).

The test registered there is run here as written: the finalist against an
otherwise identical candidate with axonal SK density zero, at 0.19 and 0.27 nA,
recording the axonal SK current, calcium and gate with the existing soma and
axial probes.

## Stages

| Stage | Candidate | Single change | Inputs |
| --- | --- | --- | --- |
| 0 | `s0-finalist` | none (byte-identical flags to `e-kv3-close2`) | 0.19, 0.27 nA |
| 1 | `s1-sk-axon0` | `--scale SK:axon:0` | 0.19, 0.27 nA |

Stage 0 is the platform transfer: the same model on the box must reproduce the
container's counts (14, 37), cycle 2 (34.76, 16.37 ms within 0.1 ms) and troughs
1-3 (within 0.1 mV). Stage 1 opens only on a recorded stage-0 decision.

## Pre-registered prediction and rejection (stage 1)

If axonal SK is the direct path of the growing delay, removing it shortens the late
trough-to-threshold delay by more than the 12.8 ms cycle repeatability limit at both
inputs and raises the counts above 14 and 37, while cycle 2 moves by less than
12.8 ms (the SK gate is below 0.003 there) and widths, peaks and early troughs stay
within the stage-0 bands.

Rejection: the delay does not shorten beyond 12.8 ms although the SK current is
removed; or spikes are lost; or the soma enters depolarisation block (above
-20 mV for more than 5 ms). A count rise away from the human (12, 43) is a
mechanism observation, not a promotion: the human's early refire (cycle 2 at or
below 22 ms) is not predicted to change, and the finalist stays experimental.

## Analysis

`docs/evidence/h01_i_sk_isolation.py` reads both stems, locates cycles with
`h01_spike_cycle_energetics.landmarks`, and reports per cycle: trough time and
voltage, trough-to-next-threshold delay, axonal SK current, calcium and gate at
trough+6 ms, and the soma axial current. It writes
`h01-i-sk/stage-1-decision.json` with each band and its verdict, and the causal
model Y3 section is updated in the same commit.

## Limits

Two evaluations. Abort 1800 s per run. No holdout is opened (0.23 nA is spent;
sweep 48 stays sealed). Both tiers (1 mV contract and usable) are reported for
every arm; a mechanism result is not a contract pass.
