# Keep/drop over the 104 H01 cells (2026-09-16)

Rule from J: simulate the human cells and let them learn; cells that work are kept,
cells that do not are dropped; no new fitting, no new donors, no tuning to rescue a
dropped cell. Spec: [2026-09-16-h01-keep-drop.md](../../specs/2026-09-16-h01-keep-drop.md).
Executor: Vast 50616476 (RTX 4090), `/workspace/braintrace`, branch
`campaign/h01-driven-window-20260914`. Causal model: Y7, keep/drop subsection.

## Result in one line

17 of 104 cells kept (13 on the L2 donor, 4 on the L4 donor, none on PV or SST); the
kept network has no contact, so the driven-window control term is
delivery-unobservable; runtime and refinement gates pass on the kept set; the
Example 21 grow/prune run on the kept manifest is recorded in Step 3 below.

## Step 1: every cell under its donor's protocol

Keep rule (registered before launch, `h01_anatomy_transfer_decision.py --keep`):
finite traces; rest (100 ms before the pulse) and return (10 ms ending 200 ms after
the pulse) both within 10 mV of the donor recording's rest ([donor-rest.json](donor-rest.json),
Allen NWBs corrected by -14 mV liquid junction potential); no -20 mV crossing before
the pulse; count within max(2, 30 percent) of the human count; and a dt-half repeat
(0.0025 ms) reproducing the count. Every candidate that held rules 1-4 got its repeat
(`h01_keep_drop_follower.py`, launched as soon as the primary run landed).

| Donor (input) | Cells | Kept | Kept counts (human) | Dropped counts seen | Drop rules |
| --- | ---: | ---: | --- | --- | --- |
| L2 Allen 541563728 (0.31 nA) | 57 | 13 | 7 x6, 8 x2, 9, 10, 11 x3 (10) | 0-6 in 40 cells; 10, 16, 83, 145 | rest/return 23, count 43 |
| L4 Allen 527952884 (0.09 nA) | 22 | 4 | 9, 11, 13, 15 (12) | 0 x6, 1 x9, 3, 4, 7 | rest/return 10, count 18 |
| L5 PV HL5BN1 (0.19 nA) | 18 | 0 | | 0 in all 18 | count 18, return 1 |
| L3 SST HL5MN1 (0.10 nA) | 7 | 0 | | 0-11; 96, 99 | spikes before the pulse in all 7 |

All 104 primary traces finite; 17 of 17 dt-half repeats reproduced the primary count
exactly. Prediction registered before launch: 20 kept, L4 the only transferring
donor. Measured: 17 kept, L2 and L4 transfer on a minority of their cells, PV silent
on every cell, SST spontaneously active on every cell. The four-cell stage C reading
(L2 block on 955432427) is not the typical L2 response: that cell drops (count 2,
return -29 mV) while 13 L2 cells hold rest, fire 7-11 spikes and return.

Kept: 1669770671, 1684504313, 2001418787, 2103991145, 2252715458, 2848552900,
3111823553, 3571083397, 3761379470, 4010150634, 4197933517, 4437316933, 5013648003,
5173982155, 5439194879, 6833911543, 751294744. Every drop with its failed rule:
[decision.json](decision.json). Receipts for all 121 runs (104 primaries, 17 repeats):
[runs/](runs/) (launch, run and terminal JSON; `trace.npz` holds voltages every 0.5 ms
and exact spike times; full traces stay on the box under `var/h01-driven/`).

Execution: launched 02:14 UTC as four chains; re-queued in donor priority at 02:40
when the launch-bound throughput (about one cell per 190 s regardless of chain
count) showed the 2 h budget was 5.4 h; last primary 09:37, last repeat 10:12,
follower done 10:14 UTC. Box time for Step 1: 8.0 h wall (process time summed over
runs: 27.6 h primaries, 7.2 h repeats, four-way shared).

## Step 2: kept-network driven window

Inputs derived from the decision by `h01_keep_drop_network.py`:
[h01-kept-network.json](../h01-kept-network.json) (17 nodes, 0 contacts: the three
construction-ready contacts each lost both endpoints to the drop list, 3955003482 a
silent L4 cell, 5584343344 and 3680152874 silent PV cells, 4188575291, 3519995546
and 4157825456 L2 cells with 6, 2 and 4 spikes), [h01-kept-components.json](../h01-kept-components.json),
[h01-kept-imports.json](../h01-kept-imports.json). Probe: each cell's donor primary
current from 2 ms for 38 ms ([kept-currents.json](kept-currents.json)). Plans
registered before launch: `plan-kept-*.json` (source commit 84c8765b, the decision commit).

| Run | Wall | Peak RSS | GPU | Steps | Spikes (17 cells) | Finite |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| kept-bench-ei-1ms | 100 s | 3.7 GB | 560 MiB | 1,600 | 0 | yes |
| kept-ctrl-ei-50ms | 765 s | 3.7 GB | 1,371 MiB | 80,000 | 31 | yes |
| kept-ctrl-e_only-50ms | 732 s | 2.9 GB | 1,371 MiB | 80,000 | 31 | yes |
| kept-ctrl-i_only-50ms | 688 s | 2.8 GB | 1,121 MiB | 80,000 | 31 | yes |
| kept-ctrl-disconnected-50ms | 690 s | 2.9 GB | 1,121 MiB | 80,000 | 31 | yes |
| kept-refine-ei-10ms-dt000625 | 212 s | 3.6 GB | 560 MiB | 16,000 | 0 | yes |
| kept-refine-ei-10ms-dt0003125 | 338 s | 3.6 GB | 560 MiB | 32,000 | 0 | yes |

Every kept cell fires 1-3 spikes inside the 38 ms window (first spike 11.6-38.0 ms),
so this is the first population window in which the probed cells actually fire (the
104-cell window's 1 nA probe drove none of the connected cells).

Gates ([kept-build-gate.json](kept-build-gate.json),
[kept-driven-window-decision.json](kept-driven-window-decision.json)):

- Build gate: PASS, 17 unique cells, 107,537 compartments, source hashes equal to the
  import audit.
- Runtime gate: PASS on all four controls (model fields equal to the bench reference,
  every array finite, registered step counts).
- Refinement gate (dt 0.000625 against 0.0003125, 10 ms): PASS, 17 of 17 cells, worst
  voltage difference 0.0014 mV, event timing difference 0 ms.
- Control gate: FAIL on its structural precondition ("checker requires exactly two
  disjoint directed pairs"): the kept network has no contact. Per the spec this is
  recorded as delivery-unobservable, not as a physics failure. The four controls are
  the same build and agree to 3.3e-6 mV (tolerance 1e-5 mV).

Receipts: [population/](population/) (launch, terminal, time and GPU samples for the
seven runs; 40x-decimated traces for `ei` and the refinement pair;
`population-run-summary.json`, `population-build-digests.json`). Box time for Step 2:
1.05 h wall (10:17-11:20 UTC). The ledger is regenerated with a `kept_cells` field; the
six terms and denominators are unchanged
([h01-population-accuracy-ledger.md](../h01-population-accuracy-ledger.md)).

## Step 3: Example 21 grow/prune on the kept manifest

Manifest: [arc-manifest/manifest.json](arc-manifest/manifest.json) (17 cells,
`original_cells` 17, 0 anatomical contacts, built from the 104-cell probe evidence
restricted to the kept list). Two launch defects were found and fixed before the run
counted: the Example 21 H01 runtime still imported the module-level geometry cache
that commit 43bd52b5 (2026-09-14, also on main) had removed, so the H01 backend had
been unlaunchable since then (fix b94d2f38, with a test); and the manifest pins the
implementation hash, so it was rebuilt against the fixed runtime.

STEP3_PLACEHOLDER

## What this does not claim

A kept cell reproduces a count and a resting level from another human donor's
recording on retained H01 anatomy under that donor's step protocol. It is not a
measurement of the H01 donor's own cell, and no channel provenance is qualified. The
kept set contains no interneuron and no contact, so functional inhibition and synaptic
delivery are not measured on it.
