# Driven window and anatomy transfer on the Vast executor

Parent contract: [human unity](2026-09-14-h01-human-unity.md), all six terms, all
104 cells, true human anatomy. This stage targets the two terms at 0.000:
`driven_window` and `anatomy_transfer`. Nothing in the biology changes: same
four deployed donors, same production importer, same electrical partition, same
implicit solver and CV policy as the pinned 808,495-compartment construction.

Why now: every previous driven attempt ran on the Windows laptop (2,705 s to a
nonfinite trace at dt 0.005; 21,286 s wall-cap abort at dt 0.000625). The Vast
box (255 cores, 503 GB, RTX 4090, `/workspace/venv314`) has never run the
104-cell network. Construction there loads all 104 components in 45 s.

Executor: Vast instance 50616476, `/workspace/braintrace`, branch
`campaign/h01-driven-window-20260914` at `995ba535`. Receipts under
`var/h01-driven/<label>/` (launch.json, run.log, run.err, time.txt,
gpu-samples.txt, terminal.json) are copied into
`docs/evidence/h01-driven-window-20260915/`.

## Stage A: step-rate measurement (registered before launch)

`examples.h01_verified_network --cells 104 --include-isolated --control ei
--solver h01_staggered_calcium_implicit --current-na 1 --dt-ms 0.000625
--duration-ms 1`, components `h01-population-components.json`, wall cap 3,600 s.
Measured quantities: construction, init, compile+run for 1,600 steps, host peak
RSS, GPU memory. Sizing rule for stage B: seconds per step from this run times
80,000 steps per 50 ms control run, plus measured construction and init, times
1.25. A run that exceeds its cap is untested, not failed.

## Stage B: driven window, 104 cells, four matched controls

Unchanged registered input: 1 nA somatic pulse from 2 to 5 ms on every cell
(the population programme's assumed probe; not a physiological stimulus claim).
Duration 50 ms, the programme's minimum deliverable window. dt 0.000625 ms, the
qualified isolated-cell step. Controls `ei`, `e_only`, `i_only`, `disconnected`
with identical cells, inputs, dt and duration. Refinement pair on `ei` over the
first 10 ms at dt 0.000625 against 0.0003125.

Gates, all existing code, no new acceptance: `h01_population_runtime_gate`
(104 unique cells, model metadata equal to the pinned construction reference,
end-of-step grid, every array finite), `h01_population_control_gate` (controls
differ only at removed receivers, conductance arrives only where the pre cell
fires, `disconnected` shows none), `h01_population_refinement_gate` (1 mV
voltage, 0.05 ms event allowance, every cell). `driven_window` becomes 1.000
only if all four controls pass the runtime gate, the control gate passes, and
the refinement pair passes for all 104 cells; otherwise it stays 0.000 and the
failing cells are named. Physiology is not qualified by this term.

Falsifier: any nonfinite trace, any cell outside the refinement allowance, or
a control that changes voltage outside its removed receivers.

## Stage C: anatomy transfer, one type-matched H01 cell per deployed donor

Each deployed donor is driven on retained production-imported H01 anatomy under
its own recording's step protocol. The cell is built by the same code path as
the population (`make_h01_ei_cell` with `_regions`, `_register_cell`); only the
somatic pulse timing and amplitude change. Runner:
`docs/evidence/h01_anatomy_transfer_run.py`.

| Donor | H01 cell (type-matched, smallest) | Pulse | Input | Human count (repeats) | Donor fit on own anatomy |
| --- | --- | --- | ---: | --- | ---: |
| l2-pyramidal-allen-541563728 | 955432427 (L2 pyramidal, 3,923 CVs) | 1020-2020 ms | 0.31 nA | 10 (none) | 10 |
| l2-pyramidal-allen-541563728 | 955432427 | 1020-2020 ms | 0.20 nA | 1 (1,1,1,1,1,0,0) | 4 |
| l4-pyramidal-allen-527952884 | 3761379470 (L4 pyramidal, 1,608 CVs) | 1020-2020 ms | 0.09 nA | 12 (none) | 8 |
| l3-sst-interneuron-hl5mn1 | 4420044370 (L3 interneuron, 1,054 CVs) | 270-1270 ms | 0.10 nA | 14 (14,14,13,12) | 16 |
| l5-pv-basket-hl5bn1 | 4853956860 (L5 interneuron, 1,664 CVs) | 270-1270 ms | 0.19 nA | 12 (none) | 14 |
| l5-pv-basket-hl5bn1 | 4853956860 | 270-1270 ms | 0.27 nA | 43 (none) | 37 |

Human counts come from the retained recordings already scored in the campaign
(E: `h01-topographic/stage-0-decision.json` and cycle tables; SST and L4:
`h01_donor_reproduction.DONORS`; PV: `h01-i-sk/stage-1-decision.json` and the
causal model Y3). Recording bias currents of a few pA are omitted and stated.
dt 0.005 ms with the implicit solver, 2,100 ms (E, L4) or 1,500 ms (I), then a
dt-halving repeat of the primary input per donor for the count.

Prediction, registered: every cell is finite and fires under its donor's
suprathreshold input. The H01 components are truncated at the volume boundary
(node shares 0.88, 0.71, 1.00, 0.96), so their load is at most the donor's;
counts are predicted at or above the donor fit's own count. Falsifier: silence,
nonfinite voltage, or a count below the donor fit's count by more than one.

Verdict per donor: `count_verdict` against the human count (exact / within one /
rejected) and the human repeat band where repeats exist. The term value is the
fraction of the four donors whose primary-input count holds the human repeat
band, or is within one of the human count where no repeats exist, with finite
traces and a dt-half count that agrees. A finite firing cell that misses the
human count is recorded as a measured negative reading, not as unavailable.

Limits: this qualifies transfer of a donor fit onto human anatomy against
another human cell's recording. It does not measure the H01 donor's own cells,
and it does not qualify channel provenance.

## Caps

Stage A 3,600 s. Stage B per run: sized from stage A, hard cap 4 h each, run
sequentially on one GPU. Stage C per run 1,800 s. Any cap hit is reported as
untested with its partial receipts kept.
