# Driven window and anatomy transfer on the Vast executor (2026-09-15)

Spec: [driven window and anatomy transfer](../../specs/2026-09-15-h01-driven-window-and-anatomy-transfer.md).
Executor: Vast 50616476 (RTX 4090, 255 cores, 503 GB), `/workspace/venv314`,
branch `campaign/h01-driven-window-20260914` at `995ba535`. Every run here has a
receipt under `var/h01-driven/<label>/` on the box (launch.json, run.log, run.err,
time.txt, gpu-samples.txt, terminal.json); the copies committed here are named
in each section.

## Headline

| Term | Before | After this stage | Basis |
| --- | ---: | ---: | --- |
| driven_window | 0.000 | PENDING | four 50 ms controls + refinement pair, existing gates |
| anatomy_transfer | 0.000 (unavailable) | **0.250 measured** | 1 of 4 deployed donors holds its human count on retained H01 anatomy |

The other four terms are unchanged (donor 0.718, coverage 55/104, construction
1.000, timestep 1.000). No score was redefined, no denominator changed, no cell
dropped.

## Stage A: the 104-cell network runs on Vast

`bench-104-ei-1ms`: 104 cells, dt 0.000625 ms, 1 ms (1,600 steps), control `ei`,
1 nA 2-5 ms probe, implicit calcium solver, `h01-population-components.json`.

| Measurement | Vast (this stage) | Windows laptop (2026-09-08) |
| --- | ---: | ---: |
| Construction | 169 s | 1,126 s |
| init_state | 177 s | 837 s |
| Compile + 1,600 steps | 479 s (≈370 s compile, ≈62 ms/step) | 267 s for 200 steps at dt 0.005 |
| Host peak RSS | 18.7 GB | 13.9 GB |
| GPU memory | 1.6 GB | — |
| Traces | all finite | 10 ms explicit run nonfinite; 10 ms implicit run wall-cap abort at 21,286 s |

Files: `bench-104-ei-1ms-build.json` (Vast build record), `bench-104-build-gate.json`
(`h01_population_build_gate` PASS: 104 unique cells, 808,495 compartments, contacts
and source hashes equal to the import audit).

**Construction reference.** Cell IDs, donors, per-cell compartment counts, total
and contacts are identical to the Windows reference. The `cells` records differ
for 17 cells in `output_cv_id`: see `output-site-tie-break.json`. The soma sample is a
CV boundary, so the two adjacent CV midpoints are equidistant and the bare
nearest-midpoint rule was decided by last-bit rounding (max interval discrepancy
1.31e-12). Neither choice is physically wrong (adjacent CVs, both bounded by the
soma sample). Production fix on this branch: `_nearest_cv` in
`braintrace/datasets/h01_network.py` resolves ties within 1e-9 to the lowest index,
with a test on the real fractions of cell 4157825456. The Vast build is the
reference for every Vast run below and records the sites actually probed.

## Stage B: driven window, four matched controls (104 cells, 50 ms)

Registered plans: `plan-ctrl-{ei,e_only,i_only,disconnected}-50ms.json`,
`plan-refine-ei-10ms-dt{000625,0003125}.json`.

PENDING — filled in when the runs complete.

**Execution failures preserved.** The first launch ran six population processes
and the transfer chain concurrently. Each unpinned JAX process holds ~1,300
threads (255-CPU thread pools) and the container's cgroup pids limit is 7,680
(read-only from inside); at 04:37 UTC every population process aborted with
`pthread_create failed: Resource temporarily unavailable` (exit 134) or `Failed
to launch ptxas`, eight minutes into stepping. Receipts: `var/h01-driven/
{ctrl-*-50ms,ctrl-e_only-50ms-r2,refine-ei-10ms-dt*}` on the box. The second
launch (`-r3`) pinned each process with `taskset -c` (64 CPUs) and
`XLA_FLAGS=--xla_gpu_force_compilation_parallelism=8`; pinning is not a bound:
two processes at the same stage held 975 and 4,775 threads, and at 04:53 UTC
three of six aborted the same way. The third population process was then killed
by hand (`refine-ei-10ms-dt000625-r3`, exit 143, `operator-kill.txt`) to keep
the two 50 ms controls under the limit. The remaining runs go through
`var/h01-driven/queue.sh`: at most two population processes at a time, the
second launched only when the container's thread total is under 4,200.

## Stage C: anatomy transfer, one type-matched H01 cell per deployed donor

Runner `h01_anatomy_transfer_run.py` builds the cell exactly as the population
does (same importer, component, `_regions` partition, donor profile, MaxCVLen
10 um, implicit solver) and drives the soma with the donor recording's step.
Artifacts: `transfer/<label>/{run.json,launch.json,terminal.json,run-decimated.npz}`
(traces decimated 20x to 0.1 ms; full arrays and their SHA256 stay on the box).
Decision: `anatomy-transfer-decision.json` (assembler `h01_anatomy_transfer_decision.py`).

| Donor | H01 cell | Membrane (soma / axon / dend, um²) | Input | Human count | Donor fit on own anatomy | On H01 anatomy | Verdict |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| L2 pyramidal (Allen 541563728, B3) | 955432427 | 149 / 130 / 4,029 | 310 pA | 10 | 10 | **2, then depolarisation block at -27 mV that outlasts the pulse** | rejected |
| same | same | same | 200 pA | 1 (repeats 1,1,1,1,1,0,0) | 4 | 2, then block | rejected (band missed) |
| L4 pyramidal (Allen 527952884) | 3761379470 | 199 / 29 / 3,548 | 90 pA | 12 | 8 | **13, regular train, repolarises to -84 mV** | within one → pass |
| L3 SST (HL5MN1) | 4420044370 | 107 / 12 / 2,210 | 100 pA | 14 (12-14) | 16 | fires spontaneously at ~80 Hz before the pulse from a -56 mV "rest", then blocks at -34 mV; 4 in the pulse | rejected |
| L5 PV basket (HL5BN1) | 4853956860 | 315 / 30 / 4,099 | 0.19 nA | 12 | 14 | 0; plateau at -37 mV, recovers after the pulse | rejected |
| same | same | same | 0.27 nA | 43 | 37 | 0; plateau at -37 mV | rejected |

dt-half repeats (0.0025 ms) of the four primary inputs: PENDING.

**Registered prediction: falsified.** The spec predicted counts at or above the
donor fit's own count because the truncated components carry less load. Three of
four donors do the opposite: the cell enters a depolarised stable state after
one or two spikes (L2), a spike-less plateau (PV), or is spontaneously active
(SST). The L2 state persists 80 ms after the pulse ends with zero input, so it
is a second stable resting state, not slow recovery.

**Load split (diagnostic, not a promotion).** `h01_transfer_load_split.py`
multiplies only the dendritic passive load (dend/apic capacitance and leak) of
the L2 profile on the same cell and input: x1 → 2 spikes then block; x3 → 1 spike
then block (`split-e310-dendload-x3`); x10 → silent, peak -62 mV
(`split-e310-dendload-x10`). Adding passive load suppresses the spike before it
removes the block, so the block is not the small dendritic load. It is a
property of the somatic active profile on this soma geometry (149 um² of soma
region against the Allen donor's ~590 um² sphere; H01 skeleton radii are as
small as 0.032 um, mean process diameter 0.4 um). What the block is made of
(which somatic current holds -27 mV) is open; nothing here names a channel.

**What this establishes.** Donor conductance densities fitted on a full
reconstruction do not transfer, as densities, onto H01 proofread-skeleton
components of 2-4 x 10³ um². One of four does (L4, 13 versus 12). The term is a
measured 0.25, not "unavailable". The path that is left for the objective is a
per-cell fit of densities on each H01 cell's own anatomy against the type-matched
human recording; see the next-stage spec named in the closing section.

## Type coverage lead (no change to the term)

`allen-human-perisomatic-models.json`: the Allen API lists 43 human perisomatic
fits (same template and import path as the two deployed Allen donors): L2 spiny 3,
L2 sparsely spiny 2, L3 aspiny 5, L3 spiny 20, L4 spiny 4, L5 aspiny 1, L5 spiny 5,
L6 spiny 3. That covers, by layer and class, the 6 L3 pyramidal, 7 L5 pyramidal,
2 L6 pyramidal and 4 L5 interneuron H01 cells that currently have no matched
donor. Nothing is imported; given stage C, importing more donors for density
transfer is not the next step.

## Verification

Tests added on this branch (all pass): `h01_anatomy_transfer_run_test.py` (8),
`h01_driven_window_gates_test.py` (2), `h01_anatomy_transfer_decision_test.py` (2),
`h01_population_ledger_test.py` (12, both with and without the Vast decisions),
`h01_network_test.py` (55, including the tie-break reproduction).
