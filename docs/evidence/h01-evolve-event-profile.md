# Example 21 H01 backend: seconds per ARC event on the 17 kept cells (2026-09-16)

Question: why the kept-manifest evolve run
([h01-keep-drop/evolve/evolve-record.json](h01-keep-drop/evolve/evolve-record.json): 60 min cap,
exit 124, 0 bytes stdout, no candidate) cannot finish one evolution stage, measured stage by
stage, and what one ARC event costs under the pinned numerical settings and under the
keep-test timestep. Script: [h01_evolve_event_profile.py](h01_evolve_event_profile.py)
(tests: `h01_evolve_event_profile_test.py`); it drives the existing `build_network`,
`init_h01_network_states`, `H01ArcModel`, `braintrace.pp_prop.sparse` and
`PPPropEpisodeTrainer` objects directly, without the evolve loop, and never edits a pinned
source. Machine-readable results: [h01-evolve-event-profile.json](h01-evolve-event-profile.json);
per-arm receipts under [h01-evolve-event-profile/](h01-evolve-event-profile/).

Executor: Vast 50616476 (RTX 4090, 24 GB), worktree `/workspace/braintrace-c3-profile` at
main 4918cde9, `/workspace/venv314`, `taskset -c 160-223`,
`XLA_FLAGS=--xla_gpu_force_compilation_parallelism=8`. The committed manifest
`docs/evidence/h01-keep-drop/arc-manifest/manifest.json` (17 cells, 0 contacts) pins the
same implementation hashes as main, so no throwaway manifest was needed; the dt / substep
overrides were applied to the `H01ArcModel` built in the profiling process only. ARC data:
`var/arc-agi-1` (400 training tasks, 416 queries). Events stepped: the first scheduled
training entry (task 007bbfb7, query 0, 385 advancing events), the same events in every arm.
The persistent XLA cache `JAX_COMPILATION_CACHE_DIR=/workspace/jax-cache` is set in the
box environment (see "compile versus run").

## Result in one line

At the pinned settings one forward ARC event costs 1.36 s and one learner event 13.2 s on 17
cells; `initialize()` scores the whole training corpus (116,128 advancing events, 43.8 h)
before the first training arm exists, and one 128-update block is 35,648 learner events
(130 h). The 60 min run spent about 100 s building and compiling and the remaining ~58 min
inside the initial scoring pass, roughly 2,500 of 116,128 events (about 9 of 416 queries); it
could not have reached its first arm. At dt 0.005 ms / 20 substeps the same numbers are
0.18 s, 1.6 s (real update), 5.8 h for the corpus pass and 15.8 h per block: still zero
blocks per hour. The GPU is launch-bound (about 4,300 stream launches per cable substep,
17 cells stepped as 17 separate kernel sets), at 12-13 million compartment-updates per second.

## 1. Stage times, 17 cells (107,537 compartments)

Wall seconds, one process per arm. "Compile" columns include one execution.

| Stage | pinned dt 0.000625 / 160 substeps | dt 0.005 / 20 substeps |
| --- | ---: | ---: |
| manifest load + archive open | 0.06 | 0.08 |
| `build_network` (17 cells) | 20.2 | 21.6 |
| `init_h01_network_states` | 23.1 | 28.1 |
| `H01ArcModel` setup | 2.2 | 2.6 |
| first `update(event)` (XLA compile + 1 run) | 41.3 | 42.2 |
| steady-state forward, median s/event (20 events) | 1.357 (min 1.348, max 1.49) | 0.180 (min 0.178, max 0.38) |
| forward s/substep inside the event loop | 0.0085 | 0.0090 |
| `learner.compile_graph` (ETP graph analysis, host) | 7.6 | 9.9 |
| learner update, 8 events, compile + 1 run | 402.6 | 313.7 |
| learner update, 8 events, warm (5 runs, median) | 105.8 | 11.6 |
| learner update, 2 events, compile + 1 run | 347.8 | 343.6 |
| learner update, 2 events, warm (5 runs, median) | 26.7 | 3.6 |
| learner s/event (slope 8 vs 2 events) | 13.18 | 1.34 |
| learner fixed s/update (intercept) | 0.32 | 0.93 |
| real `update_episode` (705-event payload, 385 advancing), compile + 1 run | not run (est. 1.4 h) | 1035.6 |
| real `update_episode`, warm | not run | 614.3 = 1.595 s per advancing event |
| real `score_episode` (705 events, 385 advancing), compile + 1 run | not run | 122.0 |
| real `score_episode`, warm | not run | 62.2 = 0.162 s per advancing event |

Memory (peak device bytes in use / host peak RSS): after build+init 13 MiB / 2.0 GB; forward
steady state 63 MiB / 4.2 GB; learner compiled and running, pinned 4.2 GB / 14.4 GB, dt 0.005
0.72 GB / 10.7 GB (0.75 GB / 15.0 GB after the real 705-event update). Host RSS is dominated by
JAX tracing and compilation of the learner (the 5.4-10 GB jump happens during the learner
compile, not during stepping). Eligibility layout: 3,344,642 elements, 26.8 MB, 323 state
blocks, 1 colour (same at both dt).

Voltages are finite in every arm. Run-to-run reproducibility of the forward path: two pinned
processes agree to 5e-11 mV over 3 events.

## 2. Compile versus run (coordinator question, measured on the pinned configuration)

The "compile + 1 run" rows above are one-time XLA compiles; the warm rows are run time.
`pinned-recompile` arm: the same 2-event learner update called four times under
`jax.log_compiles(True)`:

| Call | Seconds | XLA compile messages |
| --- | ---: | ---: |
| 1 | 139.2 | 1,295 (one `jit(jaxpr_call)` executable for the update, plus about 640 op-by-op eager executables from `model.reset_episode(learner)` outside the timed region) |
| 2 | 20.6 | 0 |
| 3 | 20.7 | 0 |
| 4 | 20.7 | 0 |

So the third call is neither near-free nor 330 s: it is run time, 10.3 s per event at the
pinned dt, and it moves with the levers (2 -> 8 events: 26.7 -> 105.8 s; 160 -> 20 substeps:
105.8 -> 11.6 s for 8 events). `learner.compile_graph` (6.8-9.9 s) is braintrace's ETP
compiler (`graph_executor.compile_graph`: jaxpr trace of the model plus eligibility-graph
analysis on the host); it builds no XLA executable. The XLA compile of the learner update is
the first call of the jitted update: 321 s cold (347.8 - 26.7) in the `pinned` arm, 118 s
in the `pinned-recompile` arm (139.2 - 20.6), where the persistent cache at
`/workspace/jax-cache` served the executables compiled by the earlier process (the forward
compile also fell from 41.3 s to 16.8 s). Warm run time also differed between the two
processes (26.7 s versus 20.6 s per 2-event update): the `pinned` arm ran while a CPU job
(`h01_c3_kept_partner_graph.py`) shared the box and the run is launch-bound, so host CPU
contention shows up in GPU-side wall time.

What one evolve arm pays before its first event, from these stages: build 20 s + init 23 s +
forward compile 41 s + learner compile ~320 s (cold; ~120 s cache hit) + `score_queries`
compile ~60 s (dt 0.005 arm: 122.0 - 62.2): about 7-8 minutes per rebuilt runtime. Every arm
(`train_parent`, `run_candidate`) rebuilds from a checkpoint, and a mutated topology changes
array shapes, so a grow/prune arm is a cache miss. Per-mutation recompile was not timed
separately (the coordinator's item 4 was conditional on the third call being near-free,
which it is not).

## 3. What one stage consumes (from `example21_evolve.py`, verified on the corpus)

Advancing events counted with `encode_episode` over the real manifest (`corpus` subcommand;
[h01-evolve-event-profile/corpus.json](h01-evolve-event-profile/corpus.json)):

| Quantity | Value |
| --- | ---: |
| training tasks / queries | 400 / 416 |
| advancing events per query (min / mean / max) | 193 / 279.2 / 705 |
| whole-corpus scoring pass (`initialize`, `train_parent`), advancing events | 116,128 (plus 177,152 padding events skipped by `cond`) |
| one 128-update block (`build_update_schedule` from cursor 0), advancing events | 35,648 (mean 278.5 per update, 122 distinct tasks) |
| `--screen-tasks 2` subset (007bbfb7, 00d62c1b), advancing events | 770 (2 queries) |
| stacked float64 events for one `score_queries` call | 416 x 705 x 441 x 8 B = 0.96 GiB |

`--screen-tasks 2` reaches only `run_candidate` (`task_ids=context.score_task_ids`);
`initialize()` and `train_parent()` call `_write_runtime` with no task subset and score all
400 tasks. The order of the first round is: `initialize` (build, score 416 queries, save) ->
`train` stage (`train_parent`: rebuild from checkpoint, 128 updates, score 416 queries) ->
`edge`/`neuron` stages (each arm: rebuild, mutate, 128 updates, score 2 tasks).

Multiplied out (JSON `arithmetic`; learner s/event is the real 705-event update where
measured, else the truncated-update slope):

| Setting | fixed (build+init+compiles) | corpus pass, forward | one update (278.5 ev) | one block (128) | blocks per hour, steady | blocks under a 3600 s cap after fixed + corpus pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pinned 0.000625 / 160 | 497 s | 157,555 s = 43.8 h | 3,671 s | 469,854 s = 130.5 h | 0.0077 | 0 (the corpus pass alone is 44x the cap) |
| dt 0.005 / 20 | 418 s | 20,864 s = 5.8 h | 444 s | 56,876 s = 15.8 h | 0.063 | 0 (the corpus pass alone is 5.8x the cap) |

Screen-subset scoring (770 events) costs 1,045 s pinned, 138 s at dt 0.005.

Could the 60 minute run have reached its first arm: no. `run.err` holds only the two
`compile_graph` warnings, which fire in the last statement of `H01Session.build`, and the
first stdout line comes from `ConsoleProgressReporter` in `_run_parent_training`, after
`initialize()` returns. Build, init, model and the two compiles took about 100-150 s of the
cap (the 93 percent GPU utilisation at 6 min in the record is the scoring loop); the
remaining ~3,450 s covered about 2,500 of the 116,128 advancing events of the initial scoring
pass (about 9 of 416 queries). The first arm (`train_parent`) is 128 updates further, 130 h
away at the pinned dt.

## 4. Where a steady-state event goes (XLA profile)

Profiled with `jax.profiler` and read back with `jax.profiler.ProfileData`; each GPU stream
event is classified by the Python source that created its HLO op (stack frames from
`Compiled.as_text()`, walked leaf to root until a known file: channel laws, DHS axial solve,
implicit calcium, delivery, encoder/readout, cell runtime bookkeeping, loop control).

Batch structure (question 4): `build_network` creates one `braincell` population per cell
(`pop_size=(1,)`) and `H01NetworkStep.update` loops over `self.cells` in Python, so the 17
cells are 17 separate kernel sets inside one XLA program, not one fused population. The
single-substep HLO has 136 while loops = 17 cells x 8 DHS level scans per cell (+1 outer
loop in the event program); the `cell-runtime` class shows exactly 17 launches per substep,
one per cell.

Per cable substep (dt 0.005 arm, single substep jitted, 5 traced executions; default XLA
flags, so sequences of kernels run as CUDA-graph command buffers and appear as one stream
event each):

| Class | GPU busy ms per substep | share of busy | stream launches per substep |
| --- | ---: | ---: | ---: |
| command buffers and unclassified (`other`) | 6.27 | 75.2 % | 3,710 |
| axial solve (DHS level scans, scatter-adds) | 1.06 | 12.7 % | 701 |
| loop control (while cond, select, carries) | 0.50 | 6.0 % | 303 |
| implicit calcium | 0.37 | 4.4 % | 255 |
| memcpy D2D | 0.12 | 1.4 % | 118 |
| cell runtime bookkeeping | 0.02 | 0.3 % | 17 |
| total | 8.34 | | 5,104 |

GPU-busy 8.3 ms per substep against 9.0 ms wall per substep inside the event loop
(0.180 s / 20): the substep loop keeps the GPU about 92 percent busy, and the launches are
what fill it: 5,104 stream events per substep is 1.6 us of GPU time per launch. The
`command_buffer_N` events cannot be attributed to a source class from the HLO text (they are
formed after HLO scheduling); the arm that disables command buffers
(`--xla_gpu_enable_command_buffer=`) to attribute every kernel was queued and not run (see
"What was not run"). The standalone single-substep call outside the loop is 24.7 ms
(idle 79 percent of its span): that is host dispatch of a 5,000-launch program per call and
is not the in-loop cost.

Whole-event trace on the pinned arm (3 events; CUPTI dropped part of one activity buffer, so
the sums are a lower bound): 690,363 launches per event = 4,315 per substep; GPU busy
1.05 s per event, axial solve 0.165 s (15.6 %), implicit calcium 0.046 s (4.3 %), loop control
0.027 s (2.6 %), `other` (command buffers) 0.82 s (77.5 %); span under the profiler 2.5 s
against 1.36 s unprofiled.

Throughput (question 3): 107,537 compartments x 160 substeps / 1.357 s = 12.7 M
compartment-updates per second pinned; x 20 / 0.180 s = 12.0 M at dt 0.005. XLA's
`cost_analysis` reports 124.6 MFLOP per event (loop bodies counted once, so this is a lower
bound): 0.09 GFLOP/s pinned = 7e-5 of the 4090's 1.29 TFLOP/s fp64 peak and 1e-6 of its
82.6 TFLOP/s fp32 peak. Even at an assumed 1,000 flop per compartment-update the pinned rate
would be 12.7 GFLOP/s, about 1 percent of fp64 peak: the arithmetic units are idle; the
substep is bounded by kernel launches and dependent tiny kernels (17 cells x 8 sequential
level scans of a few kernels each).

## 5. The 1 mV contract: what the 8x buys and costs

Over the 21 profiled events (2.1 ms of simulated time, same 385-event ARC episode, all 17
cells subthreshold between -86.4 and -79.3 mV) the soma voltage difference between
dt 0.005 / 20 substeps and dt 0.000625 / 160 substeps is 0.0013 mV maximum over all cells and
events (per-cell maxima 0.0001-0.0013 mV, final-event differences 4e-6 to 7e-4 mV). This
window contains no spike; the keep-test spikes appeared at 11.6-38 ms (116-380 events), so
this number does not settle the dt question across a spike. The 400-event single-cell
windows at 0.000625, 0.0025 and 0.005 ms that would cover a spike were queued and not run (see below); the
keep tests' own dt-half repeats (0.005 -> 0.0025) reproduced every spike count, and the
kept-refinement gate (0.000625 -> 0.0003125, 10 ms) held to 0.0014 mV.

Cost side: the 8x finer clock multiplies the forward event by 7.5x (0.180 -> 1.357 s), the
learner event by 8.3x (1.60 -> 13.2 s), and turns the corpus pass from 5.8 h into 43.8 h and a
block from 15.8 h into 130 h.

## What was not run

The serial queue (dt 0.005 arm, then command-buffer-free profile, pinned single-substep
profile, one-cell 400-event windows at 0.000625 / 0.0025 / 0.005 ms, `max_cv_length_um` 20
and 40, precision 32, no checkpoint rematerialisation) was killed at the coordinator's request
after the pinned and dt 0.005 arms, on the reading that the fixed cost swamped every lever;
the dt 0.005 arm was allowed to finish its last two stages (real update warm, score query),
and the `pinned-recompile` arm was run to answer the compile-versus-run question
([queue-log.txt](h01-evolve-event-profile/queue-log.txt)). The compile-versus-run split in
section 2 shows the levers do reach the timed region, so the remaining arms carry the per-arm
fixed cost of section 2 but no false null; they remain unmeasured here. The pinned real
705-event `update_episode` was not run (estimated 1.4 h from the slope).

## Deviations and caveats

- The `pinned` arm's 3-event whole-event trace dropped CUPTI activity records (one warning);
  its class shares are lower bounds and the launch count is approximate. The dt 0.005
  per-substep trace dropped nothing.
- The pinned arm ran while a CPU job shared the box (`h01_c3_kept_partner_graph.py`,
  one core); the later `pinned-recompile` process measured 2-event warm updates 23 percent
  faster (20.6 s versus 26.7 s) and the forward event 4 percent slower (1.413 s versus
  1.357 s). Treat pinned numbers as +-25 percent.
- The persistent XLA cache on the box means "compile" times in a second process on an
  identical program are cache loads (118 s learner, 16.8 s forward), not cold compiles.
- The learner per-event slope at dt 0.005 (1.34 s, truncated synthetic episodes with a
  squared-logit loss) is 16 percent below the real 705-event `update_episode` (1.595 s per
  advancing event, ARC request loss, 320 padding events under `cond`); the block arithmetic
  uses the real number where it exists.
- `cost_analysis` FLOPs undercount loops; the compartment-updates-per-second figure is the
  measured throughput.
