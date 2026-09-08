# H01 population: 12-cell staged build (SP8 stage 2)

Generated 2026-09-07T16:55:13 from [h01-population-build-12.json](h01-population-build-12.json). Evidence topology (2 construction-ready contacts, 4 incident cells) plus the 8 smallest isolated cells by largest-component node count, `--include-isolated --cells 12`. **Every number is measured under load** (other jobs ran on the machine throughout). Cost and construction qualification only; no physiology claim. Predictions were registered in the builder spec addenda (2026-09-07) before each launch. **Attempt 3 (below, 18:26, machine quiet by the stated definition) supersedes the verdict of the under-load sections: the 40-cell stage was approved by that cost check on 2026-09-07 18:26.** The 40-cell construction was then attempted and failed at 82.6 s (`ValueError: from_points() requires at least two points`, cell 5805562981; 7 of 104 largest components fail to load, 2 inside the 40-cell prefix); stages 2-5 were stopped by the user; no 40-cell number exists and the deliverable is the measured 12-cell network (see [h01-population-build-40.md](h01-population-build-40.md); `h01-population-build-40.json#/runs/build-only/status`, `#/status`, `#/population_deliverable`).

## Measured

| Run | Status | Construction s | Compartments | init_state s | compile+run s | Wall s | Peak RSS MB (tree) | Traces finite | Conductance probes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| build-only (`--build`, control ei) | completed | 883.3 | 84,097 | - | - | 900.3 | 690 | - | - |
| 1 ms, dt 0.005, control ei, attempt 1 | killed (construction not complete by 900 s) | untested (8 of 12 cells registered by 880 s; the 9th, 678539249, had just started discretization) | - | - | - | 901.1 | 690 | - | - |
| 1 ms, dt 0.005, control ei, attempt 2 | killed (silence 600 s) | 873.9 | 84,097 | untested, > 600 (killed 635.0 s wall after init began) | untested | 1,509.4 | 1,075 (inside init_state) | untested | untested |
| 1 ms, dt 0.005, control disconnected | not launched in attempt 2; run in attempt 3 below (completed: construction 270.5 s, init_state 201.6 s, compile+run 29.3 s, finite traces, conductance probes zero) | - | - | - | - | - | - | see attempt 3 | see attempt 3 |

- Compartments: 84,097 = 1.112x the 4-cell 75,605 (prediction ~1.12x, ~85,000: held).
- Peak RSS is the resident-set sum over the interpreter process tree (venv launcher ~6 MB + the real interpreter). Rejection at 6 GB: not triggered on any observed reading; the run-phase peak is untested.
- The first attempt's failure (`Invalid electrical region interval.` on `2451406889`) did not recur: all 12 cells of `cell_order` built in all three launches that reached that cell (prediction 1: held).
- Construction: 883.3 s (build-only) and 873.9 s (ei attempt 2) reached `Construction complete`; attempt 1 lagged build-only by 86 s at the first cell build, 55 s at the first discretization and about 21-22 s at the late registrations, and was killed at 900 s with 3 small cells left, so construction under this load varies by at least +-6 %. Cell building took ~460-520 s of that and discretization/registration ~360-420 s (the 33,965-compartment cell `3955003482` alone ~145-190 s).
- Run phase: `network.init_state()` ran with no log line for 600 s and the silence watchdog fired (635.0 s wall after the init line, RSS 1,075 MB). `init_state` at 12 cells under load therefore exceeds 600 s, against 212-275 s for SP1's whole init + compile + 200 steps at 4 cells alone. Load, a superlinear init cost, or both: untested here. The example prints nothing inside `init_state`, so the 10-minute silence rule cannot separate a live long init from a hang; a per-cell progress callback in `init_state` (or timing `init_state` in a build-only mode) is the instrument change needed before the run phase can be measured under this rule.
- Disconnected arm: not launched in attempt 2: the ei arm was killed inside init_state by the 600 s silence rule; the disconnected arm runs the identical init_state path (only projections differ) and would be killed the same way, yielding no information. Run in attempt 3 below (completed: construction 270.5 s, init_state 201.6 s, compile+run 29.3 s, finite traces, conductance probes zero; `h01-population-build-12.json#/quiet_machine_attempt_2026-09-07T17/runs/run_1ms_disconnected`).
- Limits: The 900 s abort (spec) applies to the construction check (log line 'Construction complete'), which passed at 883 s under load. The first 1 ms ei run was killed by that same 900 s limit with 3 of 12 cells left to discretize (untested, not negative); the retry used a construction limit of 1,350 s (1.5x the measured point) registered in the spec addendum before launch, with the 600 s silence kill and a 2,700 s wall cap.

## Against the SP1 linear prediction (derived from 4 cells at 1.123x nodes)

| Quantity | SP1-derived 12-cell prediction | Measured (under load) | Inside [0.5x, 2x] |
| --- | ---: | ---: | --- |
| Construction s | 275 | 883.3 / 873.9 | False (3.2x; load-confounded: the 4 incident cells alone took 318 s under load in the first attempt) |
| Init + compile + 200 steps s | 276 (a = 245 s scaled; SP1 1 ms runs 212-275 s) | untested; init_state alone > 600 | False on the lower bound alone (600 / 276 = 2.17x) |
| Peak RSS GB (run) | 1.48 | untested; 1.05 at the init_state kill (lower bound) | untested (lower bound is inside) |
| Compartments | 84925 | 84,097 | True |

The 12-cell point is **not inside 2x the SP1 linear prediction**: construction is 3.2x and the init_state lower bound alone is 2.17x. Both measurements are under load and SP1 was alone, so the excess is not attributable to the cell count without an alone-machine repeat.

## Derived projections from the measured 12-cell point (proportional rule; not measured)

| Stage | Largest-component nodes | Ratio to 4 cells | Construction s | init_state s (lower bound) | compile+run (1 ms) s | Peak RSS GB build-only | Peak RSS GB at init_state kill (lower bound of run peak) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 12 (measured, under load) | 276,064 | 1.123 | 883 | > 600 | untested | 0.67 | 1.05 |
| 40 (derived) | 537,152 | 2.186 | 1,719 | > 1,167 | untested | 1.31 | 2.04 |
| 104 (derived) | 2,804,549 | 11.411 | 8,973 | > 6,095 | untested | 6.85 | 10.66 |

Under-load 12-cell numbers scale to under-load projections; alone-machine costs would be lower by an unmeasured factor. The 104-cell construction projection (~2.5 h) and init lower bound (> 1.7 h) exceed the no-hour-plus-runs policy as single jobs; the 40-cell stage is not approved by this check because the run phase is untested (superseded by attempt 3's approval at 18:26, after which the 40-cell build attempt failed; see the header).

## Record of the first attempt (2026-09-07T15:40, failed; kept verbatim in the JSON under `first_attempt_2026-09-07T15:40`)

Status **failed** at 332.6 s wall, peak RSS 649 MB before discretization. 12-cell construction did not complete: ValueError('Invalid electrical region interval.') from h01_ei_cell._validate_regions while building isolated cell 2451406889 (8th of 12 in cell_order) at 324.2 s; the network and build JSON were never written, so compartments are unrecorded (untested, not measured).

Root cause: Floating-point rounding: on branch 162 of 2451406889 component 0 the soma region evaluates to (0.9902931374228255, 1.0000000000000002); the validator demanded hi <= 1 exactly while its own coverage check already tolerates 1e-9. Diagnosed by a read-only region evaluation of that component after the failure; fixed in the same commit by snapping endpoints within 1e-9 of 0/1 (no physiology change). Not re-run: this task allows one construction check.

## Attempt 3 (2026-09-07T18:26, **measured, machine quiet except the C3 rescan**): instrumented init_state and the run phase

Predictions registered in the builder-spec addendum of 2026-09-07 17:05 before launch. docker ps shows only synapse; no python process other than the C3 rescan worker (h01_c3_edge_list.py) above 300 MB; checked twice 60 s apart before launch and re-checked before each stage. Instrumentation: per-cell init_state progress lines and a 60 s heartbeat (elapsed s, RSS) from braintrace/datasets/h01_network_init.py; --init-only mode. Kill rules: 600 s without a progress or heartbeat line, 1,500 s init_state abort, 3,600 s wall cap (no-hour-plus policy); no construction abort.

| Run | Status | Construction s | Compartments | init_state s | compile+run s | Wall s | Peak RSS MB (interpreter peak_wset / tree) | Traces finite | Conductance probes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| init-only, control ei | completed  | 744.2 | 84,097 | 635.4 | untested | 1,393.5 | 1,290 / 1,303 | untested | untested |
| 1 ms, dt 0.005, control ei | completed  | 338.5 | 84,097 | 265.0 | 29.8 | 643.6 | 2,000 / 1,997 | True | all zero |
| 1 ms, dt 0.005, control disconnected | completed  | 270.5 | 84,097 | 201.6 | 29.3 | 508.5 | 1,988 / 1,986 | True | all zero |

- Construction: 744.2, 338.5, 270.5 s against the registered 266-311 s (SP1 alone 240-280 s x 1.11). Under load the same build took 883 s.
- init_state: init_only_ei 635.4, run_1ms_ei 265.0, run_1ms_disconnected 201.6 s against the registered 245-736 s (1x-3x the 4-cell a = 245 s); heaviest cells: `3955003482` 275.8 s, `5584343344` 98.4 s, `4188575291` 92.2 s. Under load it exceeded 600 s (killed).
- compile + 200 steps: ei 29.8 s, disconnected 29.3 s; init + compile + run (ei) 294.8 s against 276 s derived.
- Peak RSS of the real interpreter (psutil peak_wset, read inside the process): 1,290, 2,000, 1,988 MB; watcher tree sums 1,303, 1,997, 1,986 MB. 6 GB rejection: not triggered.
- Finiteness: ei True, disconnected True. Disconnected conductance probes identically zero: True; ei probes nonzero: False.
- Untested: ei conductance probes nonzero (not discriminable: 1 ms at zero input, soma pulses start at 2 ms, no presynaptic spike).
- Load: Quiet by the stated definition throughout (docker: only synapse; no python above 300 MB besides the C3 rescan worker, which was not observed running). CPU was not idle: during the init-only stage comsolbatch.exe (~2.6 cores), bdservicehost.exe (~1.9 cores), SearchIndexer (~1 core) and two Codex sandbox setups were active (87 % of 20 logical cores at 17:56); by the ei run (18:07-18:17) only this interpreter and bdservicehost (~1 core) were active (25 % CPU). The init-only numbers are therefore CPU-confounded by COMSOL; the two 1 ms runs are the clean points.

### Against the SP1 linear prediction (quiet machine)

| Quantity | SP1-derived | Measured | Inside [0.5x, 2x] |
| --- | ---: | ---: | --- |
| Construction s | 275 | 744.2, 338.5, 270.5 (test uses the two 1 ms runs) | True |
| Init + compile + 200 steps s | 276 | 294.8 | True |
| Peak RSS GB | 1.48 | 1.95 | True |

The quiet-machine 12-cell point is **inside 2x the SP1 linear prediction**.

### Derived projections from the quiet-machine 12-cell point (proportional rule; derived, not measured)

| Stage | Nodes | Ratio to 4 | Construction s | init_state s | compile+run (1 ms) s | Peak RSS GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 (derived) | 537,152 | 2.186 | 593 | 454 | 58 | 3.80 |
| 104 (derived) | 2,804,549 | 11.411 | 3,094 | 2,370 | 302 | 19.84 |

**40-cell stage: approved by this cost check (2026-09-07 18:26; `h01-population-build-12.json#/quiet_machine_attempt_2026-09-07T17/verdict_40_cell_stage`).** The approved build-only launch then failed at 82.6 s on cell 5805562981 (one-point SWC branch; 7 of 104 components fail to load, 2 inside the 40-cell prefix); stages 2-5 stopped by the user, untested; the deliverable is this 12-cell network (`h01-population-build-40.json#/status`, `#/population_deliverable`).
