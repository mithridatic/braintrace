# H01 population: 12-cell staged build (SP8 stage 2)

Generated 2026-09-07T16:55:13 from [h01-population-build-12.json](h01-population-build-12.json). Evidence topology (2 construction-ready contacts, 4 incident cells) plus the 8 smallest isolated cells by largest-component node count, `--include-isolated --cells 12`. **Every number is measured under load** (other jobs ran on the machine throughout). Cost and construction qualification only; no physiology claim. Predictions were registered in the builder spec addenda (2026-09-07) before each launch.

## Measured

| Run | Status | Construction s | Compartments | init_state s | compile+run s | Wall s | Peak RSS MB (tree) | Traces finite | Conductance probes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| build-only (`--build`, control ei) | completed | 883.3 | 84,097 | - | - | 900.3 | 690 | - | - |
| 1 ms, dt 0.005, control ei, attempt 1 | killed (construction not complete by 900 s) | untested (9 of 12 cells registered at 880 s) | - | - | - | 901.1 | 690 | - | - |
| 1 ms, dt 0.005, control ei, attempt 2 | killed (silence 600 s) | 873.9 | 84,097 | untested, > 600 (killed 635.0 s wall after init began) | untested | 1,509.4 | 1,075 (inside init_state) | untested | untested |
| 1 ms, dt 0.005, control disconnected | untested (not launched) | - | - | - | - | - | - | untested | untested |

- Compartments: 84,097 = 1.112x the 4-cell 75,605 (prediction ~1.12x, ~85,000: held).
- Peak RSS is the resident-set sum over the interpreter process tree (venv launcher ~6 MB + the real interpreter). Rejection at 6 GB: not triggered on any observed reading; the run-phase peak is untested.
- The first attempt's failure (`Invalid electrical region interval.` on `2451406889`) did not recur: all 12 cells of `cell_order` built in all three launches that reached that cell (prediction 1: held).
- Construction: 883.3 s (build-only) and 873.9 s (ei attempt 2) reached `Construction complete`; attempt 1 was ~55 s behind that pace at every stage and was killed at 900 s with 3 small cells left, so construction under this load varies by at least +-6 %. Cell building took ~460-520 s of that and discretization/registration ~360-420 s (the 33,965-compartment cell `3955003482` alone ~145-190 s).
- Run phase: `network.init_state()` ran with no log line for 600 s and the silence watchdog fired (635.0 s wall after the init line, RSS 1,075 MB). `init_state` at 12 cells under load therefore exceeds 600 s, against 212-275 s for SP1's whole init + compile + 200 steps at 4 cells alone. Load, a superlinear init cost, or both: untested here. The example prints nothing inside `init_state`, so the 10-minute silence rule cannot separate a live long init from a hang; a per-cell progress callback in `init_state` (or timing `init_state` in a build-only mode) is the instrument change needed before the run phase can be measured under this rule.
- Disconnected arm: not launched: the ei arm was killed inside init_state by the 600 s silence rule; the disconnected arm runs the identical init_state path (only projections differ) and would be killed the same way, yielding no information. Untested, not negative.
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

Under-load 12-cell numbers scale to under-load projections; alone-machine costs would be lower by an unmeasured factor. The 104-cell construction projection (~2.5 h) and init lower bound (> 1.7 h) exceed the no-hour-plus-runs policy as single jobs; the 40-cell stage is not approved by this check because the run phase is untested.

## Record of the first attempt (2026-09-07T15:40, failed; kept verbatim in the JSON under `first_attempt_2026-09-07T15:40`)

Status **failed** at 332.6 s wall, peak RSS 649 MB before discretization. 12-cell construction did not complete: ValueError('Invalid electrical region interval.') from h01_ei_cell._validate_regions while building isolated cell 2451406889 (8th of 12 in cell_order) at 324.2 s; the network and build JSON were never written, so compartments are unrecorded (untested, not measured).

Root cause: Floating-point rounding: on branch 162 of 2451406889 component 0 the soma region evaluates to (0.9902931374228255, 1.0000000000000002); the validator demanded hi <= 1 exactly while its own coverage check already tolerates 1e-9. Diagnosed by a read-only region evaluation of that component after the failure; fixed in the same commit by snapping endpoints within 1e-9 of 0/1 (no physiology change). Not re-run: this task allows one construction check.
