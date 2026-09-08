# SP1. H01 verified-network throughput benchmark

Status: spec written before code (AGENTS.md rule 7). Parent: [population programme](2026-09-07-h01-population-programme.md) SP1.
Branch `feat/h01-network`, worktree `.worktrees/h01-network`. No physiology claim is made here; the
network is a measurement instrument whose cost is being qualified.

## Question

How much wall time does the 4-cell verified H01 network (`examples/h01_verified_network.py`,
topology `docs/evidence/h01-verified-network.json`, solver `h01_staggered_scan`, dt 0.005 ms)
cost per simulated millisecond, once construction and init+compile are separated from the
marginal step cost? SP8 must be sized from these measured numbers.

## Anchor (measured, 2026-09-06)

One run at dt 0.005 ms, duration 0.005 ms (one step): construction 163.3 s,
`initialization_and_run_seconds` 157.0 s (`docs/evidence/h01-verified-network-build.json`).
Peak DHS memory about 214 MB. No per-step cost has ever been measured.

## Design

`docs/evidence/h01_network_throughput.py` (+ `h01_network_throughput_test.py`):

- `run`: launches `examples.h01_verified_network --topology ... --build --dt-ms 0.005
  --duration-ms D --solver S --output .cache/h01/bench-<label>/network` as a child process,
  samples the child tree's RSS with `psutil` once per second, kills the child when its log is
  silent for 600 s (killed = untested, never negative), and writes `run.json` (wall clocks,
  peak RSS, exit status, parsed `construction_seconds` and `initialization_and_run_seconds`
  from `network-build.json`, steps = round(D / dt)).
- `report`: reads every `run.json`, fits T(steps) = a + b*steps by least squares over the
  completed `h01_staggered_scan` runs (T = `initialization_and_run_seconds`), reports a
  (init + compile), b (s per step), 200*b (s per simulated ms at dt 0.005), the residual of
  each point, the two-repeat decision limit on b = range(b estimates from the two 1 ms
  repeats, each against the 0.05 ms point) x 2.95 and the matching limit in seconds, and
  writes `docs/evidence/h01-network-throughput.json` + `h01-network-throughput.md`.
- The real run is not unit-tested; parsing, step count, fit, limit, prediction, cap,
  watchdog rule, RSS sampling (stub process) and report rendering are.

Each run is its own detached PowerShell job (`Start-Process -NoNewWindow -PassThru
-RedirectStandardOutput`) with the machine otherwise idle; every wall clock is recorded.

## Run plan (cap 5 runs, in order)

1. D = 0.05 ms (10 steps).
2. D = 1 ms (200 steps).
3. D = 1 ms repeat.
4. D = 10 ms (2,000 steps) only if a + 2000*b predicted from runs 1-3 is under 900 s;
   otherwise stop and report the estimate.
5. `--solver staggered` control at 1 ms only if its predicted cost fits 900 s (it has no
   prediction of its own; the h01_staggered_scan 1 ms time is used as the estimate).

## Pre-registered prediction

At 0.05 ms, init + compile dominates: total `initialization_and_run_seconds` lies within
157 s plus or minus the two-repeat decision limit measured at 1 ms. No prediction is made
for b (the per-step cost is unmeasured).

## Rejection

The three durations (0.05, 1, 10 ms) are not linear in step count within the 1 ms
two-repeat decision limit (any point's residual from the least-squares line exceeds the
limit in seconds). If rejected, SP8 sizes from the 10 ms point directly.

## Unchanged

Topology, dt 0.005 ms, `--max-cv-um 10`, `--current-na 0`, precision 64, the four cells and
two projections, the `.cache/h01` archive and annotations (read from the sibling
`h01-braincell` worktree, passed explicitly).

## Outputs

`docs/evidence/h01-network-throughput.json`, `docs/evidence/h01-network-throughput.md`,
raw run directories under `.cache/h01/bench-*` (gitignored; traces are per-cell output
voltage only, small). Same commit appends a row to `docs/h01-causal-model.md` under
"Measurement function qualification" (SP0 causal-model rule).
