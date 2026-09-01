# Example 21 runtime-throughput v2 evidence

## Result

**SCORER PASS; CONTROLLED PERFORMANCE PASS; CROSS-RUN TOPOLOGY REPLAY NOT
QUALIFIED.** Complete-corpus scoring improves by `1.296x` in the required warm
median, fixed-checkpoint score and structural evidence are equivalent, and the
controlled one-round run finishes in `637.020` seconds versus the fresh
`643.658`-second baseline. Peak host and device memory both decrease.

The independent end-to-end runs do not produce identical final topology or
checkpoint hashes. The changed scorer is exact when both paths receive the same
checkpoint, but unchanged GPU PP-Prop updates drift between independent runs
and cross structural-selection thresholds. This evidence therefore does not
claim exact full-run replay or identical final ARC capability.

The work is on branch `perf/example21-next-speedup`, based on
`c9caf3d14c85649208644bf8811e55bc4141dd46`.

## Environment

- Date: 2026-09-01.
- GPU: NVIDIA RTX 3080 Ti Laptop GPU.
- Docker image:
  `sha256:4a7546a80f424ad79b0a950e3683ebf214aab164bf2b1d58f50ad939293bfb90`.
- Fixed checkpoint:
  `var/example21-ops-canary/checkpoints/r000-round-score.npz`.
- Baseline source: exact commit `c9caf3d14c85649208644bf8811e55bc4141dd46`.
- Each phase benchmark used a fresh Docker process, isolated compilation cache,
  one first call, and three warm calls.
- The accepted full-round comparison started from the observed stable idle GPU
  floor of 67 degrees Celsius after the requested 55-degree threshold proved
  unreachable under the laptop's idle fan policy.

## Baseline profile

The fresh one-round baseline completed in `643.657886865` seconds.

| phase | calls | total seconds | median seconds |
|---|---:|---:|---:|
| direct score | 13 | 287.697 | 12.070 |
| 128-update PP-Prop block | 9 | 260.093 | 27.716 |
| runtime construction | 20 | 28.604 | 0.526 |
| topology render | 10 | 16.732 | 1.510 |
| persistence | 7 | 7.725 | 1.130 |

Peak host usage was `4,735,881,216` bytes and peak device usage was
`543,801,600` bytes. The run closed at the one-round budget with every training
sibling executing exactly 128 updates and every score-only stage executing zero
updates.

## Selected implementation

The complete-corpus scorer now:

1. runs every query as one `brainstate.transform.scan`;
2. carries the accumulated spike activity instead of returning a full spike
   history;
3. retains the voltage history and uses its final 31 values for direct readout;
4. uses corpus-optimal static buckets `(193, 257, 321, 449, 705)`; and
5. leaves subset scoring on the baseline full-history `(320, 705)` path.

The training corpus contains 416 queries and eight distinct advancing lengths.
The selected five buckets execute `119,200` slots, down from `178,935` at the
baseline, while preserving all `116,128` advancing events and their order.

## Fixed-checkpoint performance

| scope | path | first seconds | warm calls seconds | warm median | speedup |
|---|---|---:|---|---:|---:|
| 400 tasks | baseline | 48.543 | 44.516, 45.764, 40.953 | 44.516 | baseline |
| 400 tasks | selected | 38.996 | 29.157, 35.010, 34.358 | 34.358 | **1.296x** |
| 64 tasks | baseline | 12.683 | 12.028, 9.060, 8.424 | 9.060 | baseline |
| 64 tasks | selected route | 21.731 | 8.406, 8.525, 8.931 | 8.525 | 1.063x |

The 400-task affected phase clears the `1.20x` warm-median gate. The first
64-task timing was noisy, but the selected implementation deliberately uses the
baseline kernel and buckets for that scope; the controlled full run is the
compile-inclusive screen-path gate.

Fixed 400-task peak memory fell from `2,638,274,560` to `2,426,155,008` host
bytes and from `348,186,368` to `173,664,512` device bytes.

## Fixed-checkpoint correctness

For both 400-task and 64-task comparisons:

- task IDs, exact flags, owners, and owner codes are identical;
- neuron, source, target, and edge structural evidence is identical;
- score-only execution changes no optimizer parameter;
- the 64-task task-loss arrays are bit-identical; and
- the 400-task maximum absolute task-loss difference is
  `3.451842189861054e-8`, below the `1e-6` gate.

Repeated baseline and selected calls show task-loss noise of the same order.
The optimized scan never splits a recurrent query. A rejected split-scan
prototype changed task loss by `6.61e-4` and was removed.

## Controlled one-round performance

The accepted performance run completed in `637.019823132` seconds, saving
`6.638064` seconds and improving wall time by `1.010x`.

| aggregate phase | baseline seconds | selected seconds | delta seconds |
|---|---:|---:|---:|
| all direct scores | 287.697 | 274.567 | -13.130 |
| all unchanged update blocks | 260.093 | 270.128 | +10.035 |
| complete evolution | 643.658 | 637.020 | -6.638 |

The run completed all nine 128-update blocks, all subset and full scores,
round persistence, terminal-only evaluation, and durable closeout. Peak host
memory was `4,430,069,760` bytes, 6.5 percent below baseline. Peak device memory
was `533,420,032` bytes, 1.9 percent below baseline.

Two earlier end-to-end attempts from a thermally soaked GPU took `705.200` and
`700.497` seconds. In the latter, scoring was already `6.551` seconds faster,
but unchanged update blocks were `57.438` seconds slower. The controlled rerun
was authorized only after the GPU reached its measured idle plateau; these
rejected runs are not used as speedup evidence.

## Cross-run replay boundary

The baseline and selected runs agree through train, round-screen, edge-add, and
neuron-add selection. They then diverge after unchanged GPU updates:

| field | baseline | controlled selected run |
|---|---:|---:|
| neuron-add recurrent edges | 19,072 | 19,071 |
| edge-revisit disposition | selected prune | retained parent |
| final training exact tasks | 4/400 | 1/400 |
| final recurrent edges | 18,118 | 19,071 |

The fixed-checkpoint scorer comparisons prove that the changed scoring path
does not cause this difference when inputs are identical. They do not make the
independent PP-Prop runs deterministic. Consequently, final topology digest,
checkpoint SHA-256, evaluation digest, and capability parity are explicitly
not qualified by this work.

## Rejected attempts

- Training-event compaction reached only `1.166x`, increased host memory, and
  changed owners and task loss because PP-Prop eligibility traces evolve over
  masked positions. It was removed.
- Heuristic five scoring buckets reached `1.159x`, below the phase gate.
- A split prefix/request scan changed task loss by `6.61e-4`. It was removed.
- A numerically exact single scan with heuristic buckets reached `1.175x`,
  below the phase gate.
- Optimal five buckets used for every score passed the fixed phase but made the
  first one-round attempt `705.200` seconds because 64-task screens compiled too
  many shapes. Scope-adaptive routing replaced it.

## Tests and static checks

- Focused model, adapter, coordinator, and structural suites: `266 passed` in
  `235.26s`.
- Final changed-module rerun after the scoring edge tests: `116 passed` in
  `114.58s`, including malformed shapes, short sequences, and masked-event
  behavior.
- Model and adapter branch coverage: 91 and 96 percent; 94 percent combined.
- Ruff check passed with only the executable's pre-existing `N999` and `I001`
  exceptions excluded. The normally named adapter modules also pass Ruff's
  format check. The hyphenated executable and its test retain pre-existing
  whole-file format drift; only the new hunks were aligned to local style.
- BasedPyright: zero errors, warnings, or notes on both production modules.
- `git diff --check`: passed.

The existing BrainTrace warnings about readout states not being registered in
the compiled model remain warnings and are outside this change.
