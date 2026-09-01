# Example 21 scoring-throughput qualification

## Result

**PASS.** The compact `(320, 705)` scorer clears both `1.20x` promotion
thresholds, preserves the fixed-checkpoint correctness and structural-selection
gates, reduces observed peak memory, and completes one full resumable GPU
evolution round through terminal-only evaluation.

The qualified source is branch `perf/example21-scoring-throughput`, based on
`25f84bc9f27129754965b447dbc6b0a1efe08378`. The benchmark mounted that exact
worktree read-only and set `PYTHONPATH=/workspace`.

## Environment

- Date: 2026-08-31.
- GPU: NVIDIA RTX 3080 Ti Laptop GPU.
- Docker image:
  `sha256:4a7546a80f424ad79b0a950e3683ebf214aab164bf2b1d58f50ad939293bfb90`.
- Qualification checkpoint:
  `var/example21-ops-canary/checkpoints/r000-round-score.npz`.
- Checkpoint bytes: `6,553,021`.
- Checkpoint SHA-256:
  `e46cedde4e079c86e47147d94c9b32dbf90b7910d2b514680c4bb924c31afed4`.
- Each timing used a fresh Docker process, one first call, and three warm calls.
- Legacy and compact paths both used the approved six-decimal structural
  evidence policy. Only their event execution layouts differed.

## Fixed-checkpoint performance

| scope | path | queries | advancing slots | executed slots | first s | warm calls s | warm median s | speedup |
|---|---|---:|---:|---:|---:|---|---:|---:|
| 64 tasks | legacy | 68 | 19,780 | 47,940 | 11.672 | 11.498, 11.857, 11.641 | 11.641 | baseline |
| 64 tasks | compact | 68 | 19,780 | 30,615 | 9.574 | 8.683, 8.737, 8.563 | 8.683 | **1.341x** |
| 400 tasks | legacy | 416 | 116,128 | 293,280 | 67.769 | 63.238, 70.380, 70.723 | 70.380 | baseline |
| 400 tasks | compact | 416 | 116,128 | 178,935 | 39.473 | 38.283, 41.362, 43.261 | 41.362 | **1.702x** |

Both scopes exceed the required `1.20x` warm-median speedup.

## Fixed-checkpoint correctness

| gate | 64 tasks | 400 tasks |
|---|---:|---:|
| exact tasks, legacy and compact | 3/64 | 4/400 |
| task exact flags | identical | identical |
| maximum per-task loss difference | `8.77e-8` | `8.77e-8` |
| task-owner sets | identical | identical |
| owner codes | identical | identical |
| neuron-score maximum difference | `0.0` | `0.0` |
| source-score maximum difference | `0.0` | `0.0` |
| target-score maximum difference | `0.0` | `0.0` |
| edge-score maximum difference | `0.0` | `0.0` |

The unresolved loss was `2.233622691206484` for the 64-task legacy path and
`2.2336226927663216` for the compact path. It was `4.052317963648911` for the
full legacy path and `4.052317963801504` for the compact path. These aggregate
differences are below `1.6e-9`; the literal per-task gate above is the
authoritative tolerance result.

The following 64-task mutation topology digests were identical between legacy
and compact execution:

| mutation | topology SHA-256 |
|---|---|
| neuron add | `d5fe5c9e1c5a5aa8fd67a33b0a848f6316ef349f35bb6f364e0d32ebf443f289` |
| neuron prune | `169195f7d05de3c5be63284d7088fe0d9962c673cbf32c1fb1e0047cd42d4939` |
| edge add | `599017915b370ffc76847a900b4e9b3357c02a8546fc458136b9d370be4d17c2` |
| edge prune | `766687b98cad73711447d798c05a1d86843233c178c2a2bda2e40d93654458ca` |
| Dale excitatory | `baf320ea18e6064a9f857944266fca9e4969835cf7270962b660da9f7f5a0a0e` |
| Dale inhibitory | `9358319c0704fbf0dd89c6945d69913d0dd935f3a9d6b058f8b51b28606e9f24` |

## Memory

| scope | path | host peak bytes | device peak bytes |
|---|---|---:|---:|
| 64 tasks | legacy | 2,197,405,696 | 213,849,344 |
| 64 tasks | compact | 2,123,821,056 | 96,657,408 |
| 400 tasks | legacy | 3,141,632,000 | 616,813,312 |
| 400 tasks | compact | 2,972,106,752 | 348,186,624 |

The compact path reduced device peak by 54.8 percent at 64 tasks and 43.6
percent at 400 tasks. Host peak decreased by 3.3 and 5.4 percent respectively.

## Tests and static checks

- Adapter module: `58 passed in 11.42s`.
- Relevant Example 21 regression set: `265 passed in 203.40s`.
- Branch-aware adapter coverage: 96 percent (`796` statements, `200` branches).
- Ruff check: passed.
- Ruff format check: passed.
- Basedpyright production adapter: zero errors; the module retains its existing
  dynamic-injection warnings.
- `git diff --check`: passed.

The regression set comprised the co-located adapter, coordinator, structural,
and BrainCell ARC modules. The existing BrainTrace compiler warnings about
unregistered readout states remained warnings and did not fail any gate.

## One-round GPU evolution canary

Command configuration:

- device `gpu`;
- rounds `1` and patience `2`;
- exactly `128` updates per training block;
- one topology operation per stage;
- 64-task structural screen; and
- complete 400-task round score and terminal evaluation.

The run completed in 12 minutes 37 seconds. Every training sibling completed
exactly 128 updates: one parent-training block plus two siblings in each of
edge, neuron, edge-revisit, and Dale stages, for 1,152 executed candidate
updates. Rescore and terminal stages executed zero updates.

| stage | scope | selected disposition | executed updates | elapsed s |
|---|---|---|---:|---:|
| train | full | accepted training | 128 | 87.74 |
| round screen | 64 tasks | accepted rescore | 0 | 17.06 |
| edge | 64 tasks | accepted prune | 256 | 97.39 |
| neuron | 64 tasks | accepted add | 256 | 119.01 |
| edge revisit | 64 tasks | accepted add | 256 | 114.87 |
| Dale | 64 tasks | retained parent | 256 | 111.68 |
| round score | full | accepted rescore | 0 | 66.87 |
| terminal evaluation | evaluation 400 | terminal | 0 | 53.39 |

The durable state is closed with `terminal_reason=round-budget`, cursor `640`,
evaluation completed, training score `4/400`, terminal evaluation `0/400`,
2,151 neurons, and 18,011 recurrent edges. The final selected checkpoint is:

`acc691d20d4419bd14558c9163b46f98cafaa573f350644e4f3cafa20c0e5e9b`

Its lineage is continuous:

`62e8a510... -> dde3c6aa... -> 14f882b2... -> 974933aa... -> 3d8f61f9... -> ff23d22e... -> acc691d2...`

The final checkpoint hash was re-read from disk and matched `run-state.json`.
The evaluation digest is
`c1c443edc90f1cc23c9f3e1078eaa92475ac88fb64e52a64526536c4981d1f94`.

## Qualification boundary

This evidence qualifies the scoring optimization and its one-round Example 21
integration on the stated GPU, checkpoint, and corpus. It does not qualify a
multi-round capability improvement, change the accepted training policy, or
claim that the branch is integrated into `main`.
