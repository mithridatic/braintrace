# Example 21 chained-operation GPU canary

## Disposition

Implementation commit `026c6fa33f8f554757dad23fca5abe1995a7113c` adds a per-round structural operation budget
(`--topology-operations-per-round`) and a screen subset for intra-round
comparisons (`--screen-tasks`). Two real GPU runs qualify it against the
complete 400-task ARC training corpus and the 400-task held-out corpus. Both
closed with exit status 0.

The runs support an optimization decision about cost and structural
throughput. They are not a capability claim: held-out strict pass@1 is
0/400 for the uninterrupted run and
0/400 for the resumed run, and one round of
either lineage reaches 4/400 exact training tasks.

## Run one — eight chained operations, uninterrupted

`--rounds 1 --patience 2 --updates 128 --topology-operations-per-round 8
--screen-tasks 64`, output `var/example21-ops-canary`.

| stage | scope | op | disposition | exact | neurons | edges | seconds | siblings |
|---|---|---|---|---|---|---|---|---|
| `r000-train` | full | 0 | accepted | 0 | 2048 | 16384 | 94.1 | training accepted (2048n/16384e) |
| `r000-round-screen` | screen | 0 | accepted | 0 | 2048 | 16384 | 13.3 | rescore accepted (2048n/16384e) |
| `r000-op00-edge` | screen | 0 | accepted | 0 | 2048 | 17204 | 80.4 | add accepted (2048n/17204e); prune rejected (2048n/15564e) |
| `r000-op01-neuron` | screen | 1 | accepted | 0 | 2151 | 19073 | 91.0 | add accepted (2151n/19073e); prune rejected (1945n/15588e) |
| `r000-op02-edge-revisit` | screen | 2 | accepted | 3 | 2151 | 18119 | 89.9 | add rejected (2151n/20027e); prune accepted (2151n/18119e) |
| `r000-op03-dale` | screen | 3 | retained-parent | 3 | 2151 | 18119 | 91.7 | excitatory rejected-regression (2151n/18119e); inhibitory rejected-regression (2151n/18119e) |
| `r000-op04-edge` | screen | 4 | retained-parent | 3 | 2151 | 18119 | 98.9 | add rejected-regression (2151n/19025e); prune rejected-regression (2151n/17213e) |
| `r000-op05-neuron` | screen | 5 | accepted | 3 | 2259 | 20021 | 107.5 | add accepted (2259n/20021e); prune rejected-regression (2043n/16465e) |
| `r000-op06-edge-revisit` | screen | 6 | retained-parent | 3 | 2259 | 20021 | 96.6 | add rejected-regression (2259n/21023e); prune rejected-regression (2259n/19019e) |
| `r000-op07-dale` | screen | 7 | retained-parent | 3 | 2259 | 20021 | 97.9 | excitatory rejected-regression (2259n/20021e); inhibitory rejected-regression (2259n/20021e) |
| `r000-round-score` | full | 8 | accepted | 4 | 2259 | 20021 | 79.0 | rescore accepted (2259n/20021e) |
| `r000-round-end` | full | 8 | accepted | 4 | 2259 | 20021 | 0.0 | — |
| `terminal-evaluation` | full | 0 | terminal | 4 | 2259 | 20021 | 73.1 | — |

The round costs 940.3 s for eight structural operations, excluding the
73.1 s terminal evaluation. The recorded single-pass baseline in
`var/example21-canary-c4d1c2b/progress.jsonl` costs 905.8 s for four
operations under complete-corpus scoring, excluding its 67.6 s terminal
evaluation. This run therefore performs twice the structural operations for
1.04 times the round time. Every screened stage
costs roughly 90 s against roughly 200 s for the same stage scored on the
complete corpus; the two scope transitions cost 13.3 s and 79.0 s.

Observations that qualify the mechanism:

- `r000-op00-edge` compares 17,204 against 15,564 from a 16,384-edge parent.
- `r000-op02-edge-revisit` proposes from 19,073, the state `r000-op01-neuron`
  selected, rather than from the round-entry state.
- `r000-op04-edge` executes at all. The historical lifecycle ended a round at
  the Dale stage, so a second cycle had no way to run.
- Cursor advances 128 per operation and ends at 1,152, which is one ordinary
  training block plus eight operations.
- All twelve stage identities are distinct.
- `r000-round-score` restores the complete-corpus scope before the round
  comparison, leaving the model at 2,259 neurons and 20,021 edges.

## Run two — interruption and resume across an operation boundary

Same configuration, output `var/example21-resume-canary`. The process was
killed with SIGKILL during `r000-op03-dale`, leaving five durable records,
`next_stage` `dale` at operation index 3, cursor 512, no pending journal, and
one orphaned candidate file.

Resuming with the identical command produced:

| stage | scope | op | disposition | exact | neurons | edges | seconds | siblings |
|---|---|---|---|---|---|---|---|---|
| `r000-train` | full | 0 | accepted | 0 | 2048 | 16384 | 112.9 | training accepted (2048n/16384e) |
| `r000-round-screen` | screen | 0 | accepted | 0 | 2048 | 16384 | 16.4 | rescore accepted (2048n/16384e) |
| `r000-op00-edge` | screen | 0 | accepted | 0 | 2048 | 17204 | 96.0 | add accepted (2048n/17204e); prune rejected (2048n/15564e) |
| `r000-op01-neuron` | screen | 1 | accepted | 0 | 2151 | 19074 | 110.3 | add accepted (2151n/19074e); prune rejected (1945n/15589e) |
| `r000-op02-edge-revisit` | screen | 2 | accepted | 2 | 2151 | 18120 | 102.9 | add rejected (2151n/20028e); prune accepted (2151n/18120e) |
| `r000-op03-dale` | screen | 3 | retained-parent | 2 | 2151 | 18120 | 103.9 | excitatory rejected-regression (2151n/18120e); inhibitory rejected-regression (2151n/18120e) |
| `r000-op04-edge` | screen | 4 | retained-parent | 2 | 2151 | 18120 | 95.6 | add rejected-regression (2151n/19026e); prune rejected-regression (2151n/17214e) |
| `r000-op05-neuron` | screen | 5 | accepted | 3 | 2259 | 20031 | 102.3 | add accepted (2259n/20031e); prune rejected (2043n/16473e) |
| `r000-op06-edge-revisit` | screen | 6 | retained-parent | 3 | 2259 | 20031 | 100.4 | add rejected-regression (2259n/21033e); prune rejected-regression (2259n/19029e) |
| `r000-op07-dale` | screen | 7 | retained-parent | 3 | 2259 | 20031 | 101.1 | excitatory rejected-regression (2259n/20031e); inhibitory rejected-regression (2259n/20031e) |
| `r000-round-score` | full | 8 | accepted | 4 | 2259 | 20031 | 81.6 | rescore accepted (2259n/20031e) |
| `r000-round-end` | full | 8 | accepted | 4 | 2259 | 20031 | 0.0 | — |
| `terminal-evaluation` | full | 0 | terminal | 4 | 2259 | 20031 | 80.4 | — |

The resumed process did not re-execute operations 00 through 02, removed the
orphaned candidate, kept the progress sequence gapless, chained every parent
digest to its predecessor's child digest, and closed with an empty candidate
directory and no pending journal.

## Known variation

The two lineages agree exactly through `r000-op00-edge` and then differ by one
recurrent edge at `r000-op01-neuron` (19,073 against 19,074) at an identical
neuron count. Selection is deterministic given its scores; the difference is in
which structure the ranking selects. Half of the structural ranking is activity
evidence, and under a screen that evidence is measured over 64 tasks rather
than 400, so a near-tie is easier to reorder under nondeterministic device
arithmetic. This is the subset-ranking approximation the specification records,
not a selection defect. The round boundary still scores and compares on the
complete corpus.
