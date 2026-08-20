# Example 21 Docker runtime acceleration

Status: implemented; measured target not met
Date: 2026-08-20
Branch: `perf/example21-docker-throughput`

## Scope and contract

This change optimizes the production Docker workload documented in
`examples/pp_prop/README.md`. The model topology, admitted data, scoring
protocol, optimizer/update budget, batch size, latent horizon, and candidate
semantics remain fixed. The speed contract is execution-only: a warm,
end-to-end run is measured from the existing Docker command, with no pytest
benchmark substituted for the production workload.

The target is at least 2x warm end-to-end speed for the canonical run, or a
median of at most 99.3 seconds. If the target is not reached, only measured
changes that pass the equivalence and resource gates may remain, and the final
report must state the measured decomposition without a 2x claim.

## Exact README baseline command

At the baseline revision, the README command is:

```powershell
docker run --rm --gpus all --env XLA_PYTHON_CLIENT_MEM_FRACTION=0.80 --volume "${PWD}/var:/work/var" --volume "${PWD}/var/jax-cache:/cache/jax" braintrace-gpu:0.11.0-py314-msgspec-arc python /opt/braintrace/examples/pp_prop/21-latent-reasoning-in-context.py --device gpu --source-manifest /datasets/arc/example21-sources.json --output-dir /work/var/example21-shared-1024n-1024e-b32-u13-l390 --neurons 1024 --recurrent-edges 1024 --max-demonstrations 10 --latent-steps 390 --training-updates 13 --training-batch-size 32 --training-chunk-size 1
```

The command above is retained as the explicit chunk-1 baseline in this spec.
The supplied canonical throughput artifact uses the same model/data protocol
with `training-updates=130`; that distinction is recorded rather than silently
changing the README command's update budget.

## Baseline provenance and machine state

These values were captured on the clean worktree before implementation:

| item | value |
| --- | --- |
| BrainTrace source revision | `f4440a385eb7a51439f5a4dbc1b8993aebc122ce` |
| BrainTrace source Git tree | `641987955f250c58b0cb7a3a08f958f2e98644e8` |
| ARC-AGI-1 source path | `C:\tmp\braintrace-example21-data\arc-agi-1` |
| ARC-AGI-1 source revision | `aa922be204204ec148a1137fe6ed4d34ddde812b` |
| ARC-AGI-1 Git tree | `5ea85050ccdc46cfe5e71b618cc5148faa4f0770` |
| configured image | `braintrace-gpu:0.11.0-py314-msgspec-arc` |
| baseline image ID/digest | unavailable: the tag was not present on this host before the build |
| Docker server | `29.0.1` |
| GPU | NVIDIA GeForce RTX 3080 Ti Laptop GPU, 16,384 MiB |
| NVIDIA driver | `595.79` |
| host JAX cache | missing at `var/jax-cache` before the run |
| source output directory | missing at `var` before the run |

The image recipe pins Python 3.14, JAX 0.11.0, CUDA 12 dependencies, and the
base image digest in `.github/containers/braintrace-gpu/Dockerfile`. A
qualifying post-change image must be rebuilt from the worktree and its image ID,
OCI source revision, ARC revision, and index/data digest must be retained with
the run logs.

## Supplied baseline matrix

The following values are the current artifacts supplied with this task. No raw
timing sidecars for these runs are present in the fresh checkout, so the phase
residual is explicitly not treated as an independently measured training phase.
Every supplied run used `training_chunk_size=1`.

| arm | model/configuration | total seconds | evaluation seconds | known non-evaluation residual |
| --- | --- | ---: | ---: | ---: |
| canonical | 1024 neurons / 1,024 edges / batch 32 / 130 updates / latent 390 | 198.5 | 21.6 | 176.9, includes training plus data/model/artifacts |
| dense-edge | 1024 neurons / 1,047,552 edges | 592.3 | 22.0 | 570.3, includes training plus data/model/artifacts |
| high-update | 1024 neurons / 1,024 edges / 1,040 updates / latent 150 | 1,859.3 | 3.8 | 1,855.5, includes training plus data/model/artifacts |

The implementation must add an opt-in machine-readable profile that separates
data, model, training, evaluation, and artifact phases, then separates training
producer encoding, consumer wait, host-to-device staging, first-call compile,
steady-state device compute, and host result copy. Profile synchronization is
diagnostic only; ordinary asynchronous execution is used for final throughput
timing.

## Optimization order

1. Benchmark compiled training chunk divisors `[1, 2, 5, 10, 13, 26, 65,
   130]` for the canonical update budget. Keep `training_bank_size=0` so
   sampling and augmentation semantics do not change. Select the fastest median
   that passes exact schedule/tensor equivalence, numerical tolerance, and
   memory gates, then validate it on the dense-edge and high-update arms.
2. If profiling shows material CPU starvation, optimize only the private
   training encoder with direct batched NumPy construction. The scalar encoder
   remains the oracle. The optimized path must preserve byte-identical events,
   masks, metadata, ordering, and read-only outputs across orientations,
   held-out demonstrations, colors, boundary grid sizes, worker counts, and
   failure cleanup.
3. For the dense-edge arm, expose and record the existing CSR backend choice
   (`default` or `jax_raw`) and benchmark both in Docker. Adopt an explicit
   backend only when it improves runtime by at least 10% and preserves forward
   and gradient tolerances, scores, and memory safety. Replacing the ETP graph
   with a dense recurrent operator is out of scope.

## Memory and correctness gates

- Keep `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`; the existing fail-closed policy
  rejects allocator or physical peak use above the 85% ceiling.
- No CUDA fault, process failure, or incomplete GPU evidence is acceptable.
- Chunk-1 and selected-chunk schedules/tensors must be identical; losses and
  parameter leaves must agree within `1e-6`, with exact score and candidate
  equality.
- `training_bank_size` remains zero.
- The complete ARC metric set, update count, latent horizon, and provenance
  must remain unchanged.

## Acceptance protocol

Build one exact ARC image from the worktree. Use the README volumes and
provenance variables, run one warm-up, then three measured warm runs per
selected configuration. Record image/cache state, source and data fingerprints,
GPU/driver, phase profile, peak device use, CUDA status, scores, candidates,
and raw stdout/stderr. The canonical median must be at most 99.3 seconds, no
supplied matrix arm may regress by more than 5%, peak device use must remain
below the existing 85% ceiling, and ARC metrics must be unchanged.

## Implementation and measurement log

This section is updated only with results produced from this branch. A missing
Docker/GPU gate is reported as unavailable rather than inferred from a CPU or
synthetic run.

### Final image and machine evidence

The final ARC image was rebuilt from this worktree after the implementation and
README changes:

| item | value |
| --- | --- |
| image | `braintrace-gpu:0.11.0-py314-msgspec-arc` |
| image ID | `sha256:29a00d54a766252a36e05cd64c43cf3b5e130583b8002e24c8a5eba97ffd6709` |
| OCI source revision | `f4440a385eb7a51439f5a4dbc1b8993aebc122ce` |
| final source-tree SHA-256 | `11a30e898324757ac7eda28df56954835a08b332dcfe78e98dbe471ce1c26f07` |
| source dirty flag | `1` (worktree changes intentionally included) |
| ARC-AGI-1 revision | `aa922be204204ec148a1137fe6ed4d34ddde812b` |
| ARC image index | 399 training tasks / 400 evaluation tasks |
| final data manifest SHA-256 | `b9ab482f3f4f03193cf5ebd73433ab1899685797079e39fb9cf8ee88d6ad7d2f` |
| Docker server | `29.0.1` |
| GPU | NVIDIA GeForce RTX 3080 Ti Laptop GPU, 16,384 MiB |
| NVIDIA driver | `595.79` |
| allocator limit | `13,744,734,208` bytes (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`) |
| persistent cache | host `var/jax-cache` mounted at `/cache/jax`; populated during the final warm-up series |

The final runs passed the GPU safety checks. One unrelated GPU container from a
different worktree briefly contended with an early sample; that sample is
retained as `var/example21-canonical-u130-c5-run1` but excluded from all clean
statistics. The later selected runs had no competing GPU container.

### Chunk sweep and selection

The canonical divisor sweep used 130 updates, chunk divisors from the planned
set, the selected model/data contract, and evaluation limit 1 for development
throughput screening. The following are the recorded result runtimes; they are
not the final full-evaluation acceptance runs:

| chunk | runtime seconds | training seconds | result | peak allocator bytes |
| ---: | ---: | ---: | --- | ---: |
| 1 | 257.000 (full evaluation run) | 198.299 | reference artifact | 2,623,537,152 |
| 2 | 196.602 | 184.763 | passed | 813,694,976 |
| 5 | 129.745 | 119.527 | provisional fastest | 1,350,565,888 |
| 10 | 158.262 | 148.413 | passed | 2,424,307,712 |
| 13 | 140.919 | 128.385 | passed | 2,424,307,712 |
| 26 | 159.825 | 148.338 | passed | 4,571,791,360 |
| 65 | — | — | failed memory gate (4 GiB allocation) | — |
| 130 | — | — | failed XLA argument gate (about 14.09 GB) | — |

Chunk 5 was selected because it was the fastest safe divisor in the screening
run and divides both 130 and 1,040. `training_bank_size` remained zero.

### Profiling evidence

The final image's diagnostic run used 25 updates, chunk 5, evaluation limit 1,
and `--profile`. It reported:

| phase or component | seconds |
| --- | ---: |
| data | 0.929 |
| model | 0.993 |
| training | 18.598 |
| evaluation | 1.126 |
| artifacts | 1.077 |
| producer encoding | 16.620 |
| consumer wait | 10.409 |
| host-to-device staging | 0.509 |
| first-call compilation | 2.466 |
| first-call device compute | 0.026 |
| steady-state device compute | 0.110 |
| host result copy | 0.009 |

The profile contains diagnostic synchronization barriers and is not a
throughput measurement. It shows that CPU episode construction, rather than
device compute, remains the limiting resource. The production encoder is now a
bounded batched NumPy path with the scalar encoder retained as its oracle, and
the outer queue remains one chunk ahead.

### Correctness and backend evidence

The co-located regression compares chunk 1 and chunk 5 on augmented,
non-plumbing training data with two workers. Events, advances, dimensions,
colors, masks, efforts, fingerprints, held-out indices, ordering, and the
deterministic fake loss stream are identical. A final-image Docker checkpoint
comparison (`sha256:29a00d54...`) found exact losses, fingerprints, metrics, and
candidate grids; all 15 parameter leaves had a maximum absolute difference of
`3.73e-9`, below the `1e-6` gate.

For the dense-edge backend screen (1,024 neurons, 1,047,552 edges, 13 updates,
chunk 1, evaluation limit 1):

| backend | runtime seconds | peak device bytes | metrics/candidate grids |
| --- | ---: | ---: | --- |
| `default` | 27.890 | 492,830,720 | exact match |
| `jax_raw` | 43.531 | 761,266,176 | exact match |

`jax_raw` is 56.1% slower, so the production default remains `default`. The
existing sparse-gradient test path is currently blocked by an unrelated
baseline compiler/JAX failure (`foreach() argument 2 is shorter than argument
1`); no explicit backend change was adopted on that basis.

### Phase-2 producer optimization

The selected chunk-5 run remains CPU-feed limited: the diagnostic profile
measured 16.620 seconds of producer encoding and 10.409 seconds of consumer
wait, while steady-state device compute was only 0.110 seconds.

Two bounded alternatives were screened against the scalar schedule/tensor
oracle on the complete smoke data path. Direct compact-batch assembly measured
about `0.70x` the existing batched-row throughput, and two concurrently
prepared chunks measured about `0.67x`; both were rejected and are not retained
in the production path. Neither experiment changed the device chunk size or
the random schedule.

The retained change caches the immutable target-free form and base fingerprint
of each source task. It removes repeated construction and hashing while
preserving augmentation, held-out selection, event bytes, metadata, and all
training/evaluation semantics. A repeated-source CPU microbenchmark measured
`7.78x` faster descriptor preparation after the cache was warm; this is a
producer microbenchmark, not an end-to-end Docker claim. The scalar encoder
and row-packing path remain the equivalence oracle.

### Phase-2 Docker follow-up

The phase-2 image was rebuilt as
`sha256:daaa80a5c2b1cb85f9fe1e84f932da97709e9e8bed4ce5c41a77d65bb20d80c5`
from the dirty worktree. Its 25-update diagnostic run completed on GPU with
producer encoding `14.685 s`, consumer wait `8.851 s`, training phase
`16.622 s`, and total runtime `19.710 s`. Peak physical device use was
`5,344,591,872` bytes (`31.1%` of the 16-GiB device), with no CUDA fault.
The corresponding prior profile was `16.620 s` producer encoding,
`10.409 s` consumer wait, and `18.598 s` training, so the cache reduced the
diagnostic producer boundary without changing the model contract.

The canonical warm-up completed in `148.345 s`. Two subsequent canonical
samples were invalid for acceptance because unrelated GPU containers were
running concurrently: clean 1 was `199.408 s` and clean 2 was `303.600 s`.
The third clean sample was not started while the external GPU workload
continued. These samples are retained as contention evidence, not as a
throughput claim or replacement for the uncontended `132.465 s` reference.

### Canonical warm-run acceptance evidence

Configuration: 1,024 neurons, 1,024 edges, batch 32, 130 updates, latent 390,
chunk 5, full 400-task evaluation, ordinary unprofiled execution. The warm-up
and three clean measured runs were:

| run | runtime seconds | peak physical device bytes | ARC metric/candidate equality |
| --- | ---: | ---: | --- |
| warm-up | 127.254 | 4,569,694,208 | baseline-equivalent |
| clean 1 | 129.900 | 4,569,694,208 | exact |
| clean 2 | 132.465 | 4,569,694,208 | exact |
| clean 3 | 133.085 | 4,569,694,208 | exact |

The clean median is `132.465 s`. The full effort-390 metrics were unchanged
across all four runs (`query pass@1=0`, `query pass@2=0`, `strict task pass@1=0`,
`strict task pass@2=0`, shape diagnostic `0.002386634844868735`, valid-cell
pixel diagnostic `0.12214543632023442`), and every candidate grid matched.
Peak physical use was 26.6%, below the 85% ceiling, with no CUDA fault.

Against the supplied chunk-1 canonical baseline of `198.5 s`, the clean
median is `0.667x` the baseline, a `33.3%` end-to-end improvement. The warm-up
was not used as the acceptance statistic. The clean measured training-phase
median was about `93.9 s`, with about `35.3 s` in evaluation; this is a real
warm-run improvement but still misses the 2x / `99.3 s` target. The no-regression
gate passes for the canonical arm, while the complete supplied matrix was not
repeated at three-run depth.

The dense-edge and high-update validation status is asymmetric: the dense-edge
backend screen passed at a reduced 13-update/evaluation-limit-1 workload; the
selected high-update run below was executed once as a full 1,040-update,
latent-150, chunk-5 development validation, while the supplied matrix's
three-run acceptance protocol was not repeated for that expensive arm.
The high-update validation completed at `707.352 s` with 1,040 updates,
latent horizon 150, chunk 5, and evaluation limit 1. All 1,040 losses were
finite; peak physical device use was `1,350,565,888` bytes (`7.86%`), and the
run had no CUDA fault. This is a single validation run, not a replacement for
the supplied arm's full three-run acceptance protocol.

### Post-implementation execution rerun

After the phase-2 producer-cache change, one ordinary full canonical run was
executed on the clear RTX 3080 Ti GPU. The configuration was 1,024 neurons,
1,024 recurrent edges, batch 32, 130 updates, latent 390, chunk 5, and the
complete 400-task evaluation. The result is retained at
`var/example21-execute-u130-c5/result.json`.

The recorded result runtime was `111.495 s` (the outer Docker command elapsed
about `116.377 s`), which is approximately `1.78x` faster than the supplied
`198.5 s` chunk-1 baseline. Peak physical device use was `4,790,943,744` bytes
(`27.9%` of the 16 GiB GPU), and the GPU safety gate was `safe`. Effort-390
metrics were `query pass@1=0`, `query pass@2=0`, `strict task pass@1=0`,
`strict task pass@2=0`, shape diagnostic `0.002386634844868735`, and valid-cell
pixel diagnostic `0.12214543632023442`, with 400 tasks and 419 queries
evaluated. No CUDA fault occurred. This is a single rerun and does not replace
the three-run acceptance median above.
