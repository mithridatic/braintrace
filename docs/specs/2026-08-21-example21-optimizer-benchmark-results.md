# Example 21 optimizer benchmark results

## Outcome

All five controlled arms completed the full 260-update training schedule and
419-query evaluation. Every arm scored 0 exact ARC queries, so the optimizers
tie on the primary ARC metric. Muon improved pixel accuracy and training loss
but reduced shape accuracy relative to Adam. Raising weight decay from `0.01`
to `0.05` did not materially improve Muon and traded AdamW pixel accuracy for
shape accuracy.

## Controlled configuration

Every arm used seed `2108`, 4096 neurons, 8192 recurrent edges, 260 updates,
60 latent steps, batch size 32, chunk size 5, four training workers,
`jax_raw`, learning rate `1e-3`, and the complete ARC-AGI-1 source manifest.
Runs were sequential and uncontended on an NVIDIA GeForce RTX 3080 Ti Laptop
GPU with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`.

All artifacts report source revision
`5261f9050448943b80114b67bebeaae461fdaf95`, source-tree SHA-256
`1d1fca67716fc828bf8405ff02a178a754ab48fdab1ccd4412ce488d1bd29465`,
`source_dirty=false`, initial parameter SHA-256
`0d3b15281001a675d01bea71d94e188cfd81c6a3bbe3e2f9071f6533941cdd79`,
and manifest SHA-256
`b9ab482f3f4f03193cf5ebd73433ab1899685797079e39fb9cf8ee88d6ad7d2f`.
The shared initial parameter hash verifies identical initialization.

The image was `braintrace-gpu:0.11.0-py314-msgspec-arc`, image ID
`sha256:3d2d6962bcfe661e1b8cc76039a1fa2dcba77544c72f63cf8045d0ef447b12bc`.
The runtime contained Python 3.14.0, BrainTrace 0.2.5, BrainState 0.5.3,
BrainPy 2.8.2, BrainTools 0.3.0, Optax 0.2.8, JAX/JAXlib 0.11.0, and
NumPy 2.4.6.

## Primary evaluation

Metrics are from the frozen shared model at the primary 60-step submission.

| Optimizer | Decay | Exact queries | Pass@1 | Pass@2 | Strict task pass@1/@2 | Shape | Pixel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adam | 0.00 | 0/419 | 0.00% | 0.00% | 0.00% / 0.00% | 56.0859% | 39.0576% |
| AdamW | 0.01 | 0/419 | 0.00% | 0.00% | 0.00% / 0.00% | 55.3699% | 39.0224% |
| AdamW | 0.05 | 0/419 | 0.00% | 0.00% | 0.00% / 0.00% | 56.0859% | 38.6503% |
| Muon | 0.01 | 0/419 | 0.00% | 0.00% | 0.00% / 0.00% | 53.4606% | 43.7929% |
| Muon | 0.05 | 0/419 | 0.00% | 0.00% | 0.00% / 0.00% | 53.4606% | 43.7743% |

Adam and AdamW at decay `0.05` tie for the highest shape diagnostic. Muon at
decay `0.01` has the highest pixel diagnostic, 4.7353 percentage points above
Adam, while its shape diagnostic is 2.6253 points below Adam. Diagnostic
metrics do not break the 0/419 tie on exact ARC performance.

## Training and resources

| Optimizer | Decay | Initial loss | Final loss | Minimum loss | Mean loss | Finite updates | Active groups moved | Runtime | JAX peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adam | 0.00 | 2.935269 | 2.481010 | 1.915024 | 2.858892 | 260/260 | 7/7 | 266.131 s | 1,301,325,056 |
| AdamW | 0.01 | 2.935270 | 2.482122 | 1.914072 | 2.858249 | 260/260 | 7/7 | 263.511 s | 1,301,297,408 |
| AdamW | 0.05 | 2.935274 | 2.464049 | 1.912090 | 2.855670 | 260/260 | 7/7 | 245.921 s | 1,282,608,384 |
| Muon | 0.01 | 2.935265 | 2.050215 | 1.624898 | 2.519945 | 260/260 | 7/7 | 384.654 s | 1,297,044,480 |
| Muon | 0.05 | 2.935268 | 2.048941 | 1.625319 | 2.519325 | 260/260 | 7/7 | 288.204 s | 1,299,012,096 |

The sum of per-group L2 deltas was 73.9160 for Adam, 73.9445 for AdamW
`0.01`, 74.3698 for AdamW `0.05`, 19.9357 for Muon `0.01`, and 21.3235 for
Muon `0.05`. These sums are movement evidence, not a global parameter-vector
L2 norm and not a quality metric.

Runtime is a single ordered observation with a shared persistent compilation
cache. In particular, each `0.05` follow-up could reuse compilation work from
its optimizer's `0.01` arm, so the runtime difference must not be attributed
to weight decay. Muon `0.01` also emitted a recoverable 2 GiB allocation
warning during evaluation; the evaluation completed and the process exited
successfully.

## Interpretation and limits

The primary conclusion is a five-way tie at 0/419 exact queries. Under the
secondary diagnostics, Muon `0.01` is the strongest pixel/loss configuration,
while Adam and AdamW `0.05` are strongest on shape. The `0.05` sensitivity
arms do not establish a better weight decay: AdamW's shape gain accompanies a
pixel loss, and Muon's evaluation is effectively unchanged.

This is one seed at a fixed learning rate and budget, not optimizer-specific
tuning. The requested 8192-edge topology is intentionally below Example 21's
1,048,576-edge full-scale qualification target. The artifacts therefore fail
structural qualification for expected scale reasons. The JAX allocator peaks
were about 1.28-1.30 GB and below the 85% VRAM ceiling, but the in-container
`nvidia-smi` monitor did not retain a process-memory peak, so the formal GPU
runtime-safety field remains `insufficient_evidence` rather than qualified.

## Retained artifacts

Each directory contains `result.json`, `report.txt`, `data_manifest.json`, and
`latent_reasoning.png`:

- `temp/example21-optimizer-benchmark-adam-s2108`
- `temp/example21-optimizer-benchmark-adamw-s2108`
- `temp/example21-optimizer-benchmark-adamw-wd005-s2108`
- `temp/example21-optimizer-benchmark-muon-s2108`
- `temp/example21-optimizer-benchmark-muon-wd005-s2108`

The raw artifacts are deliberately retained in the worktree's ignored
`temp/` area; this results document is the tracked comparison record.
