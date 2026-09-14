# H01 scoring performance, Vast.ai RTX 4090

The complete default 104-cell evolution **has not been reduced to 30 seconds**.
No full evolution speedup or ARC accuracy improvement is claimed. The user's
2.5–3 hour estimate implies a 300–360x improvement would be required; that
estimate was not independently established by completing the run here.

Base revision: `e7b8f3a`. Measurements ran on the existing `braintrace-gpu`
RTX 4090, Python 3.14.7, in an isolated checkout using `/workspace/venv314`.
Another session's evolution and tests remained active. Timing comparisons are
small samples on shared hardware, not statistically qualified benchmarks.

## Shipped changes

- Hoist the readout matrix multiplication out of the physical event loop and
  evaluate only the final 31 request feature vectors. Keep physical stepping,
  padding behavior, reset, precision, dt, and activity reduction unchanged.
- Reuse the multi-query JIT callable on its owning session. Parameter State
  values remain dynamic; separate/restored runtimes receive separate callables.
- Release the cached callable before releasing a parent runtime during mutation.

Training episodes, optimizer, finite-window eligibility, corpus size, evolution
rounds, structural operations, and durable checkpoint frequency are unchanged.
The source hash guard remains enabled: existing pinned manifests/checkpoints
do not silently become compatible with changed source. Generate a fresh source
manifest with `examples/h01_arc_manifest.py` for a new run at this revision;
continue existing checkpoints on their original pinned revision.

## Measured scoring fixture

`docs/evidence/h01_score_profile.py` uses a deterministic 104-channel stateful
fixture with 2,048 events and 360 readout channels. It exercises the actual
scoring functions but **does not simulate the 104 anatomical cables**.
Calls are synchronized with `jax.block_until_ready`; cProfile covers the first
call. Peak GPU memory is the process allocator statistic, not full application
VRAM. RSS includes imports and runtime initialization.

| Repeated-query path | Original | Optimized |
| --- | ---: | ---: |
| Cold call | 0.394 s | 0.470 s |
| Subsequent call 1 | 0.230 s | 0.0236 s |
| Subsequent call 2 | 0.220 s | 0.0249 s |
| Peak GPU bytes | 42,251,264 | 17,086,976 |
| Peak host RSS, KiB | 1,086,192 | 1,073,972 |

Subsequent scoring was approximately 9x faster with 60% lower peak GPU
allocation. Cold compilation was slower. Host RSS improved only about 1%.
Raw cProfile reports are `h01-query-{original,optimized}-20260914.txt`.
The standalone episode comparison is also retained in `h01-score-*.txt`;
it showed a smaller warm benefit and no host-RSS improvement.

Reproduce from repository root with `PYTHONPATH=.`,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`:

```sh
/workspace/venv314/bin/python docs/evidence/h01_score_profile.py --implementation original --queries
/workspace/venv314/bin/python docs/evidence/h01_score_profile.py --implementation optimized --queries
```

## Real H01 line profile

A bounded 12-cell construction/initialization profile completed under a
180-second cap: `make_h01_network` 73.5652 s and
`init_h01_network_states` 73.0921 s. These **include line-profiler overhead**
and must not be interpreted as ordinary wall-time benchmarks or extrapolated
linearly to 104 cells. Inclusive hot-function times include CV geometry
28.14 s, CV mechanisms 17.53 s, region normalization 15.01 s, and node-tree
construction 14.38 s. These overlap and must not be summed.

The exact input harness and full line report are retained alongside this file.
No real-cell construction or training optimization was shipped in this change.

## Validation and limits

- Baseline fast scoring contracts: 5 passed in 8.39 s with `pytest -n 2`.
- Execution, adapter, and session modules: 35 passed in 119.49 s with
  `pytest -n 2`, including real update/save/restore/clone and cache release.
- Final execution module with added real-cell comparison to the old scoring
  implementation: 7 passed in 24.23 s.
- Added adapter scoring/ranking integration check: 1 passed in 5.90 s.
- Scoring module statement coverage: 29/29, 100%, in the 35-test run.
  This is not full-repository coverage. The full suite was not rerun.
- Tests cover short request windows, all-masked episodes, mixed padding,
  2,048-event sequences, final voltage/tick parity, unchanged parameters,
  changed readout parameters, independent sessions, decoded real-cell score,
  ranking, and mutation release. Numerical comparisons use 1e-12 tolerances;
  real-cell activity and final fixture state are compared exactly.

The first scan-buffer candidate reduced allocation but slowed compilation and
did not improve warm execution. It was rejected in favor of the simpler
readout-hoisting change. This reinforces the need to measure compilation and
steady execution separately. Coverage initially used an import name that did
not match pytest's module path; file-directory coverage fixed that measurement.

Further work required for the 30-second goal includes independently measuring
training/compilation and topology rebuilds, then changing their architecture
without reducing the workload or weakening learning/physical contracts. This
patch does not establish that such a target is feasible.
