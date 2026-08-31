# BRA-97 performance closeout

## Outcome

The retained performance work materially speeds both the pp-prop application
path and its largest Example 21 test gate without removing tests, weakening
assertions, changing public schemas, or changing the learning objective.
Measurements below are copied from the benchmark specifications produced with
the changes; they are not re-measured in the current execution container.

## Application measurements

### Sparse pp-prop at 128K neurons

The three-seed CPU benchmark used 131,072 neurons, recurrent degree 8, batch
32, and 30 timesteps. It separated compilation, warmed updates, validation,
and total worker time.

| Measurement | Before | After | Result |
| --- | ---: | ---: | ---: |
| Warm training update | 5.244 s | 4.192 s | 1.25x faster |
| Validation pass | 9.353 s | 6.822 s | 1.37x faster |
| Total worker | 51.6 s | 41.6 s | 1.24x faster |
| Peak process-tree RSS | 1.771 GiB | 1.815 GiB | 2.5% higher |

The adopted implementation removes a general hidden-Jacobian contraction and
avoids materializing a broadcast identity. All three seeds retained the same
six-digit loss trajectory and exact final accuracy. The CPU improvement does
not generalize to CUDA, where both forms were already at the bandwidth
roofline; no GPU speedup is claimed.

### Example 21 training and Docker execution

A short RTX 3080 Ti Laptop benchmark at matched batch 16 reduced a 640-episode
run from 29.385 s to 22.538 s (1.30x). Raising the batch to 48 delivered 40.137
episodes/s versus 21.780 episodes/s for the legacy batch-16 pipeline (1.84x),
with a matched final-loss difference of 4.77e-7.

The canonical Docker workload used 1,024 neurons, batch 32, 130 updates, latent
horizon 390, chunk size 5, and the full 400-task evaluation. Its three clean
warm runs were 129.900 s, 132.465 s, and 133.085 s, for a 132.465 s median
against the supplied 198.5 s chunk-1 baseline: 33.3% lower wall time. Peak
physical device use was 4,569,694,208 bytes (26.6% of the 16-GiB device).

The full effort-390 metrics and every candidate grid were unchanged. A later
single post-cache run took 111.495 s (1.78x faster than the baseline), but it is
reported only as a single rerun, not as a replacement for the three-run median.

## Test-suite measurements

The six-worker Example 21 and latent-workspace gate originally collected 1,953
tests and took 869.3 s, with six understood fixture/isolation failures. The
optimized gate collected 1,972 tests and passed every test in three fresh
processes at 335.89 s, 334.91 s, and 321.48 s. The 334.91 s median is 61.5%
lower wall time, or 2.60x faster, while retaining the original selection and
adding 19 regressions.

The corresponding repository coverage run passed 3,064 tests and reported 95%
coverage. The changes responsible were:

- batched NumPy episode encoding with byte-for-byte scalar-oracle checks;
- one gradient transform per finite-window oracle call instead of one per
  equal-shaped chunk;
- grouped reuse of expensive stateful fixtures with explicit state reset;
- load-group xdist scheduling for fixture consumers; and
- a measured 40-test JAX cache-clear cadence, which reduced the recorded full
  suite from 833 s and 22.81 GiB without clearing to 655 s and 13.65 GiB.

The follow-up Gate C2 profile found a representative validator call spending
about 37.5 s checking 2,328 records of 512 scalars. Its strict schema and
aggregate checks are now vectorized with NumPy, and malformed bool/string,
negative, and stale-aggregate cases remain covered. No post-change end-to-end
number for the later 2,208-test selection is available, so this closeout makes
no speedup claim for that revision.

## Reproduction and limitations

Detailed commands, versions, warm-up policy, per-seed data, device conditions,
and correctness gates are retained in:

- `docs/specs/2026-08-09-turboquant-sparse-benchmark-results.md`
- `docs/specs/2026-08-19-example21-training-throughput.md`
- `docs/specs/2026-08-19-example21-pytest-throughput.md`
- `docs/specs/2026-08-20-example21-docker-throughput.md`
- `docs/specs/2026-08-22-pytest-throughput-v2.md`

The current Paperclip execution image has Python 3 but does not contain JAX,
brainstate, or pytest, so it cannot independently rerun the scientific or test
benchmarks. Cold compilation and warm execution remain separately labeled in
the source reports. GPU and CPU results are not composed across devices.

## Edge cases and regression tests

The retained tests cover non-finite and malformed evidence, exact endpoint
geometry and dtype, stale aggregate rejection, byte-identical episode encoding,
chunk-1 versus chunk-5 ordering and masking, deterministic loss streams,
parameter closeness, source pins, and JAX cache clearing disabled or set to a
different cadence. Relevant additional tests for future changes are a fresh
three-run 2,208-test gate measurement and worker-RSS sampling under the current
dependency matrix.
