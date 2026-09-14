# H01 geometry cache qualification

The complete default 104-cell evolution under 30 seconds remains **not achieved**.
This change reuses identical geometry across declaration and initialization;
dt, spatial bounds, equations, precision, and training workload are unchanged.

The former global cache used bounds object identities without retaining those
objects. New regressions reproduced stale geometry after changing spatial
bounds (35 compartments incorrectly retained), and geometry retained after its
morphology was released. The replacement stores one entry on each morphology,
checks actual bounds and branch topology, and retains immutable branch owners
to prevent address reuse. Clones sharing those branches can reuse the entry.
Changed bounds/topology miss; dead morphologies no longer leave geometry in a
global dictionary. Other construction caches are outside this change.

## Vast.ai evidence

Same four anatomical cells and 75,605 compartments as the preceding GPU solver
probe, unchanged dt=0.000625 ms and float64. cProfile and synchronized phase
timings on the RTX 4090, using `h01_event_profile.py --forward-only` with a
150-second external cap. The process completed successfully.

| Phase | Previous GPU solver probe | Geometry cache |
| --- | ---: | ---: |
| Construction | 37.319 s | 31.764 s |
| Initialization | 30.223 s | 20.577 s |
| Cold event | 14.456 s | 14.279 s |
| Warm event | 4.479 s | 4.523 s |

Construction plus initialization decreased from 67.541 to 52.341 seconds
(22.5%). These are small samples with the other session running; some tests
also overlapped the new profile. This is not an isolated statistical benchmark.
Both recorded voltage vectors match exactly (maximum error 0 mV). Peak host
RSS was 2,628,968 KiB; the preceding probe included learning, so its peak is not
a comparable RAM baseline. The lifetime regression verifies removal of this
specific retention path, not total-process memory savings.

## Validation and remaining scope

- Before implementation: two cache regressions failed on Vast.
- Construction gate: 13 passed in 16.10 s with `pytest -n 2`.
- Construction, GPU solver/learning, and network-step gate: 29 passed in
  37.91 s with `pytest -n 2`, filesystem coverage enabled. All new cache
  statements executed; whole construction-module coverage was 87.09%.
- Edge cases cover independently allocated equal bounds, changed bounds,
  topology growth, clone reuse, owner release, physical discretization parity,
  and sparse learning parity. Full ARC quality remains unmeasured.

The correction is to cache by validated contents and explicit ownership rather
than treating recyclable object addresses as durable keys. Full construction
still exceeds 30 seconds on four cells alone; no full-evolution completion
claim follows. Raw profiles remain on Vast under `/tmp/h01-geometry-event-profile`;
text profiles and the report are synced in this directory's corresponding folder.
