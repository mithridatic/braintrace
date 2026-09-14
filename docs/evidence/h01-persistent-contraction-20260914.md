# Persistent contraction rejected for production

The complete 104-cell command in 30 seconds remains unachieved. This bounded
experiment does not change production behavior, dt, pp-prop or training work.

A single Triton program performed all Schur elimination and substitution stages,
using flat indices, 256-element chunks, four warps, independent scratch arrays,
and barriers between stages. It passed four GPU tests with pytest -n 2 in 8.48 s:
root-only, chain, branched tree and 1,025-node star, with two RHS, unequal edge
coefficients, a nonzero independent sentinel and input nonmutation.

On the now-idle Vast RTX 4090, five synchronized warm samples per saved anatomical
tree gave these medians (milliseconds):

| Nodes | Production grouped | Persistent | Slowdown |
| --- | ---: | ---: | ---: |
| 64,047 | 0.597 | 2.706 | 4.53x |
| 21,334 | 0.869 | 1.193 | 1.37x |
| 26,055 | 0.885 | 1.486 | 1.68x |
| 20,958 | 0.900 | 1.149 | 1.28x |

Maximum absolute difference was 5.56e-17 on these synthetic diagonally dominant
systems using real tree topologies. These are isolated solves, not episodes or
physical-learning qualification. No derivative or physical-learning gate was
run because the performance prerequisite failed. The experimental kernel is
retained only to reproduce this negative result; it is not production-qualified.

The single-program approach sacrifices too much parallelism. Further work should
retain parallel processing for wide stages and investigate remaining startup,
compilation and episode-level costs. No full-runtime estimate can be inferred
by dividing the previous multi-hour estimate by the four-cell event speedup.

The old full evolution process (PID 805584) was still running after 2 h 46 min,
with no candidate files written. It was terminated with explicit user permission,
and absence from the process table was verified. Its output directory remains.
