# Parallel tree contraction investigation

The full default 104-cell evolution in 30 seconds is **not achieved**. This is
an opt-in solver prototype; production still uses the qualified GPU tree solver.

The prototype eliminates independent leaves and unary nodes using scalar Schur
complements. A unary elimination updates both surviving neighbors and the
connecting coefficients. Reverse-stage substitution reconstructs eliminated
voltages. Static tree schedules are built once; numerical stages are JIT
compiled. The physical probe substitutes only the existing implicit solve and
transpose callbacks, retaining equations, float64, dt=0.000625 ms, and all
160 substeps. Its report records hashes of the experimental scripts.

## Numerical evidence

Three small dense fixtures (chain, branching, star) passed forward solves and
both `jacfwd`/`jacrev` for diagonal, RHS, lower and upper coefficients on Vast
and local Windows CPU JAX. Thirteen pytest stress cases passed with `-n 2` in
6.83 s on Vast, covering the single-node case, coefficient scales through 1e6,
unequal lower/upper coefficients, residual checks, and a 2,048-node chain.

On four exported anatomical trees with well-conditioned synthetic coefficients:

| Nodes | Contraction stages | Current GPU solve | Contraction solve |
| ---: | ---: | ---: | ---: |
| 64,047 | 20 | 9.223 ms | 0.625 ms |
| 21,334 | 18 | 4.913 ms | 0.760 ms |
| 26,055 | 18 | 5.330 ms | 0.936 ms |
| 20,958 | 17 | 6.465 ms | 1.695 ms |

Maximum solution error was 1.388e-16. Schedule construction took 0.049–0.195 s
per tree and cold compilation/execution took 1.099–1.566 s. These are small
samples on a shared GPU, not isolated statistical benchmarks.

## Real physical forward evidence

The four anatomical cells have 75,605 compartments. A bounded forward probe
completed with finite output. Warm event execution fell from 4.523 s in the
preceding geometry-cache profile to 0.577 s (7.8x). Cold event time increased
from 14.279 to 21.577 s. Construction/initialization remained 32.795/21.625 s.
Maximum voltage difference over the two recorded events was 1.814e-10 mV.
Peak host RSS was 2,813,872 KiB; this is not a demonstrated memory improvement.

## Real one-event learning evidence

The capped full probe completed two sparse Muon updates with finite optimizer
state. Warm update time decreased from 24.862 to 3.295 s (7.5x), while first
compilation/update increased from 107.653 to 139.897 s. Sparse graph construction
increased from 3.356 to 8.071 s. Compared with the preceding GPU solver probe,
maximum differences were 1.708e-12 in losses and 1.687e-13 in gradient norms.
This uses synthetic squared-logit losses, not complete ARC episodes.

Peak host RSS increased from 4,728,436 to 5,579,628 KiB (about 18%). Consequently
the unrolled prototype does not yet meet the requested memory improvement.
Reducing compilation/code size while retaining the speedup is an activation
gate, alongside full parameter/optimizer/eligibility parity. A second forward
sample in this run measured 0.601 s warm and 12.884 s cold, illustrating the
variability of cold compilation and the consistency of the warm gain.

## Corrections and activation gate

The standalone evidence test initially used a package-relative import; this
directory is not a package. It now uses a sibling import. The single-node
fixture also needed an explicitly integral empty index array. Neither failure
was a solver numerical failure. Keep empty and standalone cases in the tests.
Docstrings were expanded after the recorded runs; numerical source is unchanged.

Before production activation, require real finite-window learning parity,
implicit transpose and batching coverage, robust schedule validation, and
bounded cache ownership. Raw `.prof` files remain on Vast; synchronized text
profiles and JSON evidence are retained here. The synthetic tests and short
physical probes do not establish full ARC quality or evolution completion.
