# Parallel H01 tree contraction prototype

The full 104-cell 30-second evolution objective remains unchanged and unmet.
Profile a same-system solver using independent leaf and unary-node Schur
eliminations, reducing long unary chains in parallel. Preserve dt and physical
equations. This is an evidence-only prototype, not default activation.

At each stage select nonadjacent nodes having at most one child, excluding the
root. Update both surviving neighbors and their connecting coefficients by
the exact scalar Schur complement. Retain eliminated rows for reverse-stage
back substitution. Construct schedules once on CPU; JIT the numerical solve.
Compare with existing implicit tree solves on saved anatomical trees using
well-conditioned synthetic coefficients. Require small dense oracles before
timing. Production integration additionally requires transpose/gradient,
batching, conditioning, and real physical finite-window learning qualification.
