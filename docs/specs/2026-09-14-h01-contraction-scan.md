# Compact contraction execution

Keep the full 104-cell under-30-second evolution goal intact. The preceding
opt-in contraction prototype accelerated warm physical learning but increased
peak host RAM. Test a fixed-shape compiled scan over its unchanged independent
elimination schedule. Preserve dt, float64, equations, and implicit derivatives.

Eliminated diagonal, RHS, and lower entries remain untouched by later stages;
retain them in the existing arrays. Store each eliminated row's child upper
coefficient in its now-unused upper entry. This avoids stacking whole-array
histories. Pad static schedule indices with a neutral sentinel and drop all
invalid writes. Reverse-scan substitution reads the final retained rows.

Require existing dense/Jacobian and stiffness tests, then anatomical solve
timings. Only run the bounded real learning probe if these pass. Compare
runtime, peak host RSS, losses and gradient norms with the unrolled prototype.
This remains opt-in until full physical state parity and validation gates pass.

The compact probe reduced peak RAM but remained above the production baseline.
Audit process-global discretization retention: use weak values for shared
snapshots, while each live initialized cell retains its current snapshot.
Require collection after declaration invalidation, stable initialized ownership,
and existing physical construction/learning parity before publishing that change.
