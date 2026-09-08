# Inhibitory reference compatibility

The existing NEURON references do not satisfy the same early spike-time
limit. Over 270–329.5 ms, the all-region x9 reference has its eighth spike
2.918579 ms earlier than the frozen axon2187 reference. Both have eight
events; peak and width comparisons pass, but onset comparisons fail.

This does not prove a mesh cause. Mesh, duration and probe settings differ,
and the older report has no mechanism-library hashes. The physical parameter
metadata alone cannot prove identical compiled mechanisms.

BrainCell agreement with fixed-step NEURON on the x9 mesh is evidence for
that transfer only. It does not establish mesh convergence, agreement with
the frozen reference, the full firing response, or human validity. Preserve
these separate requirements when interpreting the pending half-step audit.

[Events, differences and hashes](h01-i-reference-mesh-sensitivity.json).
