# H01 scoring performance

Authorized by the user's autonomous performance request, including remote
profiling, tests with xdist, and publication after verification.

Measure the current scoring path on Vast.ai with cProfile and synchronized JAX
timings. Preserve dt, precision, all physical events, episode reset, unchanged
31-request logits, activity reduction, parameters, and optimizer state.

Replace the full event-by-readout temporary with the smaller feature output
of the existing BrainState loop. Apply the readout once to the final 31 feature
vectors. Retain the existing activity reduction order. Test against the original compiled full-output implementation
for short sequences, masked padding, and normal request windows; compare final
state and numerical outputs. Use lightweight stateful fixtures for these
execution contracts and retain the existing real-cell integration test.

Record cold and warm timing and memory separately. A bounded probe does not
establish that the default full-corpus 104-cell evolution finishes in 30 seconds.
Do not reduce corpus, updates, topology operations, or physical resolution to
claim that target. Preserve unrelated remote jobs and local manifest changes.

Reuse the compiled multi-query scoring callable on its owning runtime. A new
runtime receives a new callable; parameter State updates remain dynamic inputs.
Verify repeated scoring does not retrace and observes changed readout weights.
The cached callable is runtime-only and must not enter checkpoint metadata.
