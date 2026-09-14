# Persistent contraction experiment

The complete 104-cell evolution must finish within 30 seconds. This experiment
tests launch overhead in the existing tree solve, without changing dt, float64,
anatomy, pp-prop, training work, or acceptance criteria.

Run all independent Schur stages and reverse substitution in one GPU program.
Use flat ragged indices and bounded chunks, scratch outputs rather than input
mutation, and barriers between dependent stages. Retain implicit differentiation
at the existing custom linear solve boundary. This is evidence-only until dense
solutions, sentinel behavior, batching, derivatives, physical learning parity,
and useful measured speed pass. A slower prototype must not be enabled.

First compare against the published grouped solver on saved anatomical trees,
with bounded runtime on Vast. Check root-only, chains, branching, shared parents,
nonzero sentinel, unequal edge coefficients and multiple RHS. Run pytest with
xdist. Only a successful prototype warrants a real event/learning profile.
The old long run was stopped with user authorization; its outputs are preserved.
