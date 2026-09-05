# Bounded DHS solver graph

The measured H01 run has reached native compilation of `jit_scan`. The installed
DHS forward-elimination kernel uses a Python loop over tree levels. Abstract
traces contain 31 top-level equations per level in the tested chain cases.
This is a candidate cause of compilation cost. It is not yet an isolated
full-run timing result.

## Implementation boundary

Add an opt-in solver adapter in this worktree. Keep the staggered voltage and
channel update order, DHS equations, units, precision, and static topology.
Do not modify the installed BrainCell package or silently replace its default.
Use `brainstate.transform.scan` for forward elimination and back substitution.
Prepare padded level indices from static topology once. Mask padding so it
cannot change diagonal or right-hand-side values. Preserve the sentinel row.
Do not convert physical arrays to NumPy during numerical updates.

## Required checks

- Reproduce graph expansion with the existing kernel before changing it.
- Compare forward elimination and back substitution with the existing kernel
  on chains, branching trees, multiple children per parent, and batches.
- Check dimensional quantities and plain arrays. Check empty edge schedules.
- Compare the resulting solution with an independent dense linear solve on
  small well-conditioned systems.
- Confirm that graph size stays bounded as the number of levels increases.
- Compare cell voltage traces with the original staggered solver under the
  same input and initial state before using the adapter in the measured pair.
- Keep the current real H01 run intact as baseline evidence. No change to
  conductance, gate laws, morphology, or time step belongs in this solver test.

Only numerical equivalence permits use in the measured circuit. Faster tracing
alone does not qualify the solver or the biological model.
