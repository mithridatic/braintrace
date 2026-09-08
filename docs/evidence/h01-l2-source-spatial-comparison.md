# Layer-2 source spatial check

Tripling every source section count preserves seven complete events,
but rejects the declared numerical gate. Maximum ordinal onset change
is 0.136042708 ms, above 0.100 ms. Maximum sampled peak change is
0.111100344 mV, above 0.100 mV. Duration change is at most 0.000607428 ms,
within the 0.010 ms limit. No structural audit reason invalidates the
comparison. The [full record](h01-l2-source-spatial-comparison.json)
retains every paired residual.

The baseline parent fields come from the separate source setup replay;
the finer run records its parents directly. This distinction remains
in the report. Region properties, geometry, input, fitted genome, and
time step remain fixed. All section segment counts triple.

The next comparison uses factors three and nine with the same limits.
It passes: onset change is at most 0.010970713 ms, peak change
0.011364285 mV, and duration change 0.000079823 ms. Both runs retain
seven events and the structural checks pass. The
[complete comparison](h01-l2-source-spatial-comparison-space9.json)
supports selecting factor nine for the mechanism test. It does not
validate the human response. The spatial audit has 16 passing tests
and 100% statement coverage, including the successive tripling path.

The [calcium-removal split](h01-l2-calcium-removal-result.md) has run. Its fit-copy
helper passes 12 tests with 100% statement coverage. The tests verify
that only the identified somatic decay changes, that the source remains
unchanged, and that malformed targets and factors are rejected. This
implementation evidence is separate from the mechanism result. The
completed split uses the selected factor-nine mesh.

The driver now accepts the opt-in calcium-decay factor and records its
source and applied values. The real retrieved fit was checked: factor 1
preserves all values, and factor 2 changes only the somatic decay from
494.01955262344603 to 988.0391052468921 ms. Eight driver guard tests and
the twelve fit-copy tests pass. These checks do not run the intervention.
The factor-nine spatial simulation has completed. Its source response
is the control for the declared calcium-removal intervention.
