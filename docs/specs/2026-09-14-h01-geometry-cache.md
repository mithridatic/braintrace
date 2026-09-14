# H01 geometry reuse

The full default 104-cell evolution must still complete within 30 seconds;
this change addresses measured repeated construction, not a substitute goal.
Keep dt, spatial bounds, channel placement, and numerical operations unchanged.

Replace the process-global geometry dictionary keyed by bounds identity with
one entry owned by a morphology. Compare immutable branch identities and
topology plus exact bounds contents. Retain branch owners so their identities
cannot be recycled while cached. Clones may reuse the entry because they share
immutable branches, but modified topology/bounds must miss. Releasing the
morphology must release its geometry rather than retain it globally.

Require regressions for equivalent separately allocated bounds, changed bounds,
topology changes, clone reuse, and cache lifetime, alongside existing physical
construction parity checks. Profile the real four-cell construction and init
on Vast with a bounded run. Do not infer full-command success from that probe.
