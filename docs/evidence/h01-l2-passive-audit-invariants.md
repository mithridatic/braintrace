# Passive settings are part of numerical comparison identity

Four regression cases showed that temporal and spatial audit helpers
could return supported when the new passive-parameter or reversal-shift
metadata differed. The recorded voltage fixtures were unchanged, so the
old checks mistook a changed physical setup for a numerical comparison.

Both helpers now reject changed or one-sided missing
applied_passive_parameters and leak_reversal_shift_mv fields. Historical
pairs with neither field retain their original metadata scope; the fix
does not reconstruct missing source evidence. Forty-three tests pass,
with 100 percent statement coverage of the two numerical audit modules.

The error arose because the driver gained a physical parameter without
extending the comparison invariants in the same change. Future physical
parameters must enter those invariants before comparison results are
used. These audit tests do not validate human physiology or the still
pending reversal-intervention result.
