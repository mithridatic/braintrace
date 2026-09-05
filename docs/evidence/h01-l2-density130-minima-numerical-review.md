# Control recovery minima persist under numerical refinement

This is a retrospective check of four individual minima from already
completed runs. It does not create a prospective extrema acceptance gate.

Tightening CVode tolerance changes minimum voltage by at most
0.000002658 mV and minimum delay after the preceding falling crossing
by at most 0.001622412 ms. Comparing mesh factors 3 and 9 changes minimum
voltage by at most 0.001329462 mV and that relative delay by at most
0.001344444 ms. The full clock-time change can also contain a spike-onset
shift; it is kept separately in the evidence.

These changes are much smaller than the four human minimum-voltage errors
of -2.094337, -4.966488, -1.914093, and -1.872330 mV. The too-negative
control minima persist at both tested refinements. This supports the
observation to be diagnosed; it does not identify its unique cause or
qualify the pending Kv3 intervention.

The [full review](h01-l2-density130-minima-numerical-review.json) retains
every minimum, voltage difference, relative delay difference, clock-time
difference, and source trace hash. Each minimum is the earliest sampled
minimum strictly between adjacent spike boundaries.
