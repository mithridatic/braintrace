# Axonal calcium-entry scaling candidate

The source optimizer fixes gamma_CaDynamics at 0.0005. It does not fit
this value to the human trace. Test 0.002 in the axon only on the slow-
removal, opening-only candidate. This is an inferred override of a frozen
source parameter, not a measured human calcium fraction.

Gamma multiplies the ionic-current contribution to free calcium in the
source pool equation. Keep pool depth, resting calcium, channel densities,
and axonal removal time 1000 ms unchanged. Keep the established sodium,
Kv3 phase, and somatic Ca_LVA settings unchanged. Use mesh factor 9 and
CVode tolerance 1e-10 at both calibration inputs.

Use the first two positive events with upward crossings at or after 1000 ms
for the late-interval comparison. The candidate improves this direct target
if its absolute error against the corresponding human pair is smaller than
the gamma-0.0005 control error. Missing pairs make the comparison invalid.
Report first-onset shift against the control and whether its magnitude is
less than 0.1 ms. Retain first shape and all later events, not just the pair.

Record calcium and SK state at each soma crossing and check the axonal SK
current identity. Do not claim exclusive SK mediation: gamma also changes
the calcium-dependent reversal voltage. The factor is inferred and needs
independent physiological support before a human-mechanism claim.

The driver option defaults to the source value when omitted and records
the override. Accept finite gamma in [0,1], including zero for a diagnostic
removal of the current-driven source term. Do not read the reserved input
or promote a candidate from this experiment.
