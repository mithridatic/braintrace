# Separate Kv3 opening and closing

The half-time-constant intervention speeds both opening and closing.
Create a new isolated diagnostic cache with an optional closing-time
override. When m is above mInf, use that closing factor. Otherwise use the
existing time factor. A disabled override must retain the existing equation.
The equilibrium and density stay unchanged. Hash source changes.

Verify 16 fixed-voltage cases at -80 and +20 mV, initial m 0 and 1,
base time factor 1 and 0.5, and closing factor 1 and 0.5. At exactly
0.5 ms, require gate error below 1e-8 against the analytic exponential
and voltage error below 1e-8 mV. This is a diagnostic piecewise model,
not a measured human rate law.

At 0.19 nA on the half-Ca_LVA candidate, run opening-only acceleration
(base factor 0.5, closing factor 1) and closing-only acceleration (base
factor 1, closing factor 0.5). Keep source Kv3 density and all other
parameters fixed. Use mesh factor 9 and CVode tolerance 1e-10.
Also reproduce the existing both-fast run in the new build with its
closing override disabled. Require identical time and voltage arrays.

Test the claim that opening-only acceleration reproduces the both-fast
advance at falling -65 mV within 0.05 ms, while closing-only acceleration
advances that crossing by less than 0.05 ms relative to source speed.
Require two initial positive spikes and a falling crossing. If these
are absent, the attribution test is invalid. Retain minima, first shape,
each later event, and each interval. This claim concerns the early return,
not exclusive mediation of the later train. No parameter is promoted.

## New within-build factorial after failed exact control

The old-build versus new-build exact time-array check failed. The original
attribution experiment is invalid under its declared control rule. Retain
that result; do not relax the exact criterion after observing the data.

Perform a separate four-case comparison entirely within the phase build.
Use explicit (opening, closing) factors (1,1), (0.5,1), (1,0.5), and
(0.5,0.5). The two mixed cases already exist in this build. Run the two
remaining reference cases before evaluating the same early-return claim.
All four use the same compiled library, equations, input, geometry, and
other parameters. The analytic phase checks validate the parameter action.
Do not claim equivalence to the old build from this comparison. Keep its
source hashes, within-build attribution, and numerical limits separate.
