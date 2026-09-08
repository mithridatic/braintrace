# H01 I source inactivation-time comparison

The complete published source profile gives a positive response on the H01
geometry where the selected candidate does not. Do not attribute this result
to one parameter: the complete profiles differ in several places.

Test one source restoration: change the candidate NaTg h closing-time factor
from 0.15 to the published source value 1.0 in all NaTg regions. Preserve h
recovery, equilibrium slope, all conductances, other channel laws, geometry,
input, and initial voltage. Gates remain dynamic; this is not an availability
clamp. Record the override separately from the frozen profile metadata.

Use the original 1 nA pulse from 2 to 5 ms, dt 0.005 ms and maximum CV length
10 um. Save voltage and h. Compare each positive crossing and peak against
the candidate. A positive excursion is a diagnostic result, not human waveform
qualification or permission to promote the override as a new default.

## Numerical and circuit checks

Repeat the single-cell response with dt 0.000625 ms and maximum CV length
2.5 um. Keep the source closing-time restoration fixed. Record each crossing,
peak, and recovery. This joint refinement checks response robustness; it does
not separate temporal from spatial error.

If positive output persists, run the real H01 pair with this explicitly
recorded diagnostic override in I only. Keep the E candidate unchanged.
Use I-only versus disconnected contacts, the same 1 nA soma pulses in both
cells, and the same illustrative 0.5 ms delay and 0.02 uS inhibitory weight.
Require actual I emission, zero disconnected conductance, matching E voltage
before delivery, and direct E response after arrival. Missing E spikes alone
are insufficient without these delivery controls. This tests functional
inhibition, not human validity or measured H01 wiring. Defaults stay fixed.
