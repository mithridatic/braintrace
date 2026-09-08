# Kv3 gate speed and conductance are distinct controls

The isolated diagnostic multiplies Kv3 mTau by a factor and leaves its
equilibrium gate unchanged. The source manifest records original and
modified mechanism hashes. Compilation succeeded in the pinned NEURON
container. Eight fixed-voltage opening and closing cases passed against
the exponential solution. The factor-1 build control at 0.19 nA has
exactly the same time and soma voltage arrays as the original control.

The whole-cell intervention halves somatic Kv3 gate time. Conductance
retains its source value. Other settings match the half-Ca_LVA candidate.
Both runs use mesh factor 9 and CVode tolerance 1e-10.

| Input, nA | Advance at falling -65 mV, ms | Onset shift, ms | Minimum, mV | Time from peak to minimum, ms |
| --- | ---: | ---: | ---: | ---: |
| 0.19 | 0.176599 | 0.170335 | -73.328839 | 1.511811 |
| 0.27 | 0.199144 | 0.058693 | -71.893494 | 1.360507 |

Both predeclared return-advance and onset-shift criteria pass. This differs
from doubled conductance, which delayed low-input onset by 6.548475 ms.
The comparison distinguishes conditional responses to the two parameters;
it does not establish complete independence of gate timing and onset.

The faster-gate minima occur earlier but are less negative than the control
minima. First-spike duration errors are -0.052420 and -0.046429 ms.
Peak errors are -1.946439 and -1.497457 mV. Onset errors against the human
trace are +7.647739 and -0.667994 ms. The trains have 47 and 124 events,
versus control 34 and 88 and human 12 and 43. All event times and intervals
are retained; these counts alone do not characterize the error.

This intervention speeds both opening and closing. Do not assign the
faster return or increased firing exclusively to either phase without a
separate test. The candidate is not promoted. Spatial refinement and
human validation remain open. The reserved input was not read.

The [whole-cell audit](h01-pv-kv3-fast-audit.json),
[fixed-voltage cases](h01-pv-kv3-kinetics-clamp.json), and
[build validation](h01-pv-kv3-kinetics-validation.json) retain the evidence.
Run `python -m docs.evidence.h01_pv_kv3_return_audit --fast` to reproduce
the whole-cell comparison. Thirty driver, datum, and crossing tests pass.
The time factor rejects zero, negative, and nonfinite values. Missing
spikes or crossings invalidate the corresponding return comparison.
