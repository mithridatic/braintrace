# Larger distributed Ih changes recovery but fails the combined shape

The combined sufficiency prediction is rejected. Increasing total Ih by
75 with location and original kinetics fixed produces a slight return
below the starting voltage, but does not restore adaptation during the
selected constant-input interval. No complete pulse spikes occur.

| Time, ms | Human deflection, mV | Candidate deflection, mV |
|---|---:|---:|
| 1019 | 0 | 0 |
| 1120 | 10.3125 | 10.645994 |
| 2019 | 8.8125 | 10.692531 |
| 2099 | -1.3125 | -0.020329 |

The candidate starting voltage shifts upward by 3.215080 mV from the
location-only control and is 3.192793 mV above the human sample. Thus,
absolute late voltage error remains 5.072824 mV even though the
stimulus-driven deflection is smaller. Do not confuse these measures.

The [full audit](h01-l2-distributed-ih75-result.json) retains all direct
samples, baseline shift, and each condition. All three Ih densities scale
by 75 within the stated relative tolerance. Integrated maximum conductance
is 15.429426 nS. Non-Ih parameters, geometry, kinetics source, bias, and
solver settings remain consistent. Raw mappings, current plateaus, and
endpoint pass. Candidate-specific numerical qualification is incomplete,
including for the small post-pulse sign change.

This is a conditional model effect at the tested settings. It does not
establish a human HCN density or justify compensating the baseline with
unmeasured injected current. The original recorded bias stays fixed.
The candidate is not promoted; resting voltage, adaptation, recovery,
and active responses must be addressed together.
