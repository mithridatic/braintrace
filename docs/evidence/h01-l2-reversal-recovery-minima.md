# Recovery depth and timing are separate direct targets

For each complete interspike interval, select the lowest recorded voltage
after the preceding -20 mV fall and before the next -20 mV rise. The
earliest sample is used on an exact tie. Preserve the actual sample time.

| After spike | Model minimum (mV) | Human minimum (mV) | Model minus human (mV) |
| --- | --- | --- | --- |
| 1 | -71.120263 | -69.562504 | -1.557759 |
| 2 | -71.450100 | -67.093754 | -4.356346 |
| 3 | -71.509808 | -70.218755 | -1.291053 |
| 4 | -71.533535 | -70.281254 | -1.252281 |

The minima occur 2.400150, 2.482735, 2.520447, and 2.493664 ms after
the model's preceding fall. The respective human delays are 1.232581,
1.442041, 1.318438, and 1.335806 ms. The model minima are both more
negative and later relative to that crossing.

The [direct observations](h01-l2-reversal-recovery-minima.json) retain
indices, physical times, interval boundaries, and source hashes. These
are distinct event-relative locations, not same-time voltage errors.
Adaptive model and fixed human sampling differ. A
[numerical sensitivity review](h01-l2-reversal-minima-numerical-review.md)
retains their changes under the existing tolerance and mesh comparisons.
No prospective extrema-specific acceptance threshold was set.

Use the observations when assessing the pending calcium-removal split.
An improved spike interval does not establish the correct recovery depth
or speed. The extrema alone do not identify calcium, SK, or another
current as the cause. No new prospective hypothesis is assigned to this
descriptive review.
