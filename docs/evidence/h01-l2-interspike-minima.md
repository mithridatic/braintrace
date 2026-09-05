# Direct recovery observations

The [datum record](h01-l2-interspike-minima.json) gives the lowest sample
after each peak and before the next upward -20 mV crossing. No trace
alignment or smoothing is applied. Values refer to the first three
event pairs during the main pulse.

| After event | Human minimum, mV | Source minimum, mV | Bias-run minimum, mV |
| --- | ---: | ---: | ---: |
| 1 | -69.562504 | -71.079995 | -71.076007 |
| 2 | -67.093754 | -71.518922 | -71.506789 |
| 3 | -70.218755 | -71.527851 | -71.526800 |

The human second minimum occurs 2.26 ms after its peak, and the next
onset interval is 220.403496 ms. The bias-run second minimum occurs
3.228125 ms after its peak, with a 123.774799 ms next-onset interval.
Thus, this model reaches a more negative minimum yet returns to the
next spike sooner. Matching the minimum alone cannot qualify this
recovery trajectory. The full voltage and state history remain relevant.

These observations do not identify a channel cause. In particular, they
do not justify increasing all outward conductances solely because the
model has extra spikes. The recovery duration and the rapid return after
a spike are separate direct targets. A mechanism intervention must test
its effect on both. Source spatial refinement and intervention-specific
numerical robustness remain outstanding.
