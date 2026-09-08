# Local current records preserve the original response

The instrumented bias-included sweep-43 run exactly reproduces all
517,576 time, voltage, and applied-current samples of its uninstrumented
control. All 15 extra arrays exactly match the retained raw samples at
the common right-limit indices. Driver and recording checks pass 28
tests. These checks do not constitute coverage of the NEURON driver.

The [observations](h01-l2-subthreshold-current-observations.json) retain
each current, Ih and Im gates, and calcium at the 12 specified times.
Currents are local densities at soma(0.5), not whole-cell totals. Values
at selected times use linear interpolation.

| Time, ms | Ih gate | Ih current, mA/cm2 | Im current, mA/cm2 |
|---|---:|---:|---:|
| 1019 | 0.075410 | -0.000103541 | 0.000000338 |
| 1120 | 0.034411 | -0.000032794 | 0.000005772 |
| 2019 | 0.021520 | -0.000019877 | 0.000007077 |
| 2099 | 0.045879 | -0.000060710 | 0.000000483 |

Ih availability decreases during the depolarizing pulse and has not
returned to its initial level at 2099 ms. Its inward current also remains
smaller in magnitude. Voltage and gate state both affect that current;
these trajectory samples do not isolate their separate effects.
The local Im outward current increases during the pulse but is small
relative to the local leak and Kv3 currents at the sampled voltages.
Small current magnitude does not by itself establish weak sensitivity.

These observations prioritize an Ih intervention for the adaptation and
return mismatch. They do not establish its sufficiency, a human HCN
conductance, or complete current balance. Other cell regions and axial
flow are not measured here. Keep the causal account conditional until
an isolated intervention tests the proposed effect. No fitted parameter
or reserved-input status changed in this measurement step.
