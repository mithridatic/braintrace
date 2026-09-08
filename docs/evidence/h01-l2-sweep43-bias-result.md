# Recorded bias does not restore the subthreshold response shape

The bias sufficiency prediction is rejected. Adding the recorded
-3.711859 pA bias nearly removes the initial voltage offset: the model
is 0.015036 mV below the human at 1019 ms. It still fails both declared
shape conditions and produces no pulse spikes.

From the 1019 ms sample, model voltage rises 12.088560 mV at 1120 ms
and 12.932369 mV at 2019 ms. The human rises 10.3125 then 8.8125 mV.
At 2099 ms the model remains 1.432744 mV above its starting sample,
while the human is 1.3125 mV below it. Late deflection error remains
4.119869 mV. Thus, the omitted bias is insufficient to explain the
adaptation and recovery mismatch.

The [full audit](h01-l2-sweep43-bias-result.json) retains each direct
sample and condition. Model, source build, geometry, solver, and waveform
identity checks pass. Applied-current plateau error is zero under the
unchanged 1e-12 nA limit. Raw adaptive sample mapping and endpoint pass.

The [initial audit](h01-l2-sweep43-bias-audit-initial.json) remains saved.
It incorrectly added bias to the float32 command without first promoting
the array, causing rounding in its expected-current calculation. The
corrected audit uses the driver's float64 conversion. A regression test
demonstrates the rounding and checks exact promoted addition. Future
input audits must compare the applied numerical convention explicitly.

This is a conditional model sufficiency result, not identification of a
unique human current pathway. Candidate numerical checks at this input
remain necessary. Current-path diagnosis can now proceed with the
recorded bias included and without confusing baseline offset with the
remaining time-dependent response error.
