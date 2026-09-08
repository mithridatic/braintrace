# Ih sufficiency split at subthreshold input

Double only soma / Ih / gbar_Ih in the bias-included sweep-43 midpoint.
Keep kinetics, calcium factor 1.5, source sodium, mesh factor 9, CVode
tolerance 1e-10, initial voltage, temperature, waveform, and endpoint
2100 ms fixed. Retain local currents and their original raw records.
The density multiplier is inferred, not measured in this human neuron.

Before seeing the response, require voltage at 2019 ms below voltage at
1120 ms, voltage at 2099 ms below voltage at 1019 ms, and no complete
-20 mV pulse spikes. These are the same necessary shape conditions as
the bias test. Report all 12 specified voltage observations, deflections,
and human errors even if they worsen. Passing these conditions does not
qualify the waveform. A failure rejects the tested factor as sufficient,
not all possible Ih contributions.

Verify exactly one genome row changes by factor two. Preserve original
fit files and record the multiplier. No other fitted parameter or reserved
input may change. Verify the helper's isolation, default identity, and
invalid-factor and invalid-target handling before the run.
