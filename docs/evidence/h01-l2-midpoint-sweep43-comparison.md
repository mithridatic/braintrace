# A subthreshold mismatch precedes spike fitting

The fixed calcium-midpoint candidate produces no complete -20 mV events
at the 110 pA command, as in the human trace. This necessary check passes.
The voltage trajectory does not match. The
[direct comparison](h01-l2-midpoint-sweep43-comparison.json) retains all
12 prospectively selected time points and both voltage and deflection
errors. Human values are exact samples; model values use linear
interpolation of the adaptive trace.

The model starts 0.422282 mV above the human 1019 ms voltage. Removing
that offset leaves a 4.1306 mV excess rise at 2019 ms. The human rises
10.3125 mV at 1120 ms, then returns to an 8.8125 mV rise at 2019 ms.
The model instead rises about 12.0907 and 12.9431 mV at those times.
At 2099 ms the human is 1.3125 mV below its initial sample; the model
is about 1.4333 mV above its initial sample. A constant voltage offset
cannot account for these changing signed errors.

Source waveform identity, adaptive raw-sample mapping, and endpoint
checks pass. Input plateau error is zero away from transition boundaries.
The model uses command only; the human recorded bias is -3.711859 pA.
This input distinction remains explicit and requires a separate control
before a unique current-path diagnosis. The result does not identify
leak, Ih, Im, morphology, or any other parameter as the unique cause.
Candidate numerical checks at this input remain open. No human voltage
tolerance was invented, and the reserved input was not accessed.

The next diagnosis must account for subthreshold adaptation and recovery
before further selecting parameters on spike timing alone.
