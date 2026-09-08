# Frozen density candidate retains subthreshold shape

All three predefined conditions pass: no complete main-pulse spike,
voltage at 2019 ms below voltage at 1120 ms, and voltage at 2099 ms
below voltage at 1019 ms. This is the previously used calibration input.

| Time (ms) | Model voltage (mV) | Human voltage (mV) | Error (mV) |
| --- | --- | --- | --- |
| 1019 | -83.989036 | -84.500000 | +0.510964 |
| 1025 | -81.314657 | -82.093758 | +0.779100 |
| 1030 | -79.928931 | -80.687500 | +0.758569 |
| 1040 | -77.935828 | -78.531250 | +0.595422 |
| 1070 | -74.940582 | -75.562500 | +0.621918 |
| 1120 | -73.793662 | -74.187500 | +0.393838 |
| 1520 | -74.006790 | -75.687500 | +1.680710 |
| 2019 | -74.008588 | -75.687500 | +1.678912 |
| 2025 | -76.695117 | -78.125000 | +1.429883 |
| 2030 | -78.101214 | -79.656250 | +1.555036 |
| 2050 | -81.611125 | -83.281250 | +1.670125 |
| 2099 | -84.385981 | -85.812500 | +1.426519 |

The baseline error is +0.510964 mV. From that baseline, the model rises
10.195374 mV at 1120 ms and 9.980448 mV at 2019 ms, compared with
human changes of 10.3125 and 8.8125 mV. The return is -0.396945 mV,
compared with -1.3125 mV in the human trace. Correct shape does not
establish correct amplitude or a full physiological fit.

All physical metadata match the frozen active candidate. Only sweep,
waveform hash, sample count, and integration time differ. The source input
hash matches. Raw mappings are exact, input plateau error is zero, all
arrays are finite, and the trace ends at 2100 ms. The
[full result](h01-l2-density130-subthreshold-result.json) retains each
sample, deflection, and residual. Candidate-specific numerical checks
remain separate. No reserved response was used.
