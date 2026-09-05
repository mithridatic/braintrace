# Measured bias current intervention

The driver adds a separate somatic current clamp from time zero.
Its amplitude is the measured 0.031445374082395006 nA for sweep 35.
The test step remains 0.19 nA from 270 to 1270 ms.
All conductances, initial voltage, and numerical settings remain fixed.
The mesh factor is 9. CVode absolute tolerance is 1e-10.

| Direct observation | No bias | Measured bias |
| --- | ---: | ---: |
| Baseline mean, mV, 200-270 ms | -86.92923 | -84.47327 |
| First spike peak time, ms | 302.58887 | 287.63770 |
| First spike peak voltage, mV | 44.24820 | 44.31977 |
| Spikes during the step | 11 | 22 |

The human trace has 12 spikes and a first peak of 19.5625 mV.
Adding bias does not correct the high model spike peaks.
It substantially increases firing. Do not promote this candidate.
The missing acquisition bias is a protocol difference, not a proven explanation
for the waveform mismatch. The fitted leak may compensate for omitted bias.
This last statement remains an inference.

The zero-bias control reproduces the previous reference voltage on the audit
grid exactly. Recorded total current equals the sum of step and bias currents.
The settled bias samples equal the requested amplitude within 1e-12 nA.
The voltage-change RSS is 6441.32809 mV on 200000 samples at 0.005 ms.
The corresponding RMS change is 14.40325 mV.
These values describe the intervention, not biological uncertainty.

The experiment starts from -80 mV and applies bias for 270 ms before the step.
It does not reproduce the full acquisition history or slow-state equilibration.
The [JSON audit](h01-pv-bias-audit.json) retains onset channel states and events.
The NPZ outputs retain voltage, calcium, gates, and both applied currents.
Run `python -m docs.evidence.h01_pv_bias_audit` to repeat the checks.

The reference driver uses `--bias-na 0` for the control and
`--bias-na .031445374082395006` for the measured-bias case.
Both use `--current-na .19 --cvode-atol 1e-10 --nseg-factor 9`.
The common evidence tests pass (4 tests). Their scope is reporting math and
spike datums. The direct output checks establish current accounting and control
parity. Neither check establishes physiological validation.
