# Spike and phase errors persist at tighter tolerance

Reducing CVode tolerance from 1e-10 to 1e-11 retains five complete
positive-peak spikes and passes all specified numerical limits.

| Measurement | Largest absolute change | Limit |
| --- | --- | --- |
| Onset | 0.0000264752 ms | 0.1 ms |
| Peak voltage | 0.0000332589 mV | 0.1 mV |
| Time above -20 mV | 0.000000678468 ms | 0.01 ms |
| Either separate phase | 0.000204609 ms | 0.01 ms |

Each phase error relative to the human trace retains its sign. This
passes the additional phase check declared before reading this output.
Both recordings match their indexed raw samples exactly. Input plateau
error is zero, excluding 1e-7 ms around transitions. Both traces are
finite and end at 2100 ms. Metadata differ only in tolerance, output
sample count, and execution time.

The [full result](h01-l2-reversal-active-tolerance-result.json) retains
all events, phases, audit outcomes, and artifact hashes. Together with
the spatial result, this supports using the observed errors for further
model diagnosis at this input. The model still has late spike onsets,
incorrect phases, and subthreshold amplitude errors. No physiological
or reserved-input qualification is established.
