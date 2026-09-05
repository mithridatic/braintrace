# Layer-2 recorded-bias split

Question: Does restoring the separately recorded bias alone account for
the source model's extra events and early event times at sweep 50?

Control: retrieved source fit, command-only input, dt 0.000625 ms.
Intervention: add the recorded -0.003711858945210089 nA bias to every
command sample, including the complete pre-pulse history. Keep the
initial state rule, geometry, mechanisms, fit, and solver unchanged.
Do not re-fit the leak or shift voltage traces.

Before examining the intervention, declare a narrow sufficiency screen:
there must be exactly five complete positive-peak excursions above
-20 mV during the main pulse, no extra negative-peak excursions, and
every ordinal onset must be within 1 ms of its human counterpart.
Failure of either condition rejects sufficiency at this setup. This
1 ms diagnostic tolerance is not a new physiological validation gate.

Retain each event, peak, duration, and individual onset error. Record
baseline voltage separately. Inspect current plateaus and all four
command transitions, including the early test pulse. A changed fit,
wrong bias, invalid input, or incomplete observation invalidates the
comparison. A numerical failure limits interpretation and must remain
explicit. Passing this screen alone does not qualify cell physiology.

Implementation adds an opt-in recorded-bias flag. The default preserves
the source command-only run. Tests cover default identity, correct
signed addition, nonfinite bias, and malformed bias shape. The driver
must record the mode and actual bias. No production model changes.
