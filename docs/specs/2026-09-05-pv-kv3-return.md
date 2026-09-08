# Somatic Kv3 conductance and voltage return

Double somatic Kv3_1 conductance on the half-Ca_LVA candidate. Keep axonal
Kv3_1 unchanged. The new soma density is 0.85769694763999 S/cm2, within
the source optimizer bounds previously recorded. This is an inferred
parameter, not a measured human density.

Retain soma NaTg factor 1.10, sodium closure factor 0.15, recovery factor 1,
inactivation slope 5 mV, and somatic Ca_LVA factor 0.5. Other parameters
stay unchanged. Run both calibration inputs with mesh factor 9 and CVode
tolerance 1e-10. Do not use the reserved trace.

At each input, test whether the first downward -65 mV crossing occurs at
least 0.1 ms earlier relative to the first peak than in the unchanged-Kv3
control. Require the first two events to retain positive peaks. Missing
spikes or crossings invalidate this comparison. Preserve first peak,
onset, width, minimum voltage and time, and every later event. A faster
voltage return does not by itself qualify the candidate.

The driver factor defaults to 1, permits 0, and rejects negative or
nonfinite values. Record it separately from the existing conductance
intervention. This diagnostic uses the existing NaTg intervention, so
the two factors must not overwrite each other.
