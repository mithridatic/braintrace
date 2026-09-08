# Kv3 timing diagnostic

The conductance split improves the falling phase but worsens low-input
onset. Test gate speed as a separate parameter. Copy the pinned diagnostic
mechanisms to a new cache and add a Kv3 m time-constant factor, default 1.
Multiply mTau only. Keep mInf, conductance, reversal voltage, and sodium
mechanisms unchanged. Hash the original and modified sources.

Before whole-cell use, compare eight fixed-voltage cases: -80 and +20 mV,
initial gate 0 and 1, time-constant factor 1 and 0.5. Use zero conductance
to isolate gate dynamics and a voltage clamp. At exactly 0.5 ms, require
agreement within 1e-8 with `mInf + (m0 - mInf) * exp(-t / tau)` and voltage
agreement within 1e-8 mV. This check must include both opening and closing.
It establishes the implemented intervention, not a human gate law.

If this passes, a later whole-cell test can halve somatic Kv3 gate time
with source Kv3 density on the half-Ca_LVA candidate. It must compare
onset, matched-voltage return, first shape, and all later spikes. Faster
opening also means faster closing in this intervention. Do not describe
it as an opening-only test or promote it from the fixed-voltage check.

The fixed-voltage check passed. Run somatic Kv3 tau factor 0.5 at both
calibration inputs on mesh factor 9 with CVode tolerance 1e-10. Keep soma
Kv3 density at its source value, soma Ca_LVA factor 0.5, sodium density
factor 1.10, slope 5, closure factor 0.15, and recovery factor 1.
At each input, report whether the downward -65 mV crossing advances at
least 0.1 ms from peak and whether onset shifts by no more than 1 ms
against the source-speed control. These are separate diagnostic decisions,
not human tolerances. Require two initial positive spikes for the return
comparison. Retain all events and shape errors.

Also run factor 1 at 0.19 nA in the modified build. Require identical
recorded time and soma voltage to the existing half-Ca_LVA control before
attributing differences to the timing intervention. The driver option
must reject zero, negative, and nonfinite values and record the factor.
