# Drive E during the measured I event

The standard soma pulses end at 5 ms. In the diagnostic I closing-time
condition, the measured contact emits at 8.355 ms. An E response under
ongoing drive is a separate requirement from synaptic delivery after its
pulse ends.

Expose per-cell pulse delays and durations in the circuit factory and CLI.
Preserve the existing 2 ms delay and 3 ms duration as defaults. Require finite
nonnegative delays, finite positive durations, and exactly E/I keys in each
explicit timing map. Record every cell's input timing in its evidence.
Reject invalid timing before constructing the network. Keep current amplitudes,
channels, regions, contacts, and solver selection independent of timing.

Check exact old-default equivalence on the fixture. Check that a delayed
pulse preserves the pre-pulse response and changes voltage after onset.
Check invalid keys, negative/zero durations, negative delays and nonfinite
values. No physiological default is promoted by these input controls.

After the pending delivery comparison, the planned E excitability diagnostic
uses 5 nA from 8 to 18 ms, with I at 1 nA from 2 to 5 ms and the same I
closing-time restoration. Observe through 25 ms. First check the disconnected
case for a complete E excursion through -20 mV and a positive peak. This
input is a model stimulus, not a measured H01 protocol or fitted human datum.
If it fails, retain the direct result and diagnose E before interpreting an
inhibition test. If it passes, compare the same stimulus with the measured
edge enabled. Keep the established numerical limits; qualification still
requires time-step refinement and human-response validation.
