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

## Paired response and numerical checks

Change only edge enablement between the two runs. Require equal physical
metadata except the control and enabled edge. Require identical I traces and
events. E voltage must match before delivery, and disconnected E conductance
must stay zero. Check the first conductance sample against the I emission
time plus the configured delay. Use local receptor voltage to determine
current direction. Soma voltage does not determine the local current sign.

For each complete E excursion, retain the rising and falling -20 mV crossings,
peak voltage, and emitted event time. Interpolate threshold crossings between
adjacent samples. Compare events in time order only when counts match. If
counts differ, report all events in both traces without forcing a pairing.
A delayed spike is not an absent spike. A local voltage change alone does
not establish reduced firing. No sign or size of the effect is assumed.

Repeat both controls at 0.0025 ms with all other settings fixed. Run them
sequentially. Require the same event counts and preserve the existing limits:
0.1 ms for onset, 0.1 mV for peak, and 0.01 ms for duration above -20 mV.
Apply these limits to each complete soma excursion in each control. Check
contact events and delivery timing separately. Report the change in each
edge-induced response at the finer step; an effect that changes sign or is
not resolved from numerical change is not qualified. Passing these checks
does not establish spatial convergence or a match to human recordings.
