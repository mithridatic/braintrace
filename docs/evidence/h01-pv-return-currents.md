# Currents during the first voltage return

The slope-5 candidate's recorded soma currents were sampled at the first
peak, downward -20 mV crossing, and subsequent voltage minimum. This uses
the retained mesh-9 traces. No simulation parameters changed.

At high input, the minimum is -69.794835 mV. Ca_LVA current is
-0.031774 mA/cm2, NaTg is -0.002295 mA/cm2, and Kv3_1 is
+0.047181 mA/cm2. At low input, the corresponding values are -0.018150,
-0.001116, and +0.030150 mA/cm2. Outward current is positive.

Thus, Ca_LVA supplies a substantial local inward current at the minimum.
Its direction opposes the voltage fall if other currents are held fixed.
This does not establish its total effect in the freely evolving cell.
Calcium entry also changes calcium state and can affect later SK current.
A conductance intervention is needed to test the post-spike response.

The audit checks sodium, potassium, and calcium channel sums against their
recorded ion currents over the whole trace. Maximum errors are below
1e-8 mA/cm2. It does not include axial, leak, Ih, or applied current and
must not be interpreted as a complete local current balance.

The [JSON](h01-pv-return-currents.json) retains all event times, voltages,
and channel currents. Run `python -m docs.evidence.h01_pv_return_currents`.
These measurements nominate a causal test; they do not prove that Ca_LVA
causes the human-model mismatch.
