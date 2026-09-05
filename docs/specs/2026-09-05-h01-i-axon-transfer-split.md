# Axon-only transfer diagnosis

Use the installed CompositeByTypePolicy. Keep maximum compartment length
2.5 um outside the axon. Set the axon maximum to 0.25 um. Verify that every
non-axon interval is unchanged before simulation. Keep all physical channel,
geometry, current, initial-state and temperature settings fixed.

Use dt 0.000625 ms and run through 330 ms with the existing candidate runner.
Compare with the completed full transfer on the same 270 to 330 ms window.
The reference is h01-pv-regional-mesh-axon2187. Keep complete raw traces and
record the explicit regional policy in output metadata.

The primary claim is that this axon-only refinement repairs event 7's rising
crossing within 0.1 ms of the reference, with matching complete event counts.
Reject that claim if either condition fails. Check all events separately
with the existing 0.1 ms onset, 0.1 mV peak and 0.01 ms width limits.
Invalid coverage or changed non-axon intervals invalidate the experiment.

No success here establishes whole-response transfer or human validity.
No failure here proves that further spatial refinement cannot help.
This is one fixed diagnostic, not a physiological fitting campaign.

The installed solver uses a hierarchical cable solve, not a general dense
matrix inverse. Runtime attribution inside that solve remains unresolved.
Use a short regional-policy run to measure cost before a longer execution;
do not equate the existing resting-loop timing with a guaranteed runtime.

## Full-response check after the early-window pass

The eight-event comparison passes all fixed limits. Extend the identical
configuration to 1500 ms and compare every complete event from 270 to 1270 ms.
Require the reference count of 40 and every original onset, peak and width
limit. The human count of 43 is separate and must not replace the numerical
reference. Also verify exact agreement with the saved 330 ms run over their
shared samples. Retain raw output and regional-policy metadata.

The completed 330 ms run took 295.8 seconds. Linear duration scaling gives
about 22.4 minutes for 1500 ms. This estimate includes uncontrolled runtime
effects and is not a guarantee. Run one full comparison, with no fitting or
default promotion. A failure requires diagnosis, not a relaxed limit.
