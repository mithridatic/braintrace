# Direct I compartment recording

The measured circuit has only I-to-E wiring. I has no incoming network edge.
Its standalone response can therefore be checked against its circuit response
under identical source geometry, placements, initial state, and drive.

Add a diagnostic recorder that uses `brainstate.transform.for_loop` to retain
each I CV voltage at each step. Keep the declared contact event, soma voltage,
contact voltage, and soma NaTg m and h. Retain CV IDs, branch IDs, and interval
bounds so each voltage is tied to a physical model location. Keep quantities
and JAX arrays inside the time loop; convert only for final file export.

Test the recorder against Network.run on a small I-to-E-only fixture with
active spiking. Require identical output events and voltage agreement within
1e-9 mV absolute and 1e-10 relative. Verify every saved contact voltage against
its identified column in the all-CV array. Additional gate probes must not
change the original traces. Reject nonpositive or nonfinite time settings.

Before a spatial diagnosis on H01, repeat the unchanged measured I baseline
and compare its soma and contact traces with the existing circuit record.
Do not transfer fixture equivalence to H01 without this check. Only then use
the all-CV traces to identify the first positive local excursion and its path.
A positive excursion is an observation, not by itself a physiological spike.
Preserve any diagnostic channel-law override explicitly and separately from
the frozen profile. No default or electrical-map change belongs in this step.
