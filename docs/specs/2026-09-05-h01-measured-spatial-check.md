# Measured H01 spatial check

The E5 diagnostic circuit passes the declared excursion limits between
0.0025 and 0.00125 ms. Test spatial resolution before treating this as a
qualified circuit response. Keep the diagnostic I closing restoration explicit.

Run the disconnected pair with maximum compartment length 5 um instead of
10 um. Keep the 0.00125 ms step, 25 ms window, E input of 5 nA from 8 to 18 ms,
and I input of 1 nA from 2 to 5 ms. Use the same solver, source anatomy,
electrical region map, channel parameters, and initial conditions. Run alone.

Check every complete E soma, I soma, and I contact excursion against the
10 um record. Require equal counts and the existing limits: onset 0.1 ms,
peak 0.1 mV, and duration above -20 mV 0.01 ms. Keep direct traces and source
hashes. Check the measured contact projection coordinates and record the
selected membrane compartment. A mesh change can move the nearest compartment;
do not interpret that difference as a new measured contact position.

If the disconnected comparison passes, run the connected control with the
same 5 um settings. Require the same waveform limits against its 10 um record.
Compare the two 5 um controls: I must remain identical, E must match before
delivery, and the edge must deliver conductance after the declared delay.
Report each E event and the change in the edge-induced onset delay across
meshes. Do not infer spike removal from a delay or a local voltage difference.

If a waveform or event check fails, retain the failure and inspect the changed
compartments and contact selection before further refinement. Do not adjust
channels, contact coordinates, inputs, or tolerances to obtain a pass.
This check does not validate the inferred electrical regions or human responses.
