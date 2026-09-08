# Spatial check of the restored spike train

The slope-5 candidate restores late positive spikes at 0.27 nA on spatial
factors 9 and 27. The slope-6 controls have none on either mesh.
The predeclared two-mesh rescue check passes.

The coarse candidate has 98 events, including 71 positive events after
500 ms. The fine candidate has 97 events, including 70 after 500 ms.
The first upward crossing shifts by +0.000401 ms. Ordinal event 97 shifts
by +4.770390 ms, and one coarse event has no ordinal partner. Thus, the
full train is not converged. Equal presence of late spikes does not imply
equal event timing or a physiologically correct train.

The first post-spike minimum changes from -69.794835 to -69.800925 mV.
Time from the peak changes from 1.870230 to 1.870829 ms. These differences
are much smaller than the human-model mismatch: the recorded first minimum
is -78.906258 mV after 0.700000 ms. The first-return mismatch therefore
persists on both meshes.

The model settings match, and each section has three times as many segments.
Section connectivity, length, integrated area, Ra, and Cm pass the audit.
No physical parameter is promoted. The source-slope intervention changes
both gate equilibrium and kinetics; this check does not isolate mediation.

The [audit](h01-pv-slope5-mesh-audit.json) retains each event, ordinal
differences, unmatched events, and three minima per mesh. Reproduce it with
`python -m docs.evidence.h01_pv_slope_mesh_audit`. The reserved trace was
not read. Further calibration can use the robust early-return mismatch,
but final acceptance still requires numerical and physiological checks.
