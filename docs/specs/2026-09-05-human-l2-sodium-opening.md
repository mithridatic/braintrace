# Isolate sodium opening speed

The candidate reaches each sampled spike peak too quickly after the
-20 mV upward crossing. Test whether slower NaTs opening can lengthen
that phase. This is a model hypothesis, not a measured human kinetic law.

Prepare a separate mechanism directory from pinned source files. Extend
the existing default-one recovery mechanism with m_opening_factor=1.
Multiply mTau by this factor only when mInf > m. Keep mInf, original rate
expressions, closing dynamics, h kinetics, conductance density, and sodium
reversal unchanged. Do not modify the source used by the live midpoint run.

Source tests must reject an unexpected original digest and show that
removing only the declared additions restores the checked recovery
mechanism exactly. Build and record all source and library hashes.

Before any whole-cell intervention, use a fixed-voltage clamp with opening,
closing, and equilibrium initial states. Include -40 mV, where vtrap uses
its finite limit. Compare factors 1 and 2 with the analytic exponential
solution. For opening, the time constant must double; for closing it must
remain unchanged. Equilibrium must remain unchanged. Inactivation must
remain unchanged when its recovery factor is one. Require maximum gate
error <=1e-8 with a documented solver and tolerance.

Also verify that factor one reproduces the existing whole-cell control
before interpreting a factor-two cell run. The driver must reject
nonpositive or nonfinite factors and record the applied factor. Numerical
audits must treat the new factor as an invariant. These tasks remain
required even if preparation and source tests pass.

Select and freeze the full-cell control after the pending calcium midpoint
result. Then declare event and phase predictions before the opening-speed
intervention. The reserved human input remains excluded. No preparation
or clamp result qualifies the full cell or H01 physiology.

The midpoint prediction passes. Freeze the full-cell control as
h01-l2-sweep50-ih75-epasminus4-ca1375. Before changing opening speed,
run the separate activation mechanism with opening factor one and
recovery factor one at identical settings. Require exact equality of
time, voltage, applied current, and the recorded soma arrays. Retain
any discrepancy and diagnose it before an intervention. The old control
has no opening-factor metadata: use its pinned mechanism to document
the baseline law. Mechanism source and library hashes must differ only
as accounted for by the isolated preparation; other physical setup
fields must agree. This is a source-equivalence check, not a numerical
or physiological qualification of the midpoint.

After default-factor equivalence passes, change only the opening factor
from one to two. Keep calcium factor 1.375 and all other frozen settings.
Require five complete positive-peak spikes and a reduction greater than
0.01 ms in the absolute rise-to-sampled-peak human error for each event.
The 0.01 ms minimum change is a diagnostic margin, not a physiological
acceptance tolerance. The earlier calcium-factor-1.5 control had phase
changes smaller than this margin under numerical refinement. The frozen
midpoint and opening intervention still need their own numerical checks.

Missing or extra events reject the combined phase prediction. Invalid
input, unintended setup changes, nonfinite data, or incomplete boundary
events invalidate the test. Record every phase, onset, peak, interval,
recovery minimum, and unmatched event. Report effects on the falling
phase separately; do not infer unchanged cell repolarization from an
unchanged gate-closing law. A changed voltage trajectory changes other
currents and states. The rising-phase prediction cannot establish a full
fit if other response errors persist or worsen.
