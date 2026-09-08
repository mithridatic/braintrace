# Whole-cell slope intervention

Slopeh changes from 6 to 5 mV in soma and axon. The candidate retains soma
NaTg factor 1.10, closure factor 0.15, recovery factor 1, and source SK
and calcium parameters. Runs use spatial factor 9 and CVode tolerance 1e-10.

The predeclared high-input rescue passes: 71 complete positive-peak events
occur after 500 ms at 0.27 nA, versus none in the unchanged-slope control.
The full train has 98 events. At 0.19 nA, there are 41 events, with 29
positive events after 500 ms. Human traces have 43 and 12 events respectively.
Counts locate the mismatch; the audit retains every event and interval.

| First-event error, model minus human | 0.19 nA | 0.27 nA |
| --- | ---: | ---: |
| Onset, ms | +7.411478 | -0.730671 |
| Peak, mV | +0.723368 | +1.208308 |
| Duration above -20 mV, ms | -0.009459 | -0.002756 |

The first post-spike minimum at low input is -72.818615 mV after
2.129655 ms, versus the human -78.843750 mV after 0.760000 ms.
At high input it is -69.794835 mV after 1.870230 ms, versus human
-78.906258 mV after 0.700000 ms. The original combined candidate reached
only -68.118965 mV after 2.045295 ms at high input. The intervention
improves that return but leaves a substantial direct mismatch.

The raw slope change is sufficient to restore late firing in this model
under these conditions. It changes both equilibrium availability and
kinetics. The result does not prove exclusive mediation by stationary
sodium current. This candidate needs its own spatial check, improved
return voltage and timing, and human train validation. No parameter is
promoted to production BrainCell. The reserved trace was not read.

The [audit](h01-pv-slope5-audit.json) retains control late events, candidate
events, intervals, and minima. Run `python -m docs.evidence.h01_pv_slope_candidate`.
The reference CLI now rejects zero, negative, NaN, and infinite slopes.
Omission retains the source setting. Ten datum and driver tests passed.
The fixed-voltage regression separately verified the implemented slope effect.
