# Frozen candidate human-response gaps

The saved I candidate fails the full event-count requirement at 0.27 nA:
40 model events versus 43 human events. The E candidate has five events,
as does its human reference, but each waveform and recovery still needs
qualification. These are donor-geometry comparisons, not the H01 diagnostic
closing-time override. Reserved traces were not used.

| First-event measurement | E model minus human | I model minus human |
| --- | ---: | ---: |
| Rising -20 mV crossing | +0.901572 ms | -0.667992 ms |
| Peak voltage | -0.071493 mV | -1.497403 mV |
| Duration above -20 mV | +0.027486 ms | -0.046441 ms |
| Recovery minimum voltage | +0.058685 mV | +7.006230 mV |
| Peak-to-minimum delay | +1.194924 ms | +0.659179 ms |

A nearly correct minimum voltage can occur at the wrong time. Neither peak
agreement nor a passing event count establishes the complete direct response.
The [full audit](h01-frozen-candidate-full-response-audit.json) retains all
model events, human targets, and recovery residuals. I comparisons are ordinal
diagnostics: unequal counts prevent complete correspondence. They do not
identify which particular biological events are absent.

No physiological pass is issued. Per-measurement allowances remain unresolved,
and the I count failure is explicit. Full response transfer and validation
at the other required inputs remain open.
