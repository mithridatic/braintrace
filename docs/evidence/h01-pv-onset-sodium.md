# Regional sodium onset split

Three runs increased NaTg conductance by 10 percent at 0.19 nA.
The candidate retained closure factor 0.18 and recovery factor 1.
Other parameters stayed at their source values. Mesh factor was 9;
CVode absolute tolerance was 1e-10. The reserved trace was not used.

| Changed region | First onset advance, ms | Onset error, ms | Peak error, mV | Duration error, ms |
| --- | ---: | ---: | ---: | ---: |
| None | 0 | 12.211840 | 3.371850 | -0.000283 |
| Soma | 3.656649 | 8.555191 | 5.798539 | -0.004756 |
| Axon | 0.196302 | 12.015538 | 3.502154 | -0.000241 |
| Both | 3.787457 | 8.424384 | 5.852106 | -0.004594 |

Errors are model minus human. Onset is the first upward -20 mV crossing.
Duration is time above -20 mV, not half-width. The human targets are
290.797663 ms, 19.5625 mV, and 0.273395 ms.

The soma and combined interventions pass the predeclared 1 ms advance
decision. The axon intervention does not. Neither single-region change
matches the combined advance within 0.1 ms. The soma difference is
0.130807 ms, so regional sufficiency remains unresolved under that rule.
Do not widen the decision limit after seeing this result.

Every intervention increases the absolute peak error. None passes the
predeclared shape screen. No candidate is promoted. This split establishes
a conditional response to sodium density, not the unique cause of the
human-model onset mismatch. Soma sodium density is not an independent
onset control: this intervention also changes the first spike shape.

All events are retained in [the audit](h01-pv-onset-sodium-audit.json).
The raw traces and metadata use the prefix `h01-pv-onset-na110-`.
Reproduce the analysis with `python -m docs.evidence.h01_pv_onset_sodium_audit`.

## Audit correction and checks

The first audit failed because the retained control predates the calcium
removal metadata field. A regression test reproduced that failure before
the fix. The audit now permits the missing field only for that legacy
control. New intervention records must explicitly state no calcium change.
Check metadata version differences before assuming a new field exists.

The audit requires complete first spikes with positive peaks and checks
input, integration, temperature, initial voltage, and intervention settings.
Missing events fail explicitly. The retained-data regression passes.
Future independent tests should cover absent spikes and altered intervention
metadata. This evidence test does not qualify the production model.
