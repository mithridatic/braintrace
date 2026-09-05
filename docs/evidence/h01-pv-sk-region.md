# Regional SK split

Both regional simulations completed on the 0.18 closure-factor candidate.
All other parameters match the existing 0.19 nA comparison.

| Doubled SK region | First rising -20 mV crossing change, ms |
| --- | ---: |
| Both | +2.710159 |
| Soma only | +0.005818 |
| Axon only | +2.702903 |

Only the axon-only result is within 0.1 ms of the both-region result.
It satisfies the predeclared regional sufficiency criterion for this onset
delay. The soma-only result does not. The comparison identifies a sufficient
regional parameter change, not a unique mediator or spike initiation site.

The axon-only and both-region cases each have 22 events. The unchanged and
soma-only cases each have 29. These counts are descriptive; the sufficiency
decision uses the specified direct crossing time. The JSON retains every
event and interval so count equality does not stand in for trajectory equality.

The preceding interpretation was limited by somatic current probes while
both regions changed. This split supplies evidence about regional causation.
It does not measure the axonal current waveform or establish which subsequent
channel states mediate later intervals.

Increasing axonal SK in this candidate worsens its already late low-current
onset. Further tuning must address that coupled effect rather than treating
SK as an independent adjustment of late firing. No candidate is promoted.

Run `python -m docs.evidence.h01_pv_sk_region_audit` to reproduce the decision.
The [JSON audit](h01-pv-sk-region-audit.json) retains the four direct responses.
The result remains conditional on mesh factor 9 and the tested input.
