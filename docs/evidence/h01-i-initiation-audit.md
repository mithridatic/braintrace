# I initiation audit (2026-09-07)

Re-scored the existing finalist traces with `h01_initiation_score.score_run`,
pulse 270-1270 ms. No new model evaluation and no holdout opened.

| Input | Trace | Spikes | Axon lead at first -20 mV crossing (ms) |
| --- | --- | --- | --- |
| 0.19 nA | e-kv3-close2-019 | 14 | 0.045364 |
| 0.23 nA | p-close2-023 | 26 | 0.036063 |
| 0.27 nA | e-kv3-close2-027 | 37 | 0.028656 |

Source: [retained audit](h01-i-initiation-audit.json), original traces in
`h01-i-energetic/`. Each has axon voltage recorded at the donor axon midpoint.

The axon already crosses before the soma in the recorded model. An absent
axon-first crossing is therefore not a supported explanation of the I finalist's
missing early burst. This probe does not locate the earliest initiation point
throughout the cell or qualify the small lead against mesh/time refinement.
It also does not exclude an effect of axonal sodium kinetics or distribution.

Next diagnostic work must distinguish recovery/availability and post-trough
axial drive on this existing initiation pathway. The driver has only one
conductance-scaling slot, already used for somatic NaTg x1.1; replacing it with
an axonal scale would change two regions relative to the finalist. Preserve the
somatic scale before attempting a causal axonal density comparison. No reserve
is spent on that confounded substitution. Noise 1 sweep 48 remains sealed.
