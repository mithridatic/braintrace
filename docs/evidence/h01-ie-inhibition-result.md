# Functional inhibition at the measured I-to-E pair (SP5 stage 1): stopped, untested

Decision JSON: `h01-ie-inhibition/decision.json` (2026-09-07). Spec
[functional inhibition](../specs/2026-09-07-h01-functional-inhibition.md) with the 2026-09-07 addendum
(1800 s / 3000 s aborts, drive contingency). Commit `e2d1291`.

## What happened

| Arm | Status | Launch | End |
| --- | --- | --- | --- |
| 1 `disconnected` (300 ms, 0.6 nA, dt 0.005) | stopped by the user before completion; untested | 2026-09-07T21:58:08.8492449-07:00 (CPU 51 percent; wait condition held at 21:57:58) | killed by the coordinator on the user's stop, after `Circuit constructed` at 21:59:57 (109 s) and before 22:11:07; no trace, no run-log entry |
| 2 `measured` | not launched (user stop); untested | - | - |
| 3 `soma` | not launched (user stop); untested | - | - |
| 4 `measured-halved` | not launched (user stop); untested | - | - |

No 300 ms arm completed, so there is no per-cycle table, no halving limit, no drive-rule evaluation
(the 0.8 nA contingency was not applied) and no cap spent (`prior_evaluations` stays 1, the
benchmark). Gate verdict: **not discriminable**. Placement hypothesis: **undetermined**.

## Registered predictions (registered, untested)

| Arm | Prediction |
| --- | --- |
| 1 disconnected | >= 3 E spikes in 300 ms; first spike between 20 and 60 ms; I fires once per pulse |
| 2 measured | soma IPSP magnitude < 0.05 mV; no E spike or cycle beyond the halving limit; site response 0.5 to 2 mV |
| 3 soma | soma IPSP magnitude >= 0.3 mV; at least one E spike delayed beyond the shift limit |
| 4 measured-halved | E spike count equal to arm 2; per-cycle ranges below 0.1 ms |

Human tiers: not applicable (no human recording of this pair). The only completed SP5 run remains
the 100 ms benchmark ([page](h01-ie-inhibition-benchmark.md)).
