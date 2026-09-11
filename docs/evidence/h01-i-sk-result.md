# SP10 result: axonal SK is the path of the finalist's late delay

Spec: [2026-09-11-h01-i-sk-isolation](../specs/2026-09-11-h01-i-sk-isolation.md).
Manifest: [h01-i-sk-manifest.json](h01-i-sk-manifest.json). Executor: Vast box, NEURON 9.0.2,
local executor of the campaign runner (no Docker). Two evaluations, cap two.

## Stage 0: platform transfer (PASS, 12 of 12 bands)

`s0-finalist` writes byte-identical flags to `e-kv3-close2` (sha256 `921cc6d4…`).
[Decision](h01-i-sk/stage-0-decision.json), [usable tier](h01-i-sk/s0-finalist-usable.md).

| Input | Count | Cycle 2 (ms) | Troughs 1-3 (mV) | Axon first |
| --- | ---: | ---: | --- | --- |
| 0.19 nA | 14 (container 14) | 34.759 (34.76) | -80.127 / -80.541 / -80.650 (-80.13 / -80.54 / -80.65) | yes |
| 0.27 nA | 37 (container 37) | 16.370 (16.37) | -78.986 / -79.298 / -79.612 (-78.99 / -79.30 / -79.61) | yes |

Runs took 19 s and 45 s on the box against several hundred seconds in the container.

## Stage 1: axonal SK density zero, single change (PASS, 22 of 22 bands)

[Decision](h01-i-sk/stage-1-decision.json), [usable tier](h01-i-sk/s1-sk-axon0-usable.md).

| Input | Row | s0-finalist | s1-sk-axon0 | Human |
| --- | --- | ---: | ---: | ---: |
| 0.19 nA | count | 14 | 29 | 12 |
| 0.19 nA | last trough-to-threshold delay (ms) | 80.8 | 32.2 | — |
| 0.19 nA | cycle 2 (ms) | 34.8 | 30.9 | 7.6 |
| 0.19 nA | late cycle (ms) | 84 | 33-35 | 96-117 |
| 0.27 nA | count | 37 | 59 | 43 |
| 0.27 nA | last trough-to-threshold delay (ms) | 27.2 | 14.1 | — |
| 0.27 nA | cycle 2 (ms) | 16.4 | 15.9 | 6.3 |
| 0.27 nA | late cycle (ms) | 30.4 | 17.1 | 24-26 |

Axonal SK current at trough+6 ms is zero by construction in the arm. Widths (0.221 ms),
peaks (17.8-18.0 mV), troughs 1-3 (within 0.04 mV of the baseline) and axon-first
initiation are unchanged. No depolarisation block (longest time above -20 mV 0.22 ms).

## Reading

Under the finalist settings the calcium-activated axonal SK current is the direct
path of the growing late delay: without it the cycle stops growing after cycle 2.
This is a mechanism result, not a repair. Both counts move away from the human, and
the human's late cycle lies between the two settings at 0.27 nA but beyond both at
0.19 nA. A brake that accumulates with calcium grows with the input; the human's late
slowing shrinks with the input. The human's early burst (cycles 2-4 at 6-10 ms) is
absent at both settings.

Late-rate f-I from the mean of the last five cycles: human 8.2 Hz at 0.19 nA and
39.4 Hz at 0.27 nA (0.39 Hz/pA, zero near 169 pA); s1-sk-axon0 28.3 and 58.5 Hz
(0.38 Hz/pA, zero near 115 pA); s0-finalist 12.0 and 32.9 Hz (0.26 Hz/pA, zero near
144 pA). The SK-free model has the human's late gain and a rheobase about 55 pA too
low; the SK dose lowers the gain without moving the rheobase to the human's. This is a
description of three lines through two points each, not a mechanism.

Both tiers: the 1 mV contract fails on count at both inputs for both arms; the usable
tier fails rate and adaptation at both inputs for both arms (see the usable pages).
The finalist stays experimental. Causal model: Y3 updated in commit `8869c22`.
