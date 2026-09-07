# l3-sst-interneuron-hl5mn1: NEURON reproduction of the published fit

Spec `docs/specs/2026-09-07-h01-donor-hl5mn1-import.md`; manifest `h01-sst-reproduction-manifest.json`; decision `h01-donors/stage-hl5mn1-decision.json`.
Detector: -20 mV rise crossing inside the pulse. Latency limit = |dt pair difference| x 2.95.

**Reproduction verdict: rejected**

## Counts and first spikes

| Input | Sweep | Registered count | Repeat band | Human re-read | Model dt | Model dt/2 | dt pair | Count exact | Repeat row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 pA | 44 | 14 | 12-14 | 14 | 16 | 16 | agrees | rejected | missed |
| 150 pA | 35 | 34 | - | 34 | 30 | 30 | agrees | rejected | - |

| Input | Registered first spike (ms) | Human re-read | Model dt | Model dt/2 | Pair limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 100 pA | 24.16 | 24.16 | 26.97 | 26.92 | 0.1337 | missed |
| 150 pA | 13.0 | 13 | 16.04 | 16 | 0.1128 | missed |

## Wall clocks (runner-measured)

| Candidate | Input | Seconds | rc | Aborted |
| --- | --- | --- | --- | --- |
| source | 100 | 7.3 | 0 | False |
| source | 150 | 14.2 | 0 | False |
| source-dt-half | 100 | 11.9 | 0 | False |
| source-dt-half | 150 | 26.0 | 0 | False |

Total 59.4 s. Stderr files: `source-100.stderr.txt` (empty), `source-150.stderr.txt` (empty), `source-dt-half-100.stderr.txt` (empty), `source-dt-half-150.stderr.txt` (empty)
