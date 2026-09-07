# l4-pyramidal-allen-527952884: NEURON reproduction of the published fit

Spec `docs/specs/2026-09-07-h01-donor-allen-l4-import.md`; manifest `h01-l4-reproduction-manifest.json`; decision `h01-donors/stage-allen-l4-decision.json`.
Detector: -20 mV rise crossing inside the pulse. Latency limit = |dt pair difference| x 2.95.

**Reproduction verdict: rejected**

## Counts and first spikes

| Input | Sweep | Registered count | Repeat band | Human re-read | Model dt | Model dt/2 | dt pair | Count exact | Repeat row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100 pA | 69 | 20 | 17-20 | 20 | 19 | 19 | agrees | missed_within_one | held |
| 90 pA | 39 | 12 | - | 12 | 8 | 8 | agrees | rejected | - |

| Input | Registered first spike (ms) | Human re-read | Model dt | Model dt/2 | Pair limit | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 100 pA | 32.18 | 32.17 | 33.74 | 33.74 | 0.004654 | missed |
| 90 pA | 33.88 | 33.88 | 39.6 | 39.6 | 0.004439 | missed |

## Wall clocks (runner-measured)

| Candidate | Input | Seconds | rc | Aborted |
| --- | --- | --- | --- | --- |
| source | 69 | 38.2 | 0 | False |
| source | 39 | 30.9 | 0 | False |
| source-dt-half | 69 | 69.9 | 0 | False |
| source-dt-half | 39 | 55.5 | 0 | False |

Total 194.4 s. Stderr files: `source-39.stderr.txt` (empty), `source-69.stderr.txt` (empty), `source-dt-half-39.stderr.txt` (empty), `source-dt-half-69.stderr.txt` (empty)
