# Gate 4 Docker proof rerun — 2026-08-25

Image: `braintrace-example21:b75b834`.

Command: `docker run --rm --gpus all -v ${PWD}:/opt/braintrace -w /opt/braintrace braintrace-example21:b75b834 python examples/pp_prop/example21_gate4.py --data-root /datasets/arc/raw/data/training --output /opt/braintrace/reviewer-gate4-result.json`.

## Measured result

- Total proof time: 101.5703 seconds, below the 180-second limit.
- CPU and GPU PP-Prop probes were finite and returned identical prediction bytes.
- CPU median: 4011.0095 ms; GPU median: 2892.4044 ms; literal lower valid median selected `gpu`.
- The selected process measured 31 fixed-validation readouts, each with five warmed decoder calls. Maximum call: 6.0337 ms, below 100 ms.
- Eight updates used `d631b094` only. `46f33fce` was forward-only and validation state was unchanged.
- Direct pre/post predictions, targets, shape and row losses, and recurrent-weight movement were recorded. Pre losses were shape 7.4920 and rows 2.2552; post losses were shape 7.4509 and rows 2.2303; movement was 0.0377942.
- All six interventions were recorded. The null intervention stayed unchanged and the five state interventions changed the direct prediction.

Structured evidence is in `gate4-evidence.json`; its SHA-256 sidecar is `gate4-evidence.sha256`.
