# Gate 4 Docker proof — 2026-08-25

Image: `braintrace-example21:b75b834`.

Command used `--gpus all`, installed the pinned `braincell==0.1.0` dependency,
mounted this worktree at `/work`, and ran the parent against `/datasets/arc/raw`.

## Measured result

- Total proof time: 89.7786 seconds, below the 180-second limit.
- CPU and GPU gradients were finite and returned identical prediction bytes.
- CPU median: 4008.6819 ms; GPU median: 2549.7727 ms.
- Literal lower valid median selected `gpu`.
- One fixed-validation request had five decoder calls: maximum 0.012994 ms,
  below the 100 ms limit.
- Eight updates used `d631b094` only; `46f33fce` remained forward-only and
  validation state was unchanged.
- Direct evidence measured prediction change, targets, shape loss 7.3375974,
  row loss 2.2277696, and recurrent-weight movement 0.2960933.
- Six interventions were recorded. The null intervention stayed unchanged and
  all five state interventions changed the direct prediction.

Structured evidence is in `gate4-evidence.json`; its SHA-256 sidecar is
`gate4-evidence.sha256`.
