# Gate 4 Docker attempt — 2026-08-25

Image: `braintrace-example21:b75b834`

Command environment: Docker Desktop 29.0.1, RTX 3080 Ti exposed with
`--gpus all`, worktree mounted at `/work`, ARC root `/datasets/arc/raw`.

## Result

The parent started the isolated CPU child and enforced its 180-second deadline.
The CPU child did not finish its first warm 705-event PP-Prop gradient call
before the deadline. The parent terminated it and raised:

```text
subprocess.TimeoutExpired: ... --backend cpu ... timed out after 179.996 seconds
RuntimeError: Gate 4 proof exceeded 180 seconds
```

No successful evidence or result file was produced. CPU/GPU prediction equality,
decoder latency, updates, and interventions remain unverified because execution
did not reach those stages.

The image does not contain its declared `braincell==0.1.0` dependency. The run
installed that pinned package before starting the proof. The first attempted
dataset root, `/datasets/arc/raw/data/training`, was also rejected because the
loader appends `data/training`; `/datasets/arc/raw` is the correct runner input.
