# H01 masked Muon: Vast 12-cell acceptance

Result: **PASS** on source `b2f0fd02bf9be7f829d12a18181977db72d81d08`.
Executed on the existing Vast RTX 4090 instance, 2026-09-11, with JAX/JAXlib
0.11.0, Optax 0.2.8, BrainState 0.5.3 and BrainCell 0.1.0. No code changes were
needed after the locally verified implementation.

Command (from the isolated source worktree):

```text
/workspace/venv314/bin/python -m examples.h01_arc_probe --cache /h01 --cells 12 --sparse-learning --output /evidence/h01-12-masked-muon-b2f0fd0.json --wall-limit-seconds 900 --rss-limit-gib 16
```

- Nine phases passed; exit code 0; 418.035 seconds overall, below 900 seconds.
- Peak process RSS 7,503.797 MiB (7.328 GiB), below 16 GiB.
- First learning update including compilation: 245.949 seconds; warm update:
  10.281 seconds. Fresh isolated compilation-cache directory.
- Both losses, gradients, optimizer state and eligibility state were finite.
  Synthetic loss was 0.4381818941 then 0.4237578354. Native/adapter zero-input
  soma error was below 1e-8 mV; all four parameter groups changed.
- Real network: 12 cells, 84,097 compartments, **two recurrent contacts**.
  Input: 5,292 weights with masked Muon. Recurrent: two independent contact
  weights with masked Muon. Readout matrix: 4,320 weights with Muon. Bias:
  360 values with Nesterov AdamW. All retained weight decay 0.1.
- Numerical tests on the same GPU environment: **20 passed in 21.62 seconds**,
  JUnit confirms zero failures/errors/skips. Includes synthetic 36- and
  240-contact cases, dense Muon oracles, sparse momentum and model write-back.
- Seventeen report acceptance checks passed, including exact Git-blob hashes
  for 13 source files. The launch entrypoint hash is independently checked
  against Git because the probe does not list its own source hash.

The log contains the two existing warnings about readout parameters absent from
the recurrent graph. This is not a zero-warning run. Contact activity is not
forced; recurrent movement can include weight decay. This bounded synthetic
probe is not full ARC training, physiological qualification or proof of improved
learning quality. The sparse optimizer remains an explicitly masked Muon variant.

Evidence: [raw report](h01-12-masked-muon-b2f0fd0.json),
[verification](h01-12-masked-muon-b2f0fd0-verification.json),
[launch](h01-12-masked-muon-b2f0fd0-launch.json),
[terminal](h01-12-masked-muon-b2f0fd0-terminal.json),
[log](h01-12-masked-muon-b2f0fd0.log),
[GPU tests](h01-masked-muon-vast-tests.xml).
The SSH test wrapper had an exit-formatting error after pytest completed;
the passing test verdict is checked from the retained JUnit and pytest log.
