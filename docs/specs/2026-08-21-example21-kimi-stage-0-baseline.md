# Stage 0: canonical baseline and borrowing audit

## Existing state

At the campaign base revision, Example 21 already defaults to Muon with weight
decay 0.1, cosine learning-rate decay without warmup, and memory/reasoning
softcaps 4/25. Those values are implementation defaults, not local validation
of the Kimi-derived choices.

## Required interface

Add `lr_warmup_fraction` as a finite value in `[0, 1)`, defaulting to `0.0`.
For cosine scheduling, a nonzero fraction performs linear warmup over the
corresponding leading fraction of optimizer updates and then cosine decay to
zero. Constant scheduling rejects nonzero warmup because that arm is defined as
a constant-rate control. The resolved fraction is reported in configuration,
optimizer policy, result JSON, checkpoint compatibility, and the model/run
manifest.

## Frozen comparisons

All pilots use protocol v2, seed 2108, 4,096 neurons and edges, width 32, 60
latent ticks, 260 updates, batch 32, a fixed source manifest, identical sampled
task schedules, full controls, and trajectory diagnostics.

1. Optimizer: Muon decay 0.1, Muon decay 0.01, AdamW decay 0.01.
2. Schedule using the optimizer winner: cosine/no warmup, cosine/1% linear
   warmup, constant.
3. Softcaps using preceding winners: 4/25, 1/25, 4/1, and legacy 1/1.

Each arm first runs 100 evaluation tasks. The combined winner then runs 419
queries at seeds 2108, 31337, and 7777 against the base defaults. If the full
gate fails or is absent, defaults remain unchanged and the values are recorded
as locally unvalidated.

## Tests

- default schedule equivalence at `lr_warmup_fraction=0.0`;
- deterministic warmup boundary and cosine endpoint;
- invalid type, NaN, infinity, negative, and one-or-larger validation;
- constant/nonzero-warmup rejection;
- CLI, configuration, report, result, and checkpoint round-trips;
- unchanged optimizer-specific decay resolution.
