# Example 21 Kimi Transfer Stage 3: Latent Attention Residual

## Status

Approved for implementation on `feat/example21-kimi-transfer`. Promotion is
not evaluated until the preregistered GPU evidence exists.

## Objective

Apply the existing `braintrace.nn.AttentionResidual` across latent reasoning
depth without changing the decoder `refinement_mixer` or the default model.

## Configuration

- `latent_residual_mixer`: `"none" | "attention_residual"`, default `"none"`.
- `latent_residual_block_size`: positive integer, default `10`.

The attention arm requires associative memory. Its static block capacity is
`ceil(max_latent_steps / latent_residual_block_size)`.

## State and Update Contract

- Each valid query-input event resets the latent-depth accumulator and caches
  its reasoning query as source zero.
- Each latent tick forms the ordinary candidate reasoning query and stores it
  in a `HiddenState`, preserving a direct pp-prop path.
- Prior candidates in the current block and all completed summaries are stored
  with stopped gradients in resettable, snapshot-compatible history.
- At every full block boundary, and at the configured final partial boundary,
  the current block mean is appended to history. One block-specific
  pseudo-query attends over source zero and all completed summaries using a
  static source mask.
- The mixed boundary candidate replaces the ordinary candidate before the
  associative-memory read.
- No Python loop drives model ticks; all updates execute inside the existing
  BrainState loop/scan drivers.

## Compatibility

The `none` arm allocates no latent-residual parameters or states and preserves
the prior tree and outputs. Configuration, CLI, result JSON, architecture
reports, snapshots, and new checkpoint metadata include the selected mixer and
block size.

## Structural Tests

- default-path parameter/state/output equivalence;
- validation and zero-memory rejection;
- query cache/reset;
- full and partial block boundaries;
- source mask and block-specific pseudo-query selection;
- current-candidate hidden-state and pp-prop parameter routing;
- reset/snapshot/restore;
- configuration, CLI, report, and checkpoint compatibility;
- compiled packed and selected trajectories.

## Promotion

Compare `none` against `attention_residual` at block size 10 on the accepted
Stage 2 stack. Alternative block sizes are not tuned unless the preregistered
arm passes structural gates and narrowly misses only the behavioral threshold.
