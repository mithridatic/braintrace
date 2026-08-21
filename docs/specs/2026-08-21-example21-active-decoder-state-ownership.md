# Example 21 active decoder state ownership

## Problem

`LatentWorkspaceModel` constructs the legacy CP decoder parameters even when
`decoder_mode="row_refinement"`. The row-refinement trace never reads those
parameters, so compiler state reconciliation emits four `STATE_MISMATCH`
warnings for `readout_projection`, `height_head`, `width_head`, and
`color_factor_head`. The compiler is correct to warn about model-owned states
that are absent from the traced program; globally suppressing that diagnostic
would hide genuine dead-parameter defects.

## Required behavior

The model owns only the decoder selected by `decoder_mode`:

- `legacy_cp` constructs the projection and three CP output heads.
- `row_refinement` does not construct those four modules and continues to own
  its refinement or attention-residual parameters.
- Compilation of either supported decoder emits no state-mismatch warning.
- Active decoder parameters retain their current ETP route classifications.
- Parameter snapshots, counts, and checkpoints naturally contain only the
  active architecture's parameters. Checkpoint restore remains exact and
  rejects a checkpoint whose parameter tree does not match the target model.
- Initialisation remains deterministic for a fixed configuration and seed.

The compiler warning and reconciliation policy are unchanged.

## Compatibility

This changes the row-refinement parameter schema by removing four parameters
that were unreachable, never optimized, and already excluded from its compiler
qualification manifest. Existing row-refinement checkpoints containing those
dead leaves are intentionally rejected by the existing exact-schema checkpoint
loader rather than silently migrated. Legacy CP checkpoints and behavior retain
their existing parameter schema.

## Verification

- Reproduce the current four warnings in a test before implementation.
- Assert row-refinement compilation emits no `STATE_MISMATCH` diagnostics or
  Python warnings and contains its active heads.
- Assert row-refinement snapshots omit all four legacy parameter prefixes.
- Assert legacy CP snapshots and compiled parameters retain all four prefixes.
- Run the co-located model and Example 21 checkpoint/qualification tests with
  coverage for the modified module.
