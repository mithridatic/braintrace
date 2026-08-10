# Configurable benchmark GPU defaults

Date: 2026-08-10

## Objective

Make a bare invocation of `16-configurable-sparse-benchmark.py` exercise the
representative GPU time-to-quality path established by the 2026-08-10 scaling
sweep.

## Defaults

- Mode: `validation-target`.
- Neurons: 32,768.
- Degree: 8.
- Target validation accuracy: 1.0.
- Device: `gpu`, which must fail rather than silently fall back to CPU.
- Maximum epochs: 5.
- Validation interval: 1 update.

## Compatibility

Every setting remains overridable through its existing command-line option.
CPU measurements must pass `--device cpu`; fixed-work measurements must pass
`--mode fixed-work`.

## Acceptance

- `parse_config([])` returns the defaults above.
- Configuration serialization still round-trips explicit alternatives.
- The README describes the new bare-invocation behavior and explicit override
  paths.
- Focused configuration and command-orchestration tests pass.
