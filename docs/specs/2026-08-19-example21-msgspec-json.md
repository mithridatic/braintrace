# Example 21 msgspec JSON serialization

Status: approved and implemented
Date: 2026-08-19
Branch: `feat/example21-row-refinement`

## Objective

Use `msgspec.json` for JSON decoding and encoding on the Example 21 production
path to reduce host-side serialization overhead during dataset loading,
fingerprint construction, and result artifact writing.

## Scope

- Migrate `21-latent-reasoning-in-context.py` and its imported
  `latent_workspace_task.py` helper.
- Keep the existing JSON object/array schemas and file names.
- Preserve sorted-key encoding for content fingerprints.
- Keep test-only fixture construction on the standard library unless it is
  exercising a migrated production function.
- Add `msgspec` to the package's runtime dependencies.

Compact output is acceptable for result and manifest artifacts. Consumers must
continue to treat them as ordinary JSON documents rather than depending on
indentation or whitespace.

## Verification

Run the co-located Example 21/task tests, import the production modules with
the project environment, and check the resulting diff for whitespace errors.
