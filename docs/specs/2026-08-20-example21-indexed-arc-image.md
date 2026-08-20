# Example 21 indexed ARC image

Status: approved
Date: 2026-08-20

## Goal

Use one reusable Docker image for every Example 21 configuration. The image
contains BrainTrace, CUDA dependencies, the immutable ARC-AGI-1 corpus, and a
prevalidated dataset index.

## Build contract

- The generic GPU image remains the dependency base.
- A tracked Example 21 Dockerfile receives ARC-AGI-1 through a named BuildKit
  context; source data does not enter Git or the ordinary repository context.
- The image records the ARC source revision and BrainTrace source revision in
  OCI labels.
- Image construction parses, validates, hashes, fingerprints, deduplicates,
  checks split leakage, and applies declared exclusions exactly once.
- The build fails unless the admitted pools contain 399 training tasks and 400
  evaluation tasks.

## Runtime contract

- Example 21 loads one prevalidated index file per split with ``msgspec``.
- Runtime loading does not enumerate, reopen, hash, or fingerprint the 800
  original task files.
- The index preserves original per-file hashes, task fingerprints, exclusions,
  duplicate/rejection accounting, source metadata, and normalized ARC tasks.
- Index integrity is checked using its schema, source declaration, source
  revision, and a SHA-256 digest of the encoded payload.
- All Example 21 parameter configurations use the same indexed image and no ARC
  bind mount.

## Acceptance

- Co-located tests prove index round-trip equivalence to ordinary loading and
  prove the indexed loader does not call file enumeration, file hashing, or
  canonical task fingerprinting.
- The image contains 399 admitted training tasks and 400 evaluation tasks.
- A timed container check loads both splits and validates no leakage.
- The documented 390-tick command uses the indexed image without a dataset
  mount.
- Superseded Example 21-specific image tags are removed only after the new image
  passes verification.
