# Lint cleanup

## Scope

The full repository is the lint target. This includes the shipped package,
co-located tests, examples, documentation sources, and notebooks.

## Requirements

- `ruff check .` reports no findings.
- basedpyright reports no errors or warnings for project-owned Python files.
- Existing suppression comments are removed when the underlying code or type
  can be expressed without a suppression.
- Public behavior and the legacy lazy-import API remain unchanged.

## Verification

Run `ruff check .` and basedpyright. Run focused tests for any behavior changed
during cleanup.
