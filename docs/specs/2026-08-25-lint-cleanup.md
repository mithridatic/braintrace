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

## Measured baseline

The first repository-wide basedpyright run analyzed 582 files and reported
7,267 errors and 79,193 warnings. It also reported missing sources for installed
development tools such as pytest and setuptools. Keep the full-repository target
and remove findings in reviewable groups. Do not hide findings with broad
configuration exclusions or new inline suppressions.

After Ruff reaches zero findings, remove the remaining inline Ruff suppressions
before the larger type cleanup. Replace late imports with explicit dynamic
imports only where the file must update `sys.path` first. Use ordinary top-level
imports everywhere else.
