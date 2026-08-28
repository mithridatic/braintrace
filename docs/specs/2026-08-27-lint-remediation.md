# Lint remediation

Status: implementation
Date: 2026-08-27
Issue: BRA-35

## Purpose

The repository must report lint and type errors directly. Source files must not
hide these reports with inline suppression directives.

## Scope

- Ruff checks all tracked Python files in `braintrace`, `examples`, and
  `docs/diagnostics`.
- Basedpyright checks production modules under `braintrace`.
- Mypy keeps its current production-module scope.
- Repository convention tests reject inline Ruff, Flake8, mypy, and pyright
  suppression directives in tracked Python files.
- Tests keep their current runtime behavior and sibling `*_test.py` layout.

Test-only modules and `braintrace/_testing` stay outside the static type-check
scope. They contain deliberate invalid-input calls and dynamic fixtures. Pytest
continues to check their behavior.

## Required changes

1. Add Ruff and basedpyright to the development dependencies and continuous
   integration checks.
2. Replace the Flake8 pre-commit hook with Ruff so there is one style-lint path.
3. Remove inline suppression directives. Use explicit types, casts, imports,
   and data validation where a checked production path needs them.
4. Keep explicit failures. Do not add silent fallbacks.
5. Preserve model execution, JAX compilation boundaries, data formats, and
   public interfaces.

## Acceptance checks

- `python -m ruff check braintrace examples docs/diagnostics repo_conventions_test.py`
- `python -m basedpyright braintrace`
- `python -m mypy braintrace`
- `python -m pytest repo_conventions_test.py -q`
- Focused sibling tests pass for production modules changed to remove a
  suppression.
- A repository search finds no inline `noqa`, `type: ignore`,
  `pyright: ignore`, or `ruff: noqa` directive in tracked Python.

## Edge cases

- Negative tests must still pass invalid values to the code under test.
- Example scripts that add a local directory to `sys.path` must still import
  their sibling helper modules when run as files.
- Type-only circular imports must not become runtime imports.
- Dynamic third-party APIs must keep explicit runtime validation before a cast.
