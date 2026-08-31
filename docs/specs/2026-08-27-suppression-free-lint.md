# Suppression-Free Lint

## Purpose

Remove source-level lint and type-check suppressions. Fix each reported problem
without changing public behavior.

## Baseline

The repository contains 192 Python lines with one or more of these directives
in 75 files:

- `noqa`
- `ruff: noqa`
- `type: ignore`
- `pyright: ignore`

The files are in `braintrace`, `examples`, and `docs`. The configured gates are
Flake8 in pre-commit, mypy in CI, and the repository convention test. Prior
verification records also use Ruff and BasedPyright.

`pragma: no cover` is a coverage directive. It is not in scope.

## Required behavior

1. Python source must contain none of the four suppression directives in the
   baseline list.
2. Code must pass Ruff, BasedPyright, mypy, Flake8, and repository convention
   checks without new file exclusions or rule exclusions.
3. Existing public APIs, exceptions, wire formats, persistence formats, and
   numerical behavior must stay unchanged.
4. Import changes must preserve script execution for examples that support it.
5. Type fixes must represent runtime values accurately. They must not replace
   explicit failures with fallbacks or unchecked casts.
6. Repeated model execution must keep its current Brainstate transform and JAX
   compilation boundaries.
7. Random operations must continue to use `brainstate.random`.

## Design

Use the smallest direct fix for each diagnostic:

- Move imports to their normal module position when initialization order does
  not require a deferred import.
- Re-export public names explicitly instead of suppressing unused-import
  diagnostics.
- Replace private-member access with an existing public operation. If no public
  operation exists, keep the access and express the ownership in the interface.
- Narrow optional and union values before use.
- Give containers and callbacks accurate annotations at their definition.
- Use package-aware imports for example helper modules while preserving direct
  script execution.
- Catch the narrow exception that a failure path expects. Keep broad exception
  handling only when the contract requires one error boundary, and structure
  that boundary so lint tools can verify it.

Do not add replacement suppressions, global exclusions, compatibility shims,
or dependencies that only hide diagnostics.

## Verification

Run these checks after implementation:

1. Search all Python files for the four forbidden directives and require zero
   matches.
2. Run `ruff check .`.
3. Run BasedPyright over `braintrace`, `examples`, and Python tools under
   `docs`.
4. Run `python -m mypy braintrace`.
5. Run Flake8 through `pre-commit run flake8 --all-files`.
6. Run `python -m pytest repo_conventions_test.py -q`.
7. Run focused sibling tests for every module whose executable behavior
   changes.
8. Run the full test suite with coverage and require more than 90 percent
   meaningful source coverage.

## Edge cases

- Package imports and direct example execution resolve helper modules from
  different roots.
- Optional JAX values may be narrowed only after runtime validation.
- Public re-exports must remain present in `__all__` and at package import time.
- Exception cleanup must preserve the original exception type and message.
- Test-only access to implementation details must not create a new public API.
