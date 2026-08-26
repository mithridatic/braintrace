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

Search tracked Python and configuration files for `noqa`, `type: ignore`,
`pyright: ignore`, and `ruff: noqa`. The search must return no suppression
directives. Explanatory prose that names a directive is not a suppression.

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

The first review found 185 suppression lines after Ruff reported no findings.
Ruff success alone is therefore not sufficient evidence for this change.

Modernize the type aliases in `_typing.py` as a separate mechanical unit.
Replace deprecated `typing` container aliases and `Union` with built-in generic
syntax. Preserve alias names, accepted values, and runtime behavior. Keep the
PyTree `Any` boundary until a later unit can define its required recursive type.

Replace the PyTree `Any` boundary with `object`. A closed recursive container
union is not valid because JAX permits registered custom PyTree node classes and
arbitrary leaf types. The `object` boundary preserves those values and requires
callers to narrow a value before they use type-specific operations. Add a test
that fixes this boundary and includes built-in containers, an arbitrary leaf,
and a custom node value.

## Type-cleanup sequence

Use the project development environment when running basedpyright so installed
test dependencies resolve. Clean one source module and its sibling test module
as one review unit. Start with `_version.py` and `_version_test.py`, because this
pair has no runtime dependency on the recurrent-network execution paths. Keep
TOML values behind an `object` boundary, then apply a typed schema for the
fields that these metadata tests own. This preserves the test assertions and
removes unknown types without a suppression.
