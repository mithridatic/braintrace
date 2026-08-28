"""Repository-wide layout and random-generation convention checks."""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent
GENERATIVE_JAX_RANDOM = re.compile(
    r"jax\.random\.(?:PRNGKey|bernoulli|bits|categorical|choice|fold_in|"
    r"normal|permutation|randint|split|truncated_normal|uniform)\b"
)
SUPPRESSION_DIRECTIVE = re.compile(
    r"#\s*(?:noqa\b|ruff:\s*noqa\b|type:\s*ignore\b|pyright:\s*ignore\b)"
)


def source_files() -> list[pathlib.Path]:
    return [
        path
        for base in (ROOT / "braintrace", ROOT / "examples")
        for path in base.rglob("*.py")
    ]


def linted_source_files() -> list[pathlib.Path]:
    return [
        path
        for base in (
            ROOT / "braintrace",
            ROOT / "examples",
            ROOT / "docs/diagnostics",
        )
        for path in base.rglob("*.py")
    ]


def test_tests_are_colocated_with_suffix_names() -> None:
    test_directories = sorted(
        path.relative_to(ROOT)
        for base in (ROOT / "braintrace", ROOT / "examples")
        for path in base.rglob("tests")
        if path.is_dir()
    )
    prefixed = sorted(
        path.relative_to(ROOT)
        for path in source_files()
        if path.name.startswith("test_")
    )

    assert test_directories == []
    assert prefixed == []


def test_generation_uses_brainstate_random() -> None:
    violations = sorted(
        path.relative_to(ROOT)
        for path in source_files()
        if GENERATIVE_JAX_RANDOM.search(path.read_text(encoding="utf-8"))
    )

    assert violations == []


def test_source_has_no_inline_lint_suppressions() -> None:
    violations = sorted(
        path.relative_to(ROOT)
        for path in linted_source_files()
        if SUPPRESSION_DIRECTIVE.search(path.read_text(encoding="utf-8"))
    )

    assert violations == []


def test_documented_example21_image_paths_exist() -> None:
    documents = (
        ROOT / "README.md",
        ROOT / "examples/pp_prop/README.md",
        ROOT / "docs/evidence/example21.md",
        ROOT / ".github/containers/braintrace-example21/Dockerfile",
    )
    referenced = {
        match.group(1).rstrip("/.,`")
        for document in documents
        for match in re.finditer(
            r"/opt/braintrace(?:/([A-Za-z0-9_.\-/]+))?",
            document.read_text(encoding="utf-8"),
        )
        if match.group(1)
    }
    missing = sorted(path for path in referenced if not (ROOT / path).exists())

    assert missing == []
