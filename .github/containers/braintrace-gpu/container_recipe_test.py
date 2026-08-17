"""Tests for the locked temporal-credit GPU container recipe."""

from pathlib import Path


DOCKERFILE = Path(__file__).with_name("Dockerfile")


def test_recipe_pins_base_python_jax_and_scientific_dependencies() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "python:3.14.0-slim-trixie@sha256:" in text
    assert '"jax[cuda12]==0.11.0"' in text
    assert '"brainstate==0.5.3"' in text
    assert '"brainevent[cuda12]==0.2.1"' in text
    assert '"numpy==2.4.6"' in text


def test_recipe_records_source_revision_and_verifies_runtime() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    assert "org.opencontainers.image.revision" in text
    assert "sys.version_info[:2] == (3, 14)" in text
    assert "jax.__version__ == '0.11.0'" in text


def test_recipe_keeps_revision_metadata_after_reusable_layers() -> None:
    text = DOCKERFILE.read_text(encoding="utf-8")

    git_install = text.index("apt-get install --no-install-recommends --yes git")
    source_copy = text.index("COPY . /opt/braintrace")
    source_install = text.index('"/opt/braintrace[cuda12,examples]"')
    revision_arg = text.index("ARG BRAINTRACE_SOURCE_COMMIT")
    revision_label = text.index("LABEL org.opencontainers.image.revision")

    assert git_install < source_copy < source_install < revision_arg < revision_label
