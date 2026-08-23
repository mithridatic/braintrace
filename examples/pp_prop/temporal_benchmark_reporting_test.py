"""Tests for strict evidence serialization and immutable packaging."""

import msgspec_json
from unittest.mock import patch

import pytest

from temporal_benchmark_reporting import (
    package_release,
    sha256_file,
    source_fingerprint,
    write_result,
)


def test_strict_result_writer_rejects_nonfinite_values(tmp_path) -> None:
    with pytest.raises(ValueError):
        write_result(tmp_path / "result.json", {"metric": float("nan")})


def test_sealed_result_requires_confirmed_clean_source(tmp_path) -> None:
    payload = {
        "sealed_test": True,
        "environment": {"source_dirty": True},
    }

    with pytest.raises(ValueError, match="clean source"):
        write_result(tmp_path / "result.json", payload)


def test_release_packaging_fails_closed_then_hashes_files(tmp_path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(msgspec_json.dumps({"status": "completed"}), encoding="utf-8")
    destination = tmp_path / "release.tar.gz"

    with pytest.raises(ValueError, match="passed scientific gates"):
        package_release((raw,), destination, {"passed": False})

    package_release((raw,), destination, {"passed": True})
    assert destination.is_file()
    assert len(sha256_file(destination)) == 64
    assert (
        (tmp_path / "SHA256SUMS").read_text(encoding="utf-8").endswith("  raw.json\n")
    )


def test_container_without_git_uses_explicit_host_provenance(tmp_path) -> None:
    environment = {
        "BRAINTRACE_SOURCE_COMMIT": "a" * 40,
        "BRAINTRACE_SOURCE_DIRTY": "false",
    }
    with (
        patch("temporal_benchmark_reporting._git_output", return_value=None),
        patch.dict("os.environ", environment, clear=False),
    ):
        fingerprint = source_fingerprint(tmp_path)

    assert fingerprint == {"source_commit": "a" * 40, "source_dirty": False}
