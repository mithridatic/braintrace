"""Fail-closed result serialization, fingerprints, hashes, and release packaging."""

from __future__ import annotations

import hashlib
import importlib.metadata
import msgspec_json
import os
import platform
import subprocess
import tarfile
from pathlib import Path
from typing import Iterable

import jax


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_output(source_root: Path, *arguments: str) -> str | None:
    command = [
        "git",
        "-c",
        f"safe.directory={source_root.as_posix()}",
        "-C",
        str(source_root),
        *arguments,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def source_fingerprint(source_root: Path) -> dict[str, object]:
    """Record commit and dirty state without claiming unavailable provenance."""
    commit = _git_output(source_root, "rev-parse", "HEAD") or os.environ.get(
        "BRAINTRACE_SOURCE_COMMIT"
    )
    status = _git_output(source_root, "status", "--porcelain")
    dirty_environment = os.environ.get("BRAINTRACE_SOURCE_DIRTY")
    dirty = None if status is None else bool(status)
    if dirty is None and dirty_environment in {"true", "false"}:
        dirty = dirty_environment == "true"
    return {
        "source_commit": commit,
        "source_dirty": dirty,
    }


def environment_fingerprint(source_root: Path) -> dict[str, object]:
    """Capture the required software, device, driver, CUDA, and image identity."""
    device = jax.devices()[0]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": _package_version("jax"),
        "jaxlib": _package_version("jaxlib"),
        "braintrace": _package_version("braintrace"),
        "brainstate": _package_version("brainstate"),
        "brainevent": _package_version("brainevent"),
        "nvidia_cuda_runtime_cu12": _package_version("nvidia-cuda-runtime-cu12"),
        "backend": jax.default_backend(),
        "device": getattr(device, "device_kind", str(device)),
        "nvidia_driver": os.environ.get("NVIDIA_DRIVER_VERSION"),
        "cuda_version": os.environ.get("CUDA_VERSION"),
        "container_image_digest": os.environ.get("BRAINTRACE_IMAGE_DIGEST"),
        **source_fingerprint(source_root),
    }


def write_result(path: Path, payload: dict[str, object]) -> None:
    """Write strict JSON after rejecting dirty sealed evidence."""
    environment = payload.get("environment")
    sealed = payload.get("sealed_test") is True
    if (
        sealed
        and isinstance(environment, dict)
        and environment.get("source_dirty") is not False
    ):
        raise ValueError("Sealed results require a confirmed clean source tree. Provide the required value for Sealed results.")
    serialized = msgspec_json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sha256sums(paths: Iterable[Path], destination: Path) -> None:
    """Write stable checksums for the supplied artifacts."""
    ordered = sorted(paths, key=lambda path: path.name)
    lines = [f"{sha256_file(path)}  {path.name}" for path in ordered]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_release(
    files: Iterable[Path], destination: Path, gate_document: dict[str, object]
) -> Path:
    """Create an immutable tar.gz only when all episodic scientific gates pass."""
    artifacts = tuple(files)
    if gate_document.get("passed") is not True:
        raise ValueError("Release packaging requires passed scientific gates. Provide the required value for Release packaging.")
    if not artifacts or any(not path.is_file() for path in artifacts):
        raise ValueError("Release files must all exist. Set Release files to all exist.")
    checksums = destination.with_name("SHA256SUMS")
    write_sha256sums(artifacts, checksums)
    with tarfile.open(destination, mode="x:gz") as archive:
        for path in (*artifacts, checksums):
            archive.add(path, arcname=path.name, recursive=False)
    return destination
