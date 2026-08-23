"""Standalone fixed-profile Example 15 evidence runner."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import msgspec_json
import os
import pathlib
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

EXAMPLE15_RUN_SCHEMA_VERSION = 1
EXAMPLE15_RUN_KIND = "braintrace_example15_static_control_run"


def fixed_config_document() -> dict[str, object]:
    """Return the immutable Example 15 numerical profile."""
    return {
        "seeds": [0, 1, 2],
        "n_epochs": 5,
        "batch_size": 32,
        "n_rec": 96,
        "degree": 8,
        "n_step": 30,
        "final_window": 5,
        "learning_rate": 0.003,
        "decay_or_rank": 0.95,
        "clip_norm": 1.0,
        "sparse_backend": "jax_raw",
        "recurrent_scale_basis": "neurons",
        "train_examples": 288,
        "validation_examples": 72,
    }


def sha256_file(path: pathlib.Path) -> str:
    """Return one file's lowercase SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(source_root: pathlib.Path, *arguments: str) -> str | None:
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


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def source_fingerprint(source_root: pathlib.Path) -> dict[str, object]:
    """Return commit and dirty state without inventing unavailable provenance."""
    commit = _git_output(source_root, "rev-parse", "HEAD") or os.environ.get(
        "BRAINTRACE_SOURCE_COMMIT"
    )
    status = _git_output(source_root, "status", "--porcelain")
    dirty: bool | None = None if status is None else bool(status)
    environment_dirty = os.environ.get("BRAINTRACE_SOURCE_DIRTY")
    if dirty is None and environment_dirty in {"true", "false"}:
        dirty = environment_dirty == "true"
    return {"source_commit": commit, "source_dirty": dirty}


def _load_example(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("_example15_static_control", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Example 15 from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_example_profile(example: Any) -> None:
    expected = fixed_config_document()
    config = example._RunConfig(seed=0, n_epochs=5, batch_size=32)
    actual = {
        "seeds": list(example.SEEDS),
        "n_epochs": example.N_EPOCH,
        "batch_size": 32,
        "n_rec": config.n_rec,
        "degree": config.degree,
        "n_step": config.n_step,
        "final_window": config.final_window,
        "learning_rate": config.learning_rate,
        "decay_or_rank": config.decay_or_rank,
        "clip_norm": config.clip_norm,
        "sparse_backend": config.sparse_backend,
        "recurrent_scale_basis": config.recurrent_scale_basis,
        "train_examples": 288,
        "validation_examples": 72,
    }
    if actual != expected:
        raise RuntimeError(
            "Example 15 numerical defaults differ from the fixed profile"
        )


def _environment(source_root: pathlib.Path, image_digest: str) -> dict[str, object]:
    import jax

    device = jax.devices()[0]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "jax": _package_version("jax"),
        "jaxlib": _package_version("jaxlib"),
        "braintrace": _package_version("braintrace"),
        "brainstate": _package_version("brainstate"),
        "scikit_learn": _package_version("scikit-learn"),
        "backend": jax.default_backend(),
        "device": getattr(device, "device_kind", str(device)),
        "container_image_digest": image_digest,
        **source_fingerprint(source_root),
    }


def run_fixed_example15(
    example_script: pathlib.Path,
    source_root: pathlib.Path,
    container_image_digest: str,
    device: str,
) -> dict[str, object]:
    """Execute unchanged Example 15 and return provenance-bound raw evidence."""
    if not container_image_digest.strip():
        raise ValueError("container_image_digest is required")
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
    example = _load_example(example_script)
    _verify_example_profile(example)
    import jax

    if device == "gpu" and jax.devices()[0].platform not in {"gpu", "cuda", "rocm"}:
        raise RuntimeError(
            f"requested device gpu, bound backend is {jax.default_backend()}"
        )
    with contextlib.redirect_stdout(sys.stderr):
        result = example.main(n_epochs=5, batch_size=32, plot=False)
    return {
        "schema_version": EXAMPLE15_RUN_SCHEMA_VERSION,
        "kind": EXAMPLE15_RUN_KIND,
        "development_only": True,
        "sealed_test": False,
        "accepted_baseline": False,
        "environment": _environment(source_root, container_image_digest),
        "example_source_sha256": sha256_file(example_script),
        "fixed_config": fixed_config_document(),
        "result": {"status": "completed", **result},
    }


def _parser() -> argparse.ArgumentParser:
    source_root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=pathlib.Path, default=source_root)
    parser.add_argument(
        "--example-script",
        type=pathlib.Path,
        default=pathlib.Path(__file__).with_name("15-sparse-temporal-learning.py"),
    )
    parser.add_argument("--container-image-digest", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="gpu")
    parser.add_argument("--json-output", type=pathlib.Path, required=True)
    return parser


def _write_json(path: pathlib.Path, document: Mapping[str, object]) -> None:
    serialized = msgspec_json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed profile once and write its raw evidence artifact."""
    values = _parser().parse_args(argv)
    if not values.example_script.is_file():
        raise FileNotFoundError(values.example_script)
    document = run_fixed_example15(
        values.example_script.resolve(),
        values.source_root.resolve(),
        values.container_image_digest,
        values.device,
    )
    _write_json(values.json_output.resolve(), document)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
