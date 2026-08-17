"""Authenticated host launcher for Example 21 capability-gate runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal, Mapping, Sequence


LaunchTarget = Literal[
    "one_update",
    "stability_256",
    "formal_gate_a",
    "gate_b_init",
    "formal_gate_b",
]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
_TARGETS: tuple[LaunchTarget, ...] = (
    "one_update",
    "stability_256",
    "formal_gate_a",
    "gate_b_init",
    "formal_gate_b",
)
_GATE_B_TARGETS = frozenset({"gate_b_init", "formal_gate_b"})
_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_HEAD_PATTERN = re.compile(r"[0-9a-f]{40}")
_DEFAULT_IMAGE = "braintrace-gpu:0.11.0-py314"
_GATE_MODULE = "examples.pp_prop.latent_workspace_binding_gate"
_DEPTH_GATE_MODULE = "examples.pp_prop.latent_workspace_depth_gate"
_GATE_A_SOURCE_COMMIT = "4737e9172b1c6ca99347af5b2c83fc795a294a16"
_GATE_A_RESULT_SHA256 = (
    "3a585e739715b31757082b50fe57b98ca50107891f7c79edaa7e5e54c90ad632"
)
_GATE_A_MANIFEST_SHA256 = (
    "69d690daa5023f5b3ce22b0e65ea09a1a6706687d792e998651422f6d6ea15cf"
)
_GATE_A_BUNDLE_SHA256 = (
    "ba850a205c4691d573facef7b8e90cabd4824905c73fcd4f6add29293cd95875"
)
_GATE_A_DIRECTORY = Path("var/example21-binding-gate")


class ProvenanceError(RuntimeError):
    """Report a fail-closed provenance or artifact-integrity violation.

    Notes
    -----
    This distinct exception lets the command-line entry point distinguish an
    authenticated-launch failure from a scientific Gate A failure.
    """


@dataclass(frozen=True)
class LaunchConfig:
    """Configure one fixed Example 21 authenticated launch target.

    Parameters
    ----------
    target : {"one_update", "stability_256", "formal_gate_a", "gate_b_init", "formal_gate_b"}
        Fixed preregistered target. Target selection changes only which hardcoded
        Gate entry point runs; it exposes no topology or training-budget knobs.
    repo_root : pathlib.Path
        Expected worktree root mounted at ``/work``.
    output_dir : pathlib.Path
        Ignored repository directory for the result, preflight, and manifest.
    image : str, default="braintrace-gpu:0.11.0-py314"
        Local image reference resolved once to an immutable image ID.
    cache_volume : str, default="braintrace-example21-jax-cache"
        Named JAX compilation-cache volume.
    """

    target: LaunchTarget
    repo_root: Path
    output_dir: Path
    image: str = _DEFAULT_IMAGE
    cache_volume: str = "braintrace-example21-jax-cache"

    def __post_init__(self) -> None:
        if self.target not in _TARGETS:
            raise ValueError(f"target must be one of {_TARGETS!r}")
        root = Path(self.repo_root).resolve()
        output = Path(self.output_dir).resolve()
        if not output.is_relative_to(root):
            raise ValueError("output_dir must be contained by repo_root")
        if output == root:
            raise ValueError("output_dir cannot be the repository root")
        if not self.image.strip():
            raise ValueError("image must be nonempty")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.cache_volume):
            raise ValueError("cache_volume contains unsupported characters")
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(self, "output_dir", output)


@dataclass(frozen=True)
class TargetPaths:
    """Hold host and container paths for one immutable launch bundle.

    Attributes
    ----------
    result, preflight, manifest : pathlib.Path
        Fixed host paths for the three companion artifacts.
    container_result : pathlib.PurePosixPath
        Result path below the container's ``/work`` mount.
    """

    result: Path
    preflight: Path
    manifest: Path
    container_result: PurePosixPath


@dataclass(frozen=True)
class CommandRecord:
    """Retain exact subprocess inputs, outputs, status, and byte digests.

    Attributes
    ----------
    argv : tuple of str
        Shell-free command arguments.
    cwd : str
        Host working directory.
    environment : mapping of str to str
        Explicit environment evidence relevant to the command.
    returncode : int
        Process exit status.
    stdout, stderr : str
        Exact captured streams.
    stdout_sha256, stderr_sha256 : str
        SHA-256 digests of the captured UTF-8 streams.
    wall_seconds : float
        Host elapsed duration.
    """

    argv: tuple[str, ...]
    cwd: str
    environment: Mapping[str, str]
    returncode: int
    stdout: str
    stderr: str
    stdout_sha256: str
    stderr_sha256: str
    wall_seconds: float


@dataclass(frozen=True)
class SourceSnapshot:
    """Retain an authenticated worktree source snapshot.

    Attributes
    ----------
    root, common_git_dir, git_dir : pathlib.Path
        Resolved worktree and Git administrative paths.
    head : str
        Full lowercase source commit.
    clean : bool
        Whether the complete worktree status is empty.
    records : tuple of CommandRecord
        Exact Git command evidence.
    """

    root: Path
    common_git_dir: Path
    git_dir: Path
    head: str
    clean: bool
    records: tuple[CommandRecord, ...]


@dataclass(frozen=True)
class ImageIdentity:
    """Retain the immutable local image identity and OCI source revision.

    Attributes
    ----------
    reference : str
        Mutable local image reference that was inspected.
    image_id : str
        Resolved immutable ``sha256:`` image ID.
    revision : str
        OCI source revision label.
    record : CommandRecord
        Exact image-inspection command evidence.
    """

    reference: str
    image_id: str
    revision: str
    record: CommandRecord


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of exact artifact bytes.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing file to hash.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_strict_json(path: str | Path) -> dict[str, Any]:
    """Load one JSON object while rejecting non-standard numeric constants.

    Parameters
    ----------
    path : str or pathlib.Path
        JSON artifact to load.

    Returns
    -------
    dict
        Parsed top-level object.
    """

    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    value = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject,
        object_pairs_hook=unique_pairs,
    )
    if not isinstance(value, dict):
        raise ValueError("artifact must contain a top-level JSON object")
    return value


def write_strict_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Atomically write one standards-compliant JSON object.

    Parameters
    ----------
    path : str or pathlib.Path
        Final artifact path.
    value : mapping
        JSON-compatible value. NaN and infinity are rejected.

    Returns
    -------
    pathlib.Path
        Final artifact path.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = json.dumps(
        value,
        allow_nan=False,
        indent=2,
        sort_keys=True,
        separators=(",", ": "),
    ) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def target_paths(
    config: LaunchConfig, head: str, target: LaunchTarget | None = None
) -> TargetPaths:
    """Return deterministic paths for a source revision and target.

    Parameters
    ----------
    config : LaunchConfig
        Launcher configuration.
    head : str
        Full 40-hex source revision.
    target : launch target, optional
        Override used to locate prerequisite admission bundles.

    Returns
    -------
    TargetPaths
        Host and container artifact paths.
    """

    if not _HEAD_PATTERN.fullmatch(head):
        raise ValueError("head must be a full lowercase 40-hex commit")
    selected = config.target if target is None else target
    if selected not in _TARGETS:
        raise ValueError(f"unknown target {selected!r}")
    stem = f"{head}-{selected.replace('_', '-')}"
    result = config.output_dir / f"{stem}.json"
    relative = result.relative_to(config.repo_root).as_posix()
    return TargetPaths(
        result=result,
        preflight=result.with_suffix(".preflight.json"),
        manifest=result.with_suffix(".manifest.json"),
        container_result=PurePosixPath("/work") / relative,
    )


def _container_path(config: LaunchConfig, path: Path) -> PurePosixPath:
    relative = path.resolve().relative_to(config.repo_root).as_posix()
    return PurePosixPath("/work") / relative


def _gate_a_artifact_paths(config: LaunchConfig) -> TargetPaths:
    stem = f"{_GATE_A_SOURCE_COMMIT}-formal-gate-a"
    result = config.repo_root / _GATE_A_DIRECTORY / f"{stem}.json"
    return TargetPaths(
        result=result,
        preflight=result.with_suffix(".preflight.json"),
        manifest=result.with_suffix(".manifest.json"),
        container_result=_container_path(config, result),
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_real(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _json_exact(left: Any, right: Any) -> bool:
    try:
        return json.dumps(
            left, allow_nan=False, sort_keys=True, separators=(",", ":")
        ) == json.dumps(
            right, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return False


def _sanitized_host_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"} or name.startswith(
            "GIT_CONFIG_"
        ):
            del environment[name]
    return environment


def _run(
    runner: CommandRunner,
    argv: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    recorded_environment: Mapping[str, str] | None = None,
) -> CommandRecord:
    start = time.perf_counter()
    completed = runner(
        list(argv),
        cwd=str(cwd),
        env=dict(environment),
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    return CommandRecord(
        argv=tuple(map(str, argv)),
        cwd=str(cwd),
        environment=dict(recorded_environment or {}),
        returncode=int(completed.returncode),
        stdout=stdout,
        stderr=stderr,
        stdout_sha256=_sha256_text(stdout),
        stderr_sha256=_sha256_text(stderr),
        wall_seconds=elapsed,
    )


def _require_success(record: CommandRecord, label: str) -> str:
    if record.returncode != 0:
        raise ProvenanceError(
            f"{label} failed with exit {record.returncode}: {record.stderr.strip()}"
        )
    return record.stdout.strip()


def _host_source_snapshot(
    config: LaunchConfig, runner: CommandRunner
) -> SourceSnapshot:
    environment = _sanitized_host_environment()
    commands = (
        ["git", "rev-parse", "--show-toplevel"],
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        ["git", "rev-parse", "--absolute-git-dir"],
        ["git", "rev-parse", "HEAD"],
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
    )
    records = tuple(
        _run(
            runner,
            command,
            cwd=config.repo_root,
            environment=environment,
            recorded_environment={"git_override_variables_removed": "true"},
        )
        for command in commands
    )
    outputs = tuple(
        _require_success(record, "host Git source preflight") for record in records
    )
    root = Path(outputs[0]).resolve()
    common = Path(outputs[1]).resolve()
    git_dir = Path(outputs[2]).resolve()
    head = outputs[3].lower()
    if root != config.repo_root:
        raise ProvenanceError(f"unexpected worktree root {root}")
    if not git_dir.is_relative_to(common):
        raise ProvenanceError("worktree Git directory is outside the common Git directory")
    if not _HEAD_PATTERN.fullmatch(head):
        raise ProvenanceError("live HEAD is not a full lowercase 40-hex commit")
    clean = outputs[4] == ""
    return SourceSnapshot(root, common, git_dir, head, clean, records)


def _inspect_image(
    config: LaunchConfig, head: str, runner: CommandRunner
) -> ImageIdentity:
    environment = _sanitized_host_environment()
    record = _run(
        runner,
        ["docker", "image", "inspect", config.image],
        cwd=config.repo_root,
        environment=environment,
    )
    raw = _require_success(record, "docker image inspect")
    try:
        values = json.loads(raw)
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError("expected exactly one local image")
        image = values[0]
        image_id = str(image["Id"])
        revision = str(image["Config"]["Labels"]["org.opencontainers.image.revision"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"invalid local image inspection: {error}") from error
    if not _IMAGE_ID_PATTERN.fullmatch(image_id):
        raise ProvenanceError("local image ID is not an immutable sha256 digest")
    if revision != head:
        raise ProvenanceError(
            f"OCI revision {revision!r} does not equal clean HEAD {head!r}"
        )
    return ImageIdentity(config.image, image_id, revision, record)


def _container_environment(head: str, image_id: str, git_dir: str) -> dict[str, str]:
    return {
        "BRAINTRACE_SOURCE_COMMIT": head,
        "BRAINTRACE_SOURCE_DIRTY": "0",
        "BRAINTRACE_IMAGE_DIGEST": image_id,
        "GIT_DIR": git_dir,
        "GIT_WORK_TREE": "/work",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": "/work",
    }


def _docker_base(
    config: LaunchConfig,
    *,
    image_id: str,
    common_git_dir: Path,
    container_environment: Mapping[str, str],
    gpu: bool,
) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--workdir",
        "/work",
        "--mount",
        f"type=bind,src={config.repo_root},dst=/work",
        "--mount",
        f"type=bind,src={common_git_dir},dst=/git-common,readonly",
    ]
    if gpu:
        command.extend(
            [
                "--gpus",
                "all",
                "--mount",
                f"type=volume,src={config.cache_volume},dst=/cache/jax",
            ]
        )
    for name in sorted(container_environment):
        command.extend(["--env", f"{name}={container_environment[name]}"])
    command.append(image_id)
    return command


def _container_source_snapshot(
    config: LaunchConfig,
    *,
    image: ImageIdentity,
    source: SourceSnapshot,
    git_dir_in_container: str,
    runner: CommandRunner,
) -> tuple[dict[str, Any], tuple[CommandRecord, ...]]:
    explicit = _container_environment(source.head, image.image_id, git_dir_in_container)
    base = _docker_base(
        config,
        image_id=image.image_id,
        common_git_dir=source.common_git_dir,
        container_environment=explicit,
        gpu=False,
    )
    commands = (
        [*base, "git", "--version"],
        [*base, "git", "rev-parse", "HEAD"],
        [*base, "git", "status", "--porcelain=v1", "--untracked-files=all"],
    )
    records = tuple(
        _run(
            runner,
            command,
            cwd=config.repo_root,
            environment=_sanitized_host_environment(),
            recorded_environment=explicit,
        )
        for command in commands
    )
    version, head, status = (
        _require_success(record, "container Git source preflight")
        for record in records
    )
    if not version.startswith("git version "):
        raise ProvenanceError("qualifying image does not report a valid Git version")
    head = head.lower()
    return {
        "git_version": version,
        "head": head,
        "clean": status == "",
        "head_matches_expected": head == source.head,
        "environment": explicit,
    }, records


def _artifact_reference(
    path: Path, *, repo_root: Path | None = None
) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if repo_root is not None:
        relative = path.resolve().relative_to(
            repo_root.resolve()
        ).as_posix()
        reference["host_path"] = str(path.resolve())
        reference["path"] = relative
        reference["repo_relative_path"] = relative
    return reference


def _validate_artifact_reference(
    value: Any,
    expected_path: Path,
    *,
    repo_root: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceError(f"{label} artifact reference is missing")
    relative = expected_path.resolve().relative_to(repo_root.resolve()).as_posix()
    if (
        not expected_path.is_file()
        or value.get("path") != relative
        or value.get("repo_relative_path") != relative
        or value.get("sha256") != sha256_file(expected_path)
        or not _is_integer(value.get("size_bytes"))
        or int(value["size_bytes"]) != expected_path.stat().st_size
    ):
        raise ProvenanceError(f"{label} artifact digest/path mismatch")
    return dict(value)


def _launch_bundle_sha256(
    target: str,
    head: str,
    preflight_sha256: str,
    result_sha256: str,
) -> str:
    return hashlib.sha256(
        (
            f"example21-launch-bundle-v1\0{target}\0{head}\0"
            f"{preflight_sha256}\0{result_sha256}"
        ).encode("utf-8")
    ).hexdigest()


def _load_gate_a_prerequisite(config: LaunchConfig) -> dict[str, Any]:
    paths = _gate_a_artifact_paths(config)
    for path in (paths.manifest, paths.preflight, paths.result):
        if not path.is_file() or not path.resolve().is_relative_to(config.repo_root):
            raise ProvenanceError("authenticated Gate A prerequisite is missing")
    manifest_sha256 = sha256_file(paths.manifest)
    result_sha256 = sha256_file(paths.result)
    if manifest_sha256 != _GATE_A_MANIFEST_SHA256:
        raise ProvenanceError("authenticated Gate A manifest SHA-256 mismatch")
    if result_sha256 != _GATE_A_RESULT_SHA256:
        raise ProvenanceError("authenticated Gate A result SHA-256 mismatch")

    manifest = load_strict_json(paths.manifest)
    if (
        not _is_integer(manifest.get("schema_version"))
        or int(manifest["schema_version"]) != 1
        or manifest.get("kind") != "example21_authenticated_launch_manifest"
        or manifest.get("target") != "formal_gate_a"
        or manifest.get("source_head") != _GATE_A_SOURCE_COMMIT
        or manifest.get("bundle_valid") is not True
        or manifest.get("process_succeeded") is not True
        or manifest.get("artifact_schema_verified") is not True
        or manifest.get("scientific_qualification_passed") is not True
        or manifest.get("failure") is not None
    ):
        raise ProvenanceError("authenticated Gate A manifest is invalid")
    preflight_reference = _validate_artifact_reference(
        manifest.get("preflight"),
        paths.preflight,
        repo_root=config.repo_root,
        label="Gate A preflight",
    )
    result_reference = _validate_artifact_reference(
        manifest.get("result"),
        paths.result,
        repo_root=config.repo_root,
        label="Gate A result",
    )
    expected_bundle = _launch_bundle_sha256(
        "formal_gate_a",
        _GATE_A_SOURCE_COMMIT,
        preflight_reference["sha256"],
        result_reference["sha256"],
    )
    if (
        manifest.get("bundle_sha256") != expected_bundle
        or expected_bundle != _GATE_A_BUNDLE_SHA256
    ):
        raise ProvenanceError("authenticated Gate A bundle digest is invalid")

    result = load_strict_json(paths.result)
    source = result.get("source")
    source_end = result.get("source_end")
    qualification = result.get("qualification")
    if (
        not _is_integer(result.get("schema_version"))
        or int(result["schema_version"]) != 3
        or result.get("control")
        != "example21_associative_workspace_binding_gate_a"
        or result.get("learner") != "pp_prop_only"
        or not isinstance(source, Mapping)
        or not _source_report_matches(source, _GATE_A_SOURCE_COMMIT)
        or not isinstance(source_end, Mapping)
        or not _source_report_matches(source_end, _GATE_A_SOURCE_COMMIT)
        or not isinstance(qualification, Mapping)
        or qualification.get("passed") is not True
        or result.get("interpretation")
        != "gate_a_passed_associative_binding"
    ):
        raise ProvenanceError("authenticated Gate A result is invalid")
    return {
        "qualification_passed": True,
        "result_sha256": result_sha256,
        "manifest_sha256": manifest_sha256,
        "source_commit": _GATE_A_SOURCE_COMMIT,
        "bundle_sha256": expected_bundle,
        "preflight_sha256": preflight_reference["sha256"],
        "result_path": paths.result.relative_to(config.repo_root).as_posix(),
        "manifest_path": paths.manifest.relative_to(config.repo_root).as_posix(),
    }


def _source_report_matches(value: Mapping[str, Any], head: str) -> bool:
    return bool(
        value.get("commit") == head
        and value.get("asserted_commit") == head
        and value.get("asserted_commit_matches_head") is True
        and value.get("commit_is_valid_40_hex") is True
        and value.get("head_command_succeeded") is True
        and value.get("dirty") is False
        and value.get("asserted_dirty") is False
        and value.get("asserted_dirty_matches_worktree") is True
        and value.get("status_command_succeeded") is True
        and value.get("verified") is True
    )


def _validated_command_record(
    value: Mapping[str, Any],
    *,
    argv_tail: Sequence[str],
    environment: Mapping[str, str] | None = None,
    cwd: str | None = None,
) -> tuple[str, str]:
    required = {
        "argv",
        "cwd",
        "environment",
        "returncode",
        "stdout",
        "stderr",
        "stdout_sha256",
        "stderr_sha256",
        "wall_seconds",
    }
    if set(value) != required:
        raise ProvenanceError("retained command record has an unexpected schema")
    argv = value["argv"]
    stdout = value["stdout"]
    stderr = value["stderr"]
    wall_seconds = value["wall_seconds"]
    if (
        not isinstance(argv, list)
        or not all(isinstance(item, str) for item in argv)
        or argv[-len(argv_tail) :] != list(argv_tail)
        or not _is_integer(value["returncode"])
        or int(value["returncode"]) != 0
        or not isinstance(value["cwd"], str)
        or (
            cwd is not None
            and _portable_host_path(value["cwd"]) != _portable_host_path(cwd)
        )
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
        or value["stdout_sha256"] != _sha256_text(stdout)
        or value["stderr_sha256"] != _sha256_text(stderr)
        or not _is_finite_real(wall_seconds)
        or float(wall_seconds) < 0.0
        or not isinstance(value["environment"], Mapping)
        or (environment is not None and dict(value["environment"]) != dict(environment))
    ):
        raise ProvenanceError("retained command record is incomplete or inconsistent")
    return stdout, stderr


def _parse_mount(value: str) -> dict[str, str | bool]:
    fields: dict[str, str | bool] = {}
    for item in value.split(","):
        if "=" in item:
            key, field_value = item.split("=", maxsplit=1)
            fields[key] = field_value
        else:
            fields[item] = True
    return fields


def _portable_host_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/").rstrip("/").casefold()


def _authenticated_host_cwd(source: Mapping[str, Any], repo_root: Path) -> str:
    if os.name != "nt" and repo_root == Path("/work"):
        return str(source["root"])
    return str(repo_root)


def _validate_preflight_semantics(
    preflight: Mapping[str, Any],
    *,
    target: Literal["one_update", "stability_256", "gate_b_init"],
    head: str,
    image_id: str,
    expected_result: Path,
    repo_root: Path,
) -> None:
    source = preflight.get("source")
    image = preflight.get("image")
    mounts = preflight.get("mounts")
    container_source = preflight.get("container_source")
    planned_gate = preflight.get("planned_gate")
    if (
        not _is_integer(preflight.get("schema_version"))
        or int(preflight["schema_version"]) != 1
        or preflight.get("kind") != "example21_authenticated_launch_preflight"
        or preflight.get("target") != target
        or preflight.get("passed") is not True
        or not isinstance(source, Mapping)
        or source.get("head") != head
        or source.get("clean") is not True
        or not isinstance(image, Mapping)
        or image.get("id") != image_id
        or image.get("oci_revision") != head
        or not isinstance(mounts, Mapping)
        or set(mounts) != {"worktree", "common_git"}
        or not isinstance(container_source, Mapping)
        or container_source.get("head") != head
        or container_source.get("clean") is not True
        or container_source.get("head_matches_expected") is not True
        or not isinstance(planned_gate, Mapping)
        or set(planned_gate) != {"argv", "environment"}
    ):
        raise ProvenanceError(f"required {target} preflight evidence is invalid")

    worktree_mount = mounts["worktree"]
    common_mount = mounts["common_git"]
    host_cwd = _authenticated_host_cwd(source, repo_root)
    if (
        not isinstance(worktree_mount, Mapping)
        or worktree_mount.get("target") != "/work"
        or worktree_mount.get("read_only", False) is not False
        or _portable_host_path(worktree_mount.get("source"))
        != _portable_host_path(source.get("root"))
        or _portable_host_path(source.get("root"))
        != _portable_host_path(host_cwd)
        or not isinstance(common_mount, Mapping)
        or common_mount.get("target") != "/git-common"
        or common_mount.get("read_only") is not True
        or _portable_host_path(common_mount.get("source"))
        != _portable_host_path(source.get("common_git_dir"))
    ):
        raise ProvenanceError("preflight does not retain the required mount contract")

    planned_environment = planned_gate["environment"]
    container_environment = container_source.get("environment")
    if not isinstance(planned_environment, Mapping) or not isinstance(
        container_environment, Mapping
    ):
        raise ProvenanceError("preflight omits the exact container environment")
    git_dir = str(planned_environment.get("GIT_DIR", ""))
    git_path = PurePosixPath(git_dir)
    if (
        not git_path.is_absolute()
        or not git_path.is_relative_to(PurePosixPath("/git-common"))
        or ".." in git_path.parts
    ):
        raise ProvenanceError("preflight GIT_DIR is outside the read-only Git mount")
    expected_environment = _container_environment(head, image_id, git_dir)
    if (
        dict(planned_environment) != expected_environment
        or dict(container_environment) != expected_environment
    ):
        raise ProvenanceError("preflight environment differs from the fixed contract")

    host_commands = source.get("commands")
    if not isinstance(host_commands, list) or len(host_commands) != 5:
        raise ProvenanceError("preflight omits host Git command evidence")
    host_tails = (
        ("git", "rev-parse", "--show-toplevel"),
        ("git", "rev-parse", "--path-format=absolute", "--git-common-dir"),
        ("git", "rev-parse", "--absolute-git-dir"),
        ("git", "rev-parse", "HEAD"),
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
    )
    host_outputs = [
        _validated_command_record(
            record,
            argv_tail=tail,
            environment={"git_override_variables_removed": "true"},
            cwd=host_cwd,
        )[0]
        for record, tail in zip(host_commands, host_tails, strict=True)
    ]
    if any(
        record["argv"] != list(tail)
        for record, tail in zip(host_commands, host_tails, strict=True)
    ):
        raise ProvenanceError("host Git command argv differs from preflight")
    if host_outputs[3].strip().lower() != head or host_outputs[4].strip() != "":
        raise ProvenanceError("host Git command evidence disagrees with clean HEAD")
    if any(
        _portable_host_path(host_outputs[index])
        != _portable_host_path(source[field])
        for index, field in enumerate(("root", "common_git_dir", "git_dir"))
    ):
        raise ProvenanceError("host Git paths disagree with retained source paths")

    inspect_record = image.get("inspect_command")
    if not isinstance(inspect_record, Mapping):
        raise ProvenanceError("preflight omits immutable image inspection evidence")
    inspect_stdout, _ = _validated_command_record(
        inspect_record,
        argv_tail=("docker", "image", "inspect", str(image.get("reference", ""))),
        environment={},
        cwd=host_cwd,
    )
    if inspect_record["argv"] != [
        "docker",
        "image",
        "inspect",
        str(image.get("reference", "")),
    ]:
        raise ProvenanceError("retained image inspection argv is not exact")
    try:
        inspection = json.loads(inspect_stdout)
        if not isinstance(inspection, list) or len(inspection) != 1:
            raise TypeError("expected exactly one retained image inspection")
        inspected = inspection[0]
        inspected_id = inspected["Id"]
        inspected_revision = inspected["Config"]["Labels"][
            "org.opencontainers.image.revision"
        ]
    except (IndexError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ProvenanceError("retained image inspection output is invalid") from error
    if inspected_id != image_id or inspected_revision != head:
        raise ProvenanceError("retained image inspection output disagrees with identity")

    container_commands = container_source.get("commands")
    if not isinstance(container_commands, list) or len(container_commands) != 3:
        raise ProvenanceError("preflight omits container Git command evidence")
    container_tails = (
        (image_id, "git", "--version"),
        (image_id, "git", "rev-parse", "HEAD"),
        (image_id, "git", "status", "--porcelain=v1", "--untracked-files=all"),
    )
    container_outputs = [
        _validated_command_record(
            record,
            argv_tail=tail,
            environment=expected_environment,
            cwd=host_cwd,
        )[0]
        for record, tail in zip(container_commands, container_tails, strict=True)
    ]
    container_base = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--workdir",
        "/work",
        "--mount",
        f"type=bind,src={worktree_mount['source']},dst=/work",
        "--mount",
        f"type=bind,src={common_mount['source']},dst=/git-common,readonly",
    ]
    for name in sorted(expected_environment):
        container_base.extend(["--env", f"{name}={expected_environment[name]}"])
    container_base.append(image_id)
    container_commands_tail = (
        ("git", "--version"),
        ("git", "rev-parse", "HEAD"),
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
    )
    if any(
        record["argv"] != [*container_base, *tail]
        for record, tail in zip(
            container_commands, container_commands_tail, strict=True
        )
    ):
        raise ProvenanceError("container Git command argv differs from preflight")
    if (
        not container_outputs[0].strip().startswith("git version ")
        or container_outputs[0].strip() != container_source.get("git_version")
        or container_outputs[1].strip().lower() != head
        or container_outputs[2].strip() != ""
    ):
        raise ProvenanceError("container Git command evidence disagrees with clean HEAD")

    argv = planned_gate["argv"]
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ProvenanceError("planned Gate command is not shell-free argv")
    mount_values = [argv[index + 1] for index, item in enumerate(argv) if item == "--mount"]
    parsed_mounts = [_parse_mount(value) for value in mount_values]
    if (
        len(parsed_mounts) != 3
        or not any(mount.get("dst") == "/work" and mount.get("type") == "bind" for mount in parsed_mounts)
        or not any(
            mount.get("dst") == "/git-common"
            and mount.get("type") == "bind"
            and mount.get("readonly") is True
            for mount in parsed_mounts
        )
        or not any(
            mount.get("dst") == "/cache/jax" and mount.get("type") == "volume"
            for mount in parsed_mounts
        )
    ):
        raise ProvenanceError("planned Gate command mount argv is not fixed")
    work_argv_mount = next(mount for mount in parsed_mounts if mount.get("dst") == "/work")
    git_argv_mount = next(
        mount for mount in parsed_mounts if mount.get("dst") == "/git-common"
    )
    cache_argv_mount = next(
        mount for mount in parsed_mounts if mount.get("dst") == "/cache/jax"
    )
    cache_source = str(cache_argv_mount.get("src", ""))
    if (
        _portable_host_path(work_argv_mount.get("src"))
        != _portable_host_path(worktree_mount.get("source"))
        or _portable_host_path(git_argv_mount.get("src"))
        != _portable_host_path(common_mount.get("source"))
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", cache_source)
    ):
        raise ProvenanceError("planned Gate mount sources disagree with the sidecar")
    container_result = str(
        PurePosixPath("/work") / expected_result.relative_to(repo_root).as_posix()
    )
    if target == "gate_b_init":
        gate_a_stem = f"{_GATE_A_SOURCE_COMMIT}-formal-gate-a"
        gate_a_base = PurePosixPath("/work") / _GATE_A_DIRECTORY.as_posix()
        expected_tail = [
            image_id,
            "python",
            "-m",
            _DEPTH_GATE_MODULE,
            "--target",
            target,
            "--gate-a-result",
            str(gate_a_base / f"{gate_a_stem}.json"),
            "--gate-a-manifest",
            str(gate_a_base / f"{gate_a_stem}.manifest.json"),
            "--output",
            container_result,
        ]
    else:
        expected_tail = [
            image_id,
            "python",
            "-m",
            _GATE_MODULE,
            "--target",
            target,
            "--output",
            container_result,
        ]
    expected_argv = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--workdir",
        "/work",
        "--mount",
        f"type=bind,src={worktree_mount['source']},dst=/work",
        "--mount",
        f"type=bind,src={common_mount['source']},dst=/git-common,readonly",
        "--gpus",
        "all",
        "--mount",
        f"type=volume,src={cache_source},dst=/cache/jax",
    ]
    for name in sorted(expected_environment):
        expected_argv.extend(["--env", f"{name}={expected_environment[name]}"])
    expected_argv.extend(expected_tail)
    if argv != expected_argv:
        raise ProvenanceError("planned Gate argv differs from its fixed target")


def _validate_manifest_execution(
    manifest: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    head: str,
    repo_root: Path,
) -> None:
    planned = preflight["planned_gate"]
    expected_environment = planned["environment"]
    gate_record = manifest.get("gate_command")
    if not isinstance(gate_record, Mapping):
        raise ProvenanceError("admission manifest omits the executed Gate command")
    host_cwd = _authenticated_host_cwd(preflight["source"], repo_root)
    _validated_command_record(
        gate_record,
        argv_tail=planned["argv"],
        environment=expected_environment,
        cwd=host_cwd,
    )
    if gate_record["argv"] != planned["argv"]:
        raise ProvenanceError("executed Gate command differs from preflight")

    postflight = manifest.get("postflight")
    if (
        manifest.get("failure") is not None
        or not isinstance(postflight, Mapping)
        or postflight.get("head") != head
        or postflight.get("clean") is not True
    ):
        raise ProvenanceError("admission postflight did not retain clean source")
    host_commands = postflight.get("host_commands")
    if not isinstance(host_commands, list) or len(host_commands) != 5:
        raise ProvenanceError("admission postflight omits host Git commands")
    host_tails = (
        ("git", "rev-parse", "--show-toplevel"),
        ("git", "rev-parse", "--path-format=absolute", "--git-common-dir"),
        ("git", "rev-parse", "--absolute-git-dir"),
        ("git", "rev-parse", "HEAD"),
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
    )
    host_outputs = [
        _validated_command_record(
            record,
            argv_tail=tail,
            environment={"git_override_variables_removed": "true"},
            cwd=host_cwd,
        )[0]
        for record, tail in zip(host_commands, host_tails, strict=True)
    ]
    if any(
        record["argv"] != list(tail)
        for record, tail in zip(host_commands, host_tails, strict=True)
    ):
        raise ProvenanceError("admission host postflight argv is not exact")
    source = preflight["source"]
    if (
        host_outputs[3].strip().lower() != head
        or host_outputs[4].strip() != ""
        or any(
            _portable_host_path(host_outputs[index])
            != _portable_host_path(source[field])
            for index, field in enumerate(("root", "common_git_dir", "git_dir"))
        )
    ):
        raise ProvenanceError("admission host postflight disagrees with clean HEAD")

    container = postflight.get("container")
    container_commands = postflight.get("container_commands")
    if (
        not isinstance(container, Mapping)
        or container.get("head") != head
        or container.get("clean") is not True
        or container.get("head_matches_expected") is not True
        or dict(container.get("environment", {})) != dict(expected_environment)
        or not isinstance(container_commands, list)
        or len(container_commands) != 3
    ):
        raise ProvenanceError("admission container postflight is incomplete")
    image_id = expected_environment["BRAINTRACE_IMAGE_DIGEST"]
    container_tails = (
        (image_id, "git", "--version"),
        (image_id, "git", "rev-parse", "HEAD"),
        (image_id, "git", "status", "--porcelain=v1", "--untracked-files=all"),
    )
    container_outputs = [
        _validated_command_record(
            record,
            argv_tail=tail,
            environment=expected_environment,
            cwd=host_cwd,
        )[0]
        for record, tail in zip(container_commands, container_tails, strict=True)
    ]
    mounts = preflight["mounts"]
    container_base = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--workdir",
        "/work",
        "--mount",
        f"type=bind,src={mounts['worktree']['source']},dst=/work",
        "--mount",
        f"type=bind,src={mounts['common_git']['source']},dst=/git-common,readonly",
    ]
    for name in sorted(expected_environment):
        container_base.extend(["--env", f"{name}={expected_environment[name]}"])
    container_base.append(image_id)
    command_tails = (
        ("git", "--version"),
        ("git", "rev-parse", "HEAD"),
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
    )
    if any(
        record["argv"] != [*container_base, *tail]
        for record, tail in zip(container_commands, command_tails, strict=True)
    ):
        raise ProvenanceError("admission container postflight argv is not exact")
    if (
        not container_outputs[0].strip().startswith("git version ")
        or container_outputs[0].strip() != container.get("git_version")
        or container_outputs[1].strip().lower() != head
        or container_outputs[2].strip() != ""
    ):
        raise ProvenanceError("admission container postflight disagrees with clean HEAD")


def _validate_gate_b_scientific_result(
    result: Mapping[str, Any],
    *,
    target: Literal["gate_b_init", "formal_gate_b"],
    head: str,
    image_id: str,
) -> bool:
    from examples.pp_prop import latent_workspace_depth_gate as depth

    config = depth.DepthGateConfig()
    controls = {
        "gate_b_init": depth.GATE_B_INITIALIZATION_CONTROL,
        "formal_gate_b": depth.GATE_B_CONTROL,
    }
    source_start = result.get("source_start")
    source_end = result.get("source_end")
    environment = result.get("environment")
    devices = environment.get("devices") if isinstance(environment, Mapping) else None
    source_files = result.get("source_files")
    prerequisites = result.get("prerequisites")
    gate_a = prerequisites.get("gate_a") if isinstance(prerequisites, Mapping) else None
    if (
        not _is_integer(result.get("schema_version"))
        or int(result["schema_version"]) != depth.GATE_B_SCHEMA_VERSION
        or result.get("control") != controls[target]
        or result.get("qualification_regime") != "preregistered_full"
        or not _json_exact(result.get("config"), asdict(config))
        or not isinstance(source_start, Mapping)
        or not _source_report_matches(source_start, head)
        or not isinstance(source_end, Mapping)
        or not _source_report_matches(source_end, head)
        or not isinstance(environment, Mapping)
        or environment.get("image_digest") != image_id
        or environment.get("backend") != "gpu"
        or not isinstance(devices, list)
        or not devices
        or not all(isinstance(device, Mapping) for device in devices)
        or not any(device.get("platform") == "gpu" for device in devices)
        or not _json_exact(
            source_files,
            {
                "latent_workspace_model.py": config.model_source_sha256,
                "latent_workspace_task.py": config.task_source_sha256,
            },
        )
        or not isinstance(gate_a, Mapping)
        or gate_a.get("qualification_passed") is not True
        or gate_a.get("result_sha256") != _GATE_A_RESULT_SHA256
        or gate_a.get("manifest_sha256") != _GATE_A_MANIFEST_SHA256
        or gate_a.get("source_commit") != _GATE_A_SOURCE_COMMIT
    ):
        raise ProvenanceError(f"{target} result provenance/configuration is invalid")

    qualification = result.get("qualification")
    if not isinstance(qualification, Mapping):
        raise ProvenanceError(f"{target} result qualification is missing")
    qualifier = (
        depth._gate_b_initialization_qualification
        if target == "gate_b_init"
        else depth._qualification_report
    )
    recomputed = qualifier(result, config=config)
    if not _json_exact(qualification, recomputed):
        raise ProvenanceError(f"{target} scientific qualification does not recompute")
    return bool(recomputed["passed"])


def _validate_fixed_config(result: Mapping[str, Any], target: LaunchTarget) -> None:
    config = result.get("config")
    if not isinstance(config, Mapping):
        raise ProvenanceError("result config is missing")
    from examples.pp_prop import latent_workspace_binding_gate as gate

    fixed = (
        gate.BindingGateConfig.stage21_stability_config()
        if target == "stability_256"
        else gate.BindingGateConfig.stage21_one_update_config()
    )
    if target == "formal_gate_a":
        model = gate._model_config(fixed, batch_size=fixed.batch_size)
        expected = {
            **asdict(fixed),
            "configuration_scale": fixed.configuration_scale,
            "qualification_regime": fixed.qualification_regime,
            "configuration_sha256": gate._configuration_digest(fixed, model),
            "model": asdict(model),
        }
        expected = gate.legacy._json_ready(expected)
    else:
        expected = {
            **asdict(fixed),
            "configuration_scale": fixed.configuration_scale,
        }
        if target == "stability_256":
            expected["qualification_regime"] = fixed.qualification_regime
    if not _json_exact(config, expected):
        raise ProvenanceError(f"{target} artifact does not use its fixed configuration")


def _validate_target_result(
    result: Mapping[str, Any],
    *,
    target: LaunchTarget,
    head: str,
    image_id: str,
    admission_manifests: Mapping[str, Path] | None,
    repo_root: Path,
    gate_a_prerequisite: Mapping[str, Any] | None = None,
    gate_b_init_bundle: Mapping[str, Any] | None = None,
) -> bool:
    if target in _GATE_B_TARGETS:
        if admission_manifests is not None:
            raise ProvenanceError("Gate B targets do not accept Gate A admissions")
        prerequisites = result.get("prerequisites")
        embedded_gate_a = (
            prerequisites.get("gate_a")
            if isinstance(prerequisites, Mapping)
            else None
        )
        if gate_a_prerequisite is None or not _json_exact(
            embedded_gate_a, gate_a_prerequisite
        ):
            raise ProvenanceError("Gate B result does not bind Gate A prerequisite")
        if target == "formal_gate_b":
            embedded_init = prerequisites.get("gate_b_initialization")
            if gate_b_init_bundle is None or not _json_exact(
                embedded_init, gate_b_init_bundle
            ):
                raise ProvenanceError(
                    "formal Gate B result does not bind initialization manifest"
                )
        elif gate_b_init_bundle is not None:
            raise ProvenanceError("Gate B initialization received a circular prerequisite")
        return _validate_gate_b_scientific_result(
            result,
            target=target,
            head=head,
            image_id=image_id,
        )

    controls = {
        "one_update": "example21_stage21_one_update_admission",
        "stability_256": "example21_stage21_stability_256_admission",
        "formal_gate_a": "example21_associative_workspace_binding_gate_a",
    }
    if (
        not _is_integer(result.get("schema_version"))
        or int(result["schema_version"]) != 3
        or result.get("control") != controls[target]
    ):
        raise ProvenanceError("result schema/control does not match launch target")
    if target != "formal_gate_a" and result.get("target") != target:
        raise ProvenanceError("admission result target mismatch")
    _validate_fixed_config(result, target)
    source = result.get("source")
    source_end = result.get("source_end")
    if not isinstance(source, Mapping) or not _source_report_matches(source, head):
        raise ProvenanceError("result start source evidence is not authenticated")
    if not isinstance(source_end, Mapping) or not _source_report_matches(source_end, head):
        raise ProvenanceError("result end source evidence is not authenticated")
    environment = result.get("environment")
    devices = environment.get("devices") if isinstance(environment, Mapping) else None
    if (
        not isinstance(environment, Mapping)
        or environment.get("image_digest") != image_id
        or environment.get("backend") != "gpu"
        or not isinstance(devices, list)
        or not devices
        or not all(isinstance(device, Mapping) for device in devices)
        or not any(device.get("platform") == "gpu" for device in devices)
    ):
        raise ProvenanceError("result image/backend evidence is not authenticated")
    qualification = result.get("qualification")
    if not isinstance(qualification, Mapping) or not isinstance(
        qualification.get("passed"), bool
    ):
        raise ProvenanceError("result qualification is incomplete")
    if target != "formal_gate_a":
        admission = result.get("admission")
        if not isinstance(admission, Mapping):
            raise ProvenanceError("admission result omits its inner evidence")
        if not _json_exact(admission.get("config"), result.get("config")):
            raise ProvenanceError("admission envelope config differs from inner evidence")
        from examples.pp_prop import latent_workspace_binding_gate as gate

        qualifier = (
            gate._one_update_admission_qualification
            if target == "one_update"
            else gate._stability_admission_qualification
        )
        recomputed = qualifier(admission)
        if (
            result.get("learner") != "pp_prop_only"
            or not _json_exact(admission.get("qualification"), recomputed)
            or not _json_exact(qualification, recomputed)
            or result.get("interpretation") != recomputed["interpretation"]
        ):
            raise ProvenanceError("admission qualification does not recompute exactly")
        return bool(recomputed["passed"])
    if target == "formal_gate_a":
        from examples.pp_prop import latent_workspace_binding_gate as gate

        initialization = result.get("initialization")
        data = result.get("data")
        if (
            not isinstance(initialization, Mapping)
            or not _json_exact(
                initialization,
                {
                "fresh_model": True,
                "model_seed": 2108,
                "parameter_sha256": gate.PREREGISTERED_GPU_INITIAL_PARAMETER_SHA256,
                "parameter_count": gate.PREREGISTERED_PARAMETER_COUNT,
                },
            )
            or not isinstance(data, Mapping)
            or data.get("training_schedule_sha256")
            != gate.PREREGISTERED_TRAINING_SCHEDULE_SHA256
            or data.get("validation_schedule_sha256")
            != gate.PREREGISTERED_STABILITY_DIGESTS["validation_schedule_sha256"]
        ):
            raise ProvenanceError(
                "formal result initialization/schedule differs from admissions"
            )
        admissions = result.get("stage21_admissions")
        if (
            not isinstance(admissions, Mapping)
            or set(admissions) != {"one_update", "stability_256"}
            or admission_manifests is None
            or set(admission_manifests) != {"one_update", "stability_256"}
        ):
            raise ProvenanceError("formal result omits authenticated admissions")
        for name, path in admission_manifests.items():
            evidence = admissions.get(name)
            bundle = load_authenticated_admission(
                path,
                target=name,
                head=head,
                image_id=image_id,
                repo_root=repo_root,
            )
            expected = {
                "target": name,
                "source_head": head,
                "image_digest": image_id,
                "bundle_sha256": bundle["manifest"]["bundle_sha256"],
                "manifest_sha256": bundle["manifest_sha256"],
                "preflight_sha256": bundle["preflight_sha256"],
                "result_sha256": bundle["result_sha256"],
                "admission": bundle["admission"],
            }
            if not isinstance(evidence, Mapping) or not _json_exact(evidence, expected):
                raise ProvenanceError(f"formal result does not bind {name} manifest")
        try:
            training = result["training"]
            recomputed = gate._qualification_report(
                evaluation=result["evaluation"],
                diagnostics=result["diagnostics"],
                compiler=training["compiler"],
                training=training,
                environment=environment,
                architecture=result["architecture"],
                gate_native_separation=result["gate_native_key_separation"],
                standard_arc_separation=result["standard_arc_key_separation"],
                marginals=data["marginals"],
                source_start=source,
                source_end=source_end,
                one_update_admission=admissions["one_update"]["admission"],
                stability_admission=admissions["stability_256"]["admission"],
                initialization=initialization,
                data=data,
                config=gate.BindingGateConfig(),
            )
        except (KeyError, TypeError) as error:
            raise ProvenanceError("formal scientific evidence is incomplete") from error
        if (
            result.get("learner") != "pp_prop_only"
            or not _json_exact(qualification, recomputed)
            or result.get("interpretation") != recomputed["interpretation"]
        ):
            raise ProvenanceError("formal scientific qualification does not recompute")
        return bool(recomputed["passed"])
    raise AssertionError("unreachable launch target")


def load_authenticated_admission(
    manifest_path: str | Path,
    *,
    target: Literal["one_update", "stability_256"],
    head: str,
    image_id: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate and load one immutable-image admission bundle.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        Fixed companion manifest created by this launcher.
    target : {"one_update", "stability_256"}
        Required admission target.
    head : str
        Exact full source revision.
    image_id : str
        Exact inspected immutable image ID.
    repo_root : str or pathlib.Path
        Worktree root containing every referenced artifact.

    Returns
    -------
    dict
        Validated manifest, preflight, result, and admission report.
    """

    root = Path(repo_root).resolve()
    path = Path(manifest_path).resolve()
    if not path.is_relative_to(root) or path.parent == root:
        raise ProvenanceError(f"required {target} admission manifest path is invalid")
    expected_stem = f"{head}-{target.replace('_', '-')}"
    expected_manifest = path.parent / f"{expected_stem}.manifest.json"
    expected_preflight = path.parent / f"{expected_stem}.preflight.json"
    expected_result = path.parent / f"{expected_stem}.json"
    if path != expected_manifest:
        raise ProvenanceError(f"required {target} admission manifest path is not fixed")
    for artifact in (expected_manifest, expected_preflight, expected_result):
        if not artifact.is_file() or not artifact.resolve().is_relative_to(root):
            raise ProvenanceError(f"required {target} admission bundle is incomplete")

    value = load_strict_json(expected_manifest)
    if (
        not _is_integer(value.get("schema_version"))
        or int(value["schema_version"]) != 1
        or value.get("kind") != "example21_authenticated_launch_manifest"
        or value.get("target") != target
        or value.get("bundle_valid") is not True
        or value.get("process_succeeded") is not True
        or value.get("artifact_schema_verified") is not True
        or value.get("scientific_qualification_passed") is not True
        or value.get("source_head") != head
    ):
        raise ProvenanceError(f"required {target} admission manifest is invalid")
    references = {
        "preflight": (value.get("preflight"), expected_preflight),
        "result": (value.get("result"), expected_result),
    }
    for label, (reference, expected_path) in references.items():
        if not isinstance(reference, Mapping):
            raise ProvenanceError(f"required {target} {label} reference is missing")
        if (
            reference.get("path") != expected_path.relative_to(root).as_posix()
            or reference.get("repo_relative_path")
            != expected_path.relative_to(root).as_posix()
            or reference.get("sha256") != sha256_file(expected_path)
            or not _is_integer(reference.get("size_bytes"))
            or int(reference["size_bytes"]) != expected_path.stat().st_size
        ):
            raise ProvenanceError(f"required {target} {label} digest/path mismatch")

    expected_bundle_sha256 = hashlib.sha256(
        (
            f"example21-launch-bundle-v1\0{target}\0{head}\0"
            f"{references['preflight'][0]['sha256']}\0"
            f"{references['result'][0]['sha256']}"
        ).encode("utf-8")
    ).hexdigest()
    if value.get("bundle_sha256") != expected_bundle_sha256:
        raise ProvenanceError(f"required {target} bundle digest is invalid")

    preflight = load_strict_json(expected_preflight)
    _validate_preflight_semantics(
        preflight,
        target=target,
        head=head,
        image_id=image_id,
        expected_result=expected_result,
        repo_root=root,
    )
    _validate_manifest_execution(value, preflight, head=head, repo_root=root)

    result = load_strict_json(expected_result)
    scientific = _validate_target_result(
        result,
        target=target,
        head=head,
        image_id=image_id,
        admission_manifests=None,
        repo_root=root,
    )
    if not scientific:
        raise ProvenanceError(f"required {target} admission did not pass")
    admission = result.get("admission")
    if not isinstance(admission, Mapping):
        raise ProvenanceError(f"required {target} admission report is missing")
    return {
        "manifest": value,
        "preflight": preflight,
        "result": result,
        "admission": dict(admission),
        "manifest_sha256": sha256_file(expected_manifest),
        "preflight_sha256": sha256_file(expected_preflight),
        "result_sha256": sha256_file(expected_result),
    }


def _load_gate_b_init_manifest(
    config: LaunchConfig,
    *,
    head: str,
    image_id: str,
) -> dict[str, Any]:
    paths = target_paths(config, head, "gate_b_init")
    for path in (paths.manifest, paths.preflight, paths.result):
        if not path.is_file() or not path.resolve().is_relative_to(config.repo_root):
            raise ProvenanceError("required Gate B initialization manifest is missing")
    manifest = load_strict_json(paths.manifest)
    if (
        not _is_integer(manifest.get("schema_version"))
        or int(manifest["schema_version"]) != 1
        or manifest.get("kind") != "example21_authenticated_launch_manifest"
        or manifest.get("target") != "gate_b_init"
        or manifest.get("source_head") != head
        or manifest.get("bundle_valid") is not True
        or manifest.get("process_succeeded") is not True
        or manifest.get("artifact_schema_verified") is not True
        or manifest.get("scientific_qualification_passed") is not True
        or manifest.get("failure") is not None
    ):
        raise ProvenanceError("required Gate B initialization manifest is invalid")
    preflight_reference = _validate_artifact_reference(
        manifest.get("preflight"),
        paths.preflight,
        repo_root=config.repo_root,
        label="Gate B initialization preflight",
    )
    result_reference = _validate_artifact_reference(
        manifest.get("result"),
        paths.result,
        repo_root=config.repo_root,
        label="Gate B initialization result",
    )
    bundle_sha256 = _launch_bundle_sha256(
        "gate_b_init",
        head,
        preflight_reference["sha256"],
        result_reference["sha256"],
    )
    if manifest.get("bundle_sha256") != bundle_sha256:
        raise ProvenanceError("required Gate B initialization bundle is invalid")

    gate_a = _load_gate_a_prerequisite(config)
    preflight = load_strict_json(paths.preflight)
    prerequisites = preflight.get("gate_b_prerequisites")
    if (
        not isinstance(prerequisites, Mapping)
        or not _json_exact(prerequisites.get("gate_a"), gate_a)
        or prerequisites.get("gate_b_initialization") is not None
    ):
        raise ProvenanceError(
            "Gate B initialization preflight prerequisite binding is invalid"
        )
    _validate_preflight_semantics(
        preflight,
        target="gate_b_init",
        head=head,
        image_id=image_id,
        expected_result=paths.result,
        repo_root=config.repo_root,
    )
    _validate_manifest_execution(
        manifest,
        preflight,
        head=head,
        repo_root=config.repo_root,
    )
    result = load_strict_json(paths.result)
    passed = _validate_target_result(
        result,
        target="gate_b_init",
        head=head,
        image_id=image_id,
        admission_manifests=None,
        repo_root=config.repo_root,
        gate_a_prerequisite=gate_a,
        gate_b_init_bundle=None,
    )
    if not passed:
        raise ProvenanceError("required Gate B initialization admission did not pass")
    return {
        "target": "gate_b_init",
        "source_head": head,
        "image_digest": image_id,
        "bundle_sha256": bundle_sha256,
        "manifest_sha256": sha256_file(paths.manifest),
        "preflight_sha256": preflight_reference["sha256"],
        "result_sha256": result_reference["sha256"],
        "admission": result,
    }


def _load_admission_manifests(
    config: LaunchConfig, head: str, image_id: str
) -> dict[str, Path]:
    manifests = {
        target: target_paths(config, head, target).manifest
        for target in ("one_update", "stability_256")
    }
    for target, path in manifests.items():
        if not path.is_file():
            raise ProvenanceError(f"required {target} admission manifest is missing")
        load_authenticated_admission(
            path,
            target=target,
            head=head,
            image_id=image_id,
            repo_root=config.repo_root,
        )
    return manifests


def gate_command(
    config: LaunchConfig,
    *,
    image_id: str,
    head: str,
    paths: TargetPaths,
    git_dir_in_container: str,
    admission_manifests: Mapping[str, Path] | None,
    common_git_dir: Path | None = None,
) -> list[str]:
    """Build the fixed immutable-image command for one target.

    Parameters
    ----------
    config : LaunchConfig
        Fixed target configuration.
    image_id, head : str
        Verified immutable image and full source revision.
    paths : TargetPaths
        Deterministic result paths.
    git_dir_in_container : str
        Worktree Git administrative directory below ``/git-common``.
    admission_manifests : mapping, optional
        Authenticated admission manifests required by formal Gate A.
    common_git_dir : pathlib.Path, optional
        Host common Git directory. Defaults to ``repo_root/.git`` only for the
        pure command-construction test seam; real launches always pass the
        discovered directory.

    Returns
    -------
    list of str
        Shell-free Docker argv.
    """

    explicit = _container_environment(head, image_id, git_dir_in_container)
    command = _docker_base(
        config,
        image_id=image_id,
        common_git_dir=(config.repo_root / ".git")
        if common_git_dir is None
        else common_git_dir,
        container_environment=explicit,
        gpu=True,
    )
    module = _DEPTH_GATE_MODULE if config.target in _GATE_B_TARGETS else _GATE_MODULE
    command.extend(["python", "-m", module])
    if config.target in _GATE_B_TARGETS:
        if admission_manifests is not None:
            raise ProvenanceError("Gate B targets do not accept Gate A admissions")
        gate_a = _gate_a_artifact_paths(config)
        command.extend(
            [
                "--target",
                config.target,
                "--gate-a-result",
                str(gate_a.container_result),
                "--gate-a-manifest",
                str(_container_path(config, gate_a.manifest)),
            ]
        )
        if config.target == "formal_gate_b":
            init_manifest = target_paths(
                config, head, "gate_b_init"
            ).manifest
            command.extend(
                [
                    "--gate-b-init-manifest",
                    str(_container_path(config, init_manifest)),
                ]
            )
    elif config.target == "formal_gate_a":
        if set(admission_manifests or {}) != {"one_update", "stability_256"}:
            raise ProvenanceError("formal Gate A requires both admission manifests")
        for target, flag in (
            ("one_update", "--one-update-manifest"),
            ("stability_256", "--stability-manifest"),
        ):
            host_path = admission_manifests[target]
            relative = host_path.resolve().relative_to(config.repo_root).as_posix()
            command.extend([flag, str(PurePosixPath("/work") / relative)])
    else:
        command.extend(["--target", config.target])
    command.extend(["--output", str(paths.container_result)])
    return command


def _record_json(record: CommandRecord) -> dict[str, Any]:
    return {**asdict(record), "argv": list(record.argv)}


def launch(
    config: LaunchConfig, *, command_runner: CommandRunner = subprocess.run
) -> Path:
    """Run one authenticated admission or formal Gate target.

    Parameters
    ----------
    config : LaunchConfig
        Fixed target and artifact directory.
    command_runner : callable, optional
        Subprocess seam used by co-located tests.

    Returns
    -------
    pathlib.Path
        Companion manifest path.

    Raises
    ------
    ProvenanceError
        If source, image, command, artifact, or postflight evidence fails.
    """

    source_start = _host_source_snapshot(config, command_runner)
    paths = target_paths(config, source_start.head)
    if not source_start.clean:
        raise ProvenanceError("source is dirty before launch")
    for path in (paths.result, paths.preflight, paths.manifest):
        if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
            raise ProvenanceError(f"artifact already exists: {path}")
    ignore = _run(
        command_runner,
        ["git", "check-ignore", "--quiet", str(paths.result)],
        cwd=config.repo_root,
        environment=_sanitized_host_environment(),
    )
    if ignore.returncode != 0:
        raise ProvenanceError("output path is not ignored by Git")

    preflight: dict[str, Any] = {
        "schema_version": 1,
        "kind": "example21_authenticated_launch_preflight",
        "target": config.target,
        "passed": False,
        "source": {
            "root": str(source_start.root),
            "common_git_dir": str(source_start.common_git_dir),
            "git_dir": str(source_start.git_dir),
            "head": source_start.head,
            "clean": source_start.clean,
            "commands": [_record_json(record) for record in source_start.records],
        },
    }
    gate_a_prerequisite: dict[str, Any] | None = None
    gate_b_init_bundle: dict[str, Any] | None = None
    try:
        image = _inspect_image(config, source_start.head, command_runner)
        relative_git_dir = source_start.git_dir.relative_to(source_start.common_git_dir)
        git_dir_in_container = str(
            PurePosixPath("/git-common") / relative_git_dir.as_posix()
        )
        container_start, container_records = _container_source_snapshot(
            config,
            image=image,
            source=source_start,
            git_dir_in_container=git_dir_in_container,
            runner=command_runner,
        )
        if not container_start["head_matches_expected"] or not container_start["clean"]:
            raise ProvenanceError("container source preflight disagrees with clean HEAD")
        admissions = (
            _load_admission_manifests(config, source_start.head, image.image_id)
            if config.target == "formal_gate_a"
            else None
        )
        if config.target in _GATE_B_TARGETS:
            gate_a_prerequisite = _load_gate_a_prerequisite(config)
        if config.target == "formal_gate_b":
            gate_b_init_bundle = _load_gate_b_init_manifest(
                config,
                head=source_start.head,
                image_id=image.image_id,
            )
        command = gate_command(
            config,
            image_id=image.image_id,
            head=source_start.head,
            paths=paths,
            git_dir_in_container=git_dir_in_container,
            admission_manifests=admissions,
            common_git_dir=source_start.common_git_dir,
        )
        preflight.update(
            passed=True,
            image={
                "reference": image.reference,
                "id": image.image_id,
                "oci_revision": image.revision,
                "inspect_command": _record_json(image.record),
            },
            mounts={
                "worktree": {"source": str(config.repo_root), "target": "/work"},
                "common_git": {
                    "source": str(source_start.common_git_dir),
                    "target": "/git-common",
                    "read_only": True,
                },
            },
            container_source={
                **container_start,
                "commands": [_record_json(record) for record in container_records],
            },
            planned_gate={
                "argv": command,
                "environment": _container_environment(
                    source_start.head, image.image_id, git_dir_in_container
                ),
            },
            admission_manifests=(
                {
                    name: _artifact_reference(path, repo_root=config.repo_root)
                    for name, path in admissions.items()
                }
                if admissions is not None
                else None
            ),
            gate_b_prerequisites=(
                {
                    "gate_a": gate_a_prerequisite,
                    "gate_b_initialization": gate_b_init_bundle,
                }
                if config.target in _GATE_B_TARGETS
                else None
            ),
        )
    except BaseException as error:
        preflight["error"] = f"{type(error).__name__}: {error}"
        write_strict_json(paths.preflight, preflight)
        raise
    write_strict_json(paths.preflight, preflight)
    preflight_reference = _artifact_reference(
        paths.preflight, repo_root=config.repo_root
    )

    gate_record: CommandRecord | None = None
    source_end: SourceSnapshot | None = None
    container_end: dict[str, Any] | None = None
    container_end_records: tuple[CommandRecord, ...] = ()
    result_reference: dict[str, Any] | None = None
    artifact_valid = False
    scientific_passed = False
    failure: BaseException | None = None
    try:
        gate_record = _run(
            command_runner,
            command,
            cwd=config.repo_root,
            environment=_sanitized_host_environment(),
            recorded_environment=preflight["planned_gate"]["environment"],
        )
        _require_success(gate_record, f"{config.target} Gate command")
        if not paths.result.is_file():
            raise ProvenanceError("Gate command did not create its declared result")
        result = load_strict_json(paths.result)
        scientific_passed = _validate_target_result(
            result,
            target=config.target,
            head=source_start.head,
            image_id=image.image_id,
            admission_manifests=admissions,
            repo_root=config.repo_root,
            gate_a_prerequisite=gate_a_prerequisite,
            gate_b_init_bundle=gate_b_init_bundle,
        )
        artifact_valid = True
        result_reference = _artifact_reference(paths.result, repo_root=config.repo_root)
    except BaseException as error:
        failure = error
    finally:
        try:
            source_end = _host_source_snapshot(config, command_runner)
            container_end, container_end_records = _container_source_snapshot(
                config,
                image=image,
                source=source_start,
                git_dir_in_container=git_dir_in_container,
                runner=command_runner,
            )
            if (
                source_end.head != source_start.head
                or not source_end.clean
                or container_end["head"] != source_start.head
                or not container_end["clean"]
            ):
                raise ProvenanceError("source changed or became dirty during launch")
        except BaseException as error:
            if failure is None:
                failure = error

    if failure is None:
        try:
            if (
                sha256_file(paths.preflight) != preflight_reference["sha256"]
                or paths.preflight.stat().st_size
                != preflight_reference["size_bytes"]
                or not _json_exact(load_strict_json(paths.preflight), preflight)
                or result_reference is None
                or sha256_file(paths.result) != result_reference["sha256"]
                or paths.result.stat().st_size != result_reference["size_bytes"]
            ):
                raise ProvenanceError(
                    "preflight or result changed before manifest signing"
                )
            if config.target in _GATE_B_TARGETS:
                reloaded_gate_a = _load_gate_a_prerequisite(config)
                if not _json_exact(reloaded_gate_a, gate_a_prerequisite):
                    raise ProvenanceError(
                        "Gate A prerequisite changed before signing"
                    )
            if config.target == "formal_gate_b":
                reloaded_init = _load_gate_b_init_manifest(
                    config,
                    head=source_start.head,
                    image_id=image.image_id,
                )
                if not _json_exact(reloaded_init, gate_b_init_bundle):
                    raise ProvenanceError(
                        "Gate B initialization prerequisite changed before signing"
                    )
            retained_result = load_strict_json(paths.result)
            rechecked_scientific = _validate_target_result(
                retained_result,
                target=config.target,
                head=source_start.head,
                image_id=image.image_id,
                admission_manifests=admissions,
                repo_root=config.repo_root,
                gate_a_prerequisite=gate_a_prerequisite,
                gate_b_init_bundle=gate_b_init_bundle,
            )
            if rechecked_scientific is not scientific_passed:
                raise ProvenanceError(
                    "scientific qualification changed before manifest signing"
                )
        except BaseException as error:
            failure = error

    process_succeeded = gate_record is not None and gate_record.returncode == 0
    bundle_valid = failure is None and source_end is not None and artifact_valid
    manifest = {
        "schema_version": 1,
        "kind": "example21_authenticated_launch_manifest",
        "target": config.target,
        "source_head": source_start.head,
        "bundle_valid": bundle_valid,
        "process_succeeded": process_succeeded,
        "artifact_schema_verified": artifact_valid,
        "scientific_qualification_passed": scientific_passed,
        "preflight": preflight_reference,
        "result": result_reference,
        "gate_command": _record_json(gate_record) if gate_record is not None else None,
        "postflight": {
            "head": source_end.head if source_end is not None else "unavailable",
            "clean": source_end.clean if source_end is not None else False,
            "host_commands": (
                [_record_json(record) for record in source_end.records]
                if source_end is not None
                else []
            ),
            "container": container_end,
            "container_commands": [
                _record_json(record) for record in container_end_records
            ],
        },
        "failure": (
            f"{type(failure).__name__}: {failure}" if failure is not None else None
        ),
    }
    manifest["bundle_sha256"] = _launch_bundle_sha256(
        config.target,
        source_start.head,
        preflight_reference["sha256"],
        result_reference["sha256"] if result_reference else "missing",
    )
    write_strict_json(paths.manifest, manifest)
    if failure is not None:
        if isinstance(failure, ProvenanceError):
            raise failure
        raise ProvenanceError(str(failure)) from failure
    return paths.manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=_TARGETS)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--image", default=_DEFAULT_IMAGE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Launch one fixed target and print its authenticated manifest path.

    Parameters
    ----------
    argv : sequence of str, optional
        Command-line arguments excluding the executable name.

    Returns
    -------
    int
        Zero when provenance and artifact validation succeed; two otherwise.
    """

    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    output = args.output_dir
    if output is None:
        output = Path(
            "var/example21-depth-gate"
            if args.target in _GATE_B_TARGETS
            else "var/example21-binding-gate"
        )
    if not output.is_absolute():
        output = root / output
    try:
        manifest = launch(
            LaunchConfig(
                target=args.target,
                repo_root=root,
                output_dir=output,
                image=args.image,
            )
        )
    except ProvenanceError as error:
        print(str(error), file=os.sys.stderr)
        return 2
    print(manifest)
    value = load_strict_json(manifest)
    if (
        args.target
        in {"one_update", "stability_256", "gate_b_init", "formal_gate_b"}
        and value["scientific_qualification_passed"] is not True
    ):
        print(
            f"{args.target} bundle is valid but scientific admission failed",
            file=os.sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
