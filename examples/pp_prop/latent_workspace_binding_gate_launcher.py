"""Authenticated host launcher for Example 21 capability-gate runs."""

from __future__ import annotations

import argparse
import hashlib
import msgspec_json
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
    "gate_c_init",
    "formal_gate_c",
    "gate_c2_controls",
    "gate_c3_controls",
]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
_TARGETS: tuple[LaunchTarget, ...] = (
    "one_update",
    "stability_256",
    "formal_gate_a",
    "gate_b_init",
    "formal_gate_b",
    "gate_c_init",
    "formal_gate_c",
    "gate_c2_controls",
    "gate_c3_controls",
)
_GATE_B_TARGETS = frozenset({"gate_b_init", "formal_gate_b"})
_GATE_C_V1_TARGETS = frozenset({"gate_c_init", "formal_gate_c"})
_GATE_C_TARGETS = frozenset(
    {*_GATE_C_V1_TARGETS, "gate_c2_controls", "gate_c3_controls"}
)
_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_HEAD_PATTERN = re.compile(r"[0-9a-f]{40}")
_DEFAULT_IMAGE = "braintrace-gpu:0.11.0-py314"
_GATE_MODULE = "examples.pp_prop.latent_workspace_binding_gate"
_DEPTH_GATE_MODULE = "examples.pp_prop.latent_workspace_depth_gate"
_ABLATION_GATE_MODULE = "examples.pp_prop.latent_workspace_ablation_gate"
_GATE_C_SOURCE_FILES = (
    "examples/pp_prop/latent_workspace_model.py",
    "examples/pp_prop/latent_workspace_task.py",
    "examples/pp_prop/latent_workspace_binding_control.py",
    "examples/pp_prop/latent_workspace_binding_gate.py",
    "examples/pp_prop/latent_workspace_depth_gate.py",
    "examples/pp_prop/latent_workspace_ablation_gate.py",
)
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
_GATE_B_SOURCE_COMMIT = "dafa64a8b4c3848241baa117affa55b632518a8e"
_GATE_B_IMAGE_ID = (
    "sha256:35349cb07c49e275b15c5c563a8d75fa08b49d4b0829d86939c1c09fb1ef6d16"
)
_GATE_B_PREFLIGHT_SHA256 = (
    "91e86d92670cd33d3f4206ff3d5096e3721104996a9506223a9e34c082dd052f"
)
_GATE_B_RESULT_SHA256 = (
    "6456537ea108cea8892d00c8a71c1f647217e074b525bc9ed01b64aef9001766"
)
_GATE_B_MANIFEST_SHA256 = (
    "99c42985e203413eb0600a5dabe321188776eff8058500dc86f4a1618b413eab"
)
_GATE_B_BUNDLE_SHA256 = (
    "be07e8c92d8deaa94508f34dcee45f5feb09740cb2804778d6280a2fa3c64851"
)
_GATE_B_DIRECTORY = Path("var/example21-depth-gate")
_GATE_C_DIRECTORY = Path("var/example21-causal-gate")
_GATE_C3_DETERMINISTIC_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "XLA_FLAGS": "--xla_gpu_deterministic_ops=true",
}
_GATE_C3_QUALIFICATION_CRITERIA = (
    "schema_and_control",
    "exact_configuration",
    "prerequisites_authenticated",
    "initialization_authenticated",
    "deterministic_environment_authenticated",
    "canonical_schedules_complete",
    "no_behavioral_or_optimizer_updates",
    "paired_h0_operational_equivalence",
    "no_read_and_removed_path_complete",
    "mechanism_oracle_complete",
    "source_and_gpu_authenticated",
)


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
    target : {"one_update", "stability_256", "formal_gate_a", "gate_b_init", "formal_gate_b", "gate_c_init", "formal_gate_c", "gate_c2_controls", "gate_c3_controls"}
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
            raise ValueError(f"Target must be one of {_TARGETS!r}. Set Target to one of {_TARGETS!r}.")
        root = Path(self.repo_root).resolve()
        output = Path(self.output_dir).resolve()
        if not output.is_relative_to(root):
            raise ValueError("output_dir must be contained by repo_root. Set output_dir to contained by repo_root.")
        if output == root:
            raise ValueError("output_dir cannot be the repository root. Fix the input condition named in the error, then rerun the operation.")
        if self.target in _GATE_C_TARGETS and output != (root / _GATE_C_DIRECTORY):
            raise ValueError(
                "Gate C output_dir must be repo_root/var/example21-causal-gate. Set Gate C output_dir to repo_root/var/example21-causal-gate."
            )
        if not self.image.strip():
            raise ValueError("Image must be nonempty. Set Image to nonempty.")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", self.cache_volume):
            raise ValueError("cache_volume contains unsupported characters. Use a supported option or change the configuration.")
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
    Stdout, stderr : str
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
        raise ValueError(f"Non-finite JSON constant {value!r}. Use finite values.")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r}. Fix the input condition named in the error, then rerun the operation.")
            result[key] = value
        return result

    value = msgspec_json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject,
        object_pairs_hook=unique_pairs,
    )
    if not isinstance(value, dict):
        raise ValueError("Artifact must contain a top-level JSON object. Add a top-level JSON object to Artifact.")
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
    payload = msgspec_json.dumps(
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
        raise ValueError("Head must be a full lowercase 40-hex commit. Set Head to a full lowercase 40-hex commit.")
    selected = config.target if target is None else target
    if selected not in _TARGETS:
        raise ValueError(f"Unknown target {selected!r}. Set the named field to one of the supported values, then rerun the operation.")
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


def _formal_gate_b_artifact_paths(config: LaunchConfig) -> TargetPaths:
    stem = f"{_GATE_B_SOURCE_COMMIT}-formal-gate-b"
    result = config.repo_root / _GATE_B_DIRECTORY / f"{stem}.json"
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
        return msgspec_json.dumps(
            left, allow_nan=False, sort_keys=True, separators=(",", ":")
        ) == msgspec_json.dumps(
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
            f"{label} failed with exit {record.returncode}: {record.stderr.strip()}. Correct the reported inputs, then retry the operation."
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
        raise ProvenanceError(f"Unexpected worktree root {root}. Use the expected value or update the contract.")
    if not git_dir.is_relative_to(common):
        raise ProvenanceError("Worktree Git directory is outside the common Git directory. Set the named field to a value in the stated range, then rerun the operation.")
    if not _HEAD_PATTERN.fullmatch(head):
        raise ProvenanceError("Live HEAD is not a full lowercase 40-hex commit. Free resources or reduce the allocation.")
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
        values = msgspec_json.loads(raw)
        if not isinstance(values, list) or len(values) != 1:
            raise ValueError("Expected exactly one local image. Fix the input condition named in the error, then rerun the operation.")
        image = values[0]
        image_id = str(image["Id"])
        revision = str(image["Config"]["Labels"]["org.opencontainers.image.revision"])
    except (KeyError, TypeError, ValueError, msgspec_json.JSONDecodeError) as error:
        raise ProvenanceError(f"Invalid local image inspection: {error}. Set the named field to a value in the stated range, then rerun the operation.") from error
    if not _IMAGE_ID_PATTERN.fullmatch(image_id):
        raise ProvenanceError("Local image ID is not an immutable sha256 digest. Fix the input condition named in the error, then rerun the operation.")
    if revision != head:
        raise ProvenanceError(
            f"OCI revision {revision!r} does not equal clean HEAD {head!r}. Fix the input condition named in the error, then rerun the operation."
        )
    return ImageIdentity(config.image, image_id, revision, record)


def _container_environment(
    head: str,
    image_id: str,
    git_dir: str,
    *,
    target: LaunchTarget | None = None,
) -> dict[str, str]:
    environment = {
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
    if target == "gate_c3_controls":
        environment.update(_GATE_C3_DETERMINISTIC_ENVIRONMENT)
    return environment


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
    explicit = _container_environment(
        source.head,
        image.image_id,
        git_dir_in_container,
        target=config.target,
    )
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
        raise ProvenanceError("Qualifying image does not report a valid Git version. Fix the input condition named in the error, then rerun the operation.")
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
        raise ProvenanceError(f"{label} artifact reference is missing. Provide the missing value or resource, then rerun the operation.")
    relative = expected_path.resolve().relative_to(repo_root.resolve()).as_posix()
    if (
        not expected_path.is_file()
        or value.get("path") != relative
        or value.get("repo_relative_path") != relative
        or value.get("sha256") != sha256_file(expected_path)
        or not _is_integer(value.get("size_bytes"))
        or int(value["size_bytes"]) != expected_path.stat().st_size
    ):
        raise ProvenanceError(f"{label} artifact digest/path mismatch. Use matching values and structures.")
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
            raise ProvenanceError("Authenticated Gate A prerequisite is missing. Provide the missing value or resource, then rerun the operation.")
    manifest_sha256 = sha256_file(paths.manifest)
    result_sha256 = sha256_file(paths.result)
    if manifest_sha256 != _GATE_A_MANIFEST_SHA256:
        raise ProvenanceError("Authenticated Gate A manifest SHA-256 mismatch. Use matching values and structures.")
    if result_sha256 != _GATE_A_RESULT_SHA256:
        raise ProvenanceError("Authenticated Gate A result SHA-256 mismatch. Use matching values and structures.")

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
        raise ProvenanceError("Authenticated Gate A manifest is invalid. Set the named field to a value in the stated range, then rerun the operation.")
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
        raise ProvenanceError("Authenticated Gate A bundle digest is invalid. Set the named field to a value in the stated range, then rerun the operation.")

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
        raise ProvenanceError("Authenticated Gate A result is invalid. Set the named field to a value in the stated range, then rerun the operation.")
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
        raise ProvenanceError("Retained command record has an unexpected schema. Use the expected value or update the contract.")
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
        raise ProvenanceError("Retained command record is incomplete or inconsistent. Use matching values and structures.")
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
    target: LaunchTarget,
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
        raise ProvenanceError(f"Required {target} preflight evidence is invalid. Set the named field to a value in the stated range, then rerun the operation.")

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
        raise ProvenanceError("Preflight does not retain the required mount contract. Fix the input condition named in the error, then rerun the operation.")

    planned_environment = planned_gate["environment"]
    container_environment = container_source.get("environment")
    if not isinstance(planned_environment, Mapping) or not isinstance(
        container_environment, Mapping
    ):
        raise ProvenanceError("Preflight omits the exact container environment. Fix the input condition named in the error, then rerun the operation.")
    git_dir = str(planned_environment.get("GIT_DIR", ""))
    git_path = PurePosixPath(git_dir)
    if (
        not git_path.is_absolute()
        or not git_path.is_relative_to(PurePosixPath("/git-common"))
        or ".." in git_path.parts
    ):
        raise ProvenanceError("Preflight GIT_DIR is outside the read-only Git mount. Set the named field to a value in the stated range, then rerun the operation.")
    expected_environment = _container_environment(
        head,
        image_id,
        git_dir,
        target=target,
    )
    if (
        dict(planned_environment) != expected_environment
        or dict(container_environment) != expected_environment
    ):
        raise ProvenanceError("Preflight environment differs from the fixed contract. Use matching values and structures.")

    host_commands = source.get("commands")
    if not isinstance(host_commands, list) or len(host_commands) != 5:
        raise ProvenanceError("Preflight omits host Git command evidence. Fix the input condition named in the error, then rerun the operation.")
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
        raise ProvenanceError("Host Git command argv differs from preflight. Use matching values and structures.")
    if host_outputs[3].strip().lower() != head or host_outputs[4].strip() != "":
        raise ProvenanceError("Host Git command evidence disagrees with clean HEAD. Use matching values and structures.")
    if any(
        _portable_host_path(host_outputs[index])
        != _portable_host_path(source[field])
        for index, field in enumerate(("root", "common_git_dir", "git_dir"))
    ):
        raise ProvenanceError("Host Git paths disagree with retained source paths. Use matching values and structures.")

    inspect_record = image.get("inspect_command")
    if not isinstance(inspect_record, Mapping):
        raise ProvenanceError("Preflight omits immutable image inspection evidence. Fix the input condition named in the error, then rerun the operation.")
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
        raise ProvenanceError("Retained image inspection argv is not exact. Fix the input condition named in the error, then rerun the operation.")
    try:
        inspection = msgspec_json.loads(inspect_stdout)
        if not isinstance(inspection, list) or len(inspection) != 1:
            raise TypeError("Expected exactly one retained image inspection. Fix the input condition named in the error, then rerun the operation.")
        inspected = inspection[0]
        inspected_id = inspected["Id"]
        inspected_revision = inspected["Config"]["Labels"][
            "org.opencontainers.image.revision"
        ]
    except (IndexError, KeyError, TypeError, msgspec_json.JSONDecodeError) as error:
        raise ProvenanceError("Retained image inspection output is invalid. Set the named field to a value in the stated range, then rerun the operation.") from error
    if inspected_id != image_id or inspected_revision != head:
        raise ProvenanceError("Retained image inspection output disagrees with identity. Use matching values and structures.")

    container_commands = container_source.get("commands")
    if not isinstance(container_commands, list) or len(container_commands) != 3:
        raise ProvenanceError("Preflight omits container Git command evidence. Fix the input condition named in the error, then rerun the operation.")
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
        raise ProvenanceError("Container Git command argv differs from preflight. Use matching values and structures.")
    if (
        not container_outputs[0].strip().startswith("git version ")
        or container_outputs[0].strip() != container_source.get("git_version")
        or container_outputs[1].strip().lower() != head
        or container_outputs[2].strip() != ""
    ):
        raise ProvenanceError("Container Git command evidence disagrees with clean HEAD. Use matching values and structures.")

    argv = planned_gate["argv"]
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ProvenanceError("Planned Gate command is not shell-free argv. Fix the input condition named in the error, then rerun the operation.")
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
        raise ProvenanceError("Planned Gate command mount argv is not fixed. Fix the input condition named in the error, then rerun the operation.")
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
        raise ProvenanceError("Planned Gate mount sources disagree with the sidecar. Use matching values and structures.")
    container_result = str(
        PurePosixPath("/work") / expected_result.relative_to(repo_root).as_posix()
    )
    if target in _GATE_B_TARGETS:
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
        ]
        if target == "formal_gate_b":
            gate_b_base = PurePosixPath("/work") / _GATE_B_DIRECTORY.as_posix()
            expected_tail.extend(
                [
                    "--gate-b-init-manifest",
                    str(gate_b_base / f"{head}-gate-b-init.manifest.json"),
                ]
            )
        expected_tail.extend(["--output", container_result])
    elif target in _GATE_C_TARGETS:
        gate_a_stem = f"{_GATE_A_SOURCE_COMMIT}-formal-gate-a"
        gate_a_base = PurePosixPath("/work") / _GATE_A_DIRECTORY.as_posix()
        gate_b_stem = f"{_GATE_B_SOURCE_COMMIT}-formal-gate-b.manifest.json"
        gate_b_base = PurePosixPath("/work") / _GATE_B_DIRECTORY.as_posix()
        expected_tail = [
            image_id,
            "python",
            "-m",
            _ABLATION_GATE_MODULE,
            "--target",
            target,
            "--gate-a-result",
            str(gate_a_base / f"{gate_a_stem}.json"),
            "--gate-a-manifest",
            str(gate_a_base / f"{gate_a_stem}.manifest.json"),
            "--gate-b-manifest",
            str(gate_b_base / gate_b_stem),
        ]
        if target in {
            "formal_gate_c",
            "gate_c2_controls",
            "gate_c3_controls",
        }:
            gate_c_base = PurePosixPath("/work") / _GATE_C_DIRECTORY.as_posix()
            expected_tail.extend(
                [
                    "--gate-c-init-manifest",
                    str(gate_c_base / f"{head}-gate-c-init.manifest.json"),
                ]
            )
        expected_tail.extend(["--output", container_result])
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
        raise ProvenanceError("Planned Gate argv differs from its fixed target. Use matching values and structures.")


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
        raise ProvenanceError("Admission manifest omits the executed Gate command. Fix the input condition named in the error, then rerun the operation.")
    host_cwd = _authenticated_host_cwd(preflight["source"], repo_root)
    _validated_command_record(
        gate_record,
        argv_tail=planned["argv"],
        environment=expected_environment,
        cwd=host_cwd,
    )
    if gate_record["argv"] != planned["argv"]:
        raise ProvenanceError("Executed Gate command differs from preflight. Use matching values and structures.")

    postflight = manifest.get("postflight")
    if (
        manifest.get("failure") is not None
        or not isinstance(postflight, Mapping)
        or postflight.get("head") != head
        or postflight.get("clean") is not True
    ):
        raise ProvenanceError("Admission postflight did not retain clean source. Fix the input condition named in the error, then rerun the operation.")
    host_commands = postflight.get("host_commands")
    if not isinstance(host_commands, list) or len(host_commands) != 5:
        raise ProvenanceError("Admission postflight omits host Git commands. Fix the input condition named in the error, then rerun the operation.")
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
        raise ProvenanceError("Admission host postflight argv is not exact. Fix the input condition named in the error, then rerun the operation.")
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
        raise ProvenanceError("Admission host postflight disagrees with clean HEAD. Use matching values and structures.")

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
        raise ProvenanceError("Admission container postflight is incomplete. Fix the input condition named in the error, then rerun the operation.")
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
        raise ProvenanceError("Admission container postflight argv is not exact. Fix the input condition named in the error, then rerun the operation.")
    if (
        not container_outputs[0].strip().startswith("git version ")
        or container_outputs[0].strip() != container.get("git_version")
        or container_outputs[1].strip().lower() != head
        or container_outputs[2].strip() != ""
    ):
        raise ProvenanceError("Admission container postflight disagrees with clean HEAD. Use matching values and structures.")


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
        raise ProvenanceError(f"{target} result provenance/configuration is invalid. Set the named field to a value in the stated range, then rerun the operation.")

    qualification = result.get("qualification")
    if not isinstance(qualification, Mapping):
        raise ProvenanceError(f"{target} result qualification is missing. Provide the missing value or resource, then rerun the operation.")
    qualifier = (
        depth._gate_b_initialization_qualification
        if target == "gate_b_init"
        else depth._qualification_report
    )
    recomputed = qualifier(result, config=config)
    if not _json_exact(qualification, recomputed):
        raise ProvenanceError(f"{target} scientific qualification does not recompute. Fix the input condition named in the error, then rerun the operation.")
    return bool(recomputed["passed"])


def _validate_gate_c_scientific_result(
    result: Mapping[str, Any],
    *,
    target: Literal["gate_c_init", "formal_gate_c"],
    head: str,
    image_id: str,
    prerequisites: Mapping[str, Any],
) -> bool:
    """Recompute a fixed Gate C qualification from retained raw evidence."""

    from examples.pp_prop import latent_workspace_ablation_gate as gate_c

    if target not in _GATE_C_V1_TARGETS:
        raise ProvenanceError("Unsupported Gate C launch target. Use a supported option or change the configuration.")
    config = gate_c.GateCConfig()
    formal = target == "formal_gate_c"
    source_start = result.get("source_start")
    source_end = result.get("source_end")
    environment = result.get("environment")
    devices = environment.get("devices") if isinstance(environment, Mapping) else None
    regimes = result.get("regimes")
    source_files = result.get("source_files")
    arms = result.get("arms")
    formal_arm_names = {
        "full",
        "query_only",
        "terminal_only",
        "legacy",
        "frozen_write",
    }
    expected_result_keys = (
        {
            "schema_version",
            "control",
            "qualification_regime",
            "learner",
            "prerequisites",
            "regimes",
            "arms",
            "mechanism_oracle",
            "source_start",
            "source_end",
            "source_files",
            "environment",
            "total_wall_seconds",
            "qualification",
        }
        if formal
        else {
            "schema_version",
            "control",
            "qualification_regime",
            "prerequisites",
            "regimes",
            "initialization",
            "source_start",
            "source_end",
            "source_files",
            "environment",
            "qualification",
        }
    )
    expected_prerequisite_names = (
        {"gate_a", "gate_b", "gate_c_initialization"}
        if formal
        else {"gate_a", "gate_b"}
    )
    if formal and isinstance(regimes, Mapping):
        target_shape_valid = (
            isinstance(arms, Mapping)
            and set(arms) == formal_arm_names
            and all(
                isinstance(arms[name], Mapping)
                and set(arms[name]) == {"gate_a", "gate_b"}
                and all(
                    isinstance(arms[name][regime], Mapping)
                    for regime in ("gate_a", "gate_b")
                )
                for name in formal_arm_names
            )
            and isinstance(result.get("mechanism_oracle"), Mapping)
            and result.get("learner") == "pp_prop_only"
            and _is_finite_real(result.get("total_wall_seconds"))
            and float(result["total_wall_seconds"]) >= 0.0
        )
    else:
        target_shape_valid = not formal and isinstance(
            result.get("initialization"), Mapping
    )
    if (
        set(result) != expected_result_keys
        or not isinstance(prerequisites, Mapping)
        or set(prerequisites) != expected_prerequisite_names
        or not _is_integer(result.get("schema_version"))
        or int(result["schema_version"]) != gate_c.GATE_C_SCHEMA_VERSION
        or result.get("control")
        != (gate_c.GATE_C_CONTROL if formal else gate_c.GATE_C_INITIALIZATION_CONTROL)
        or result.get("qualification_regime") != "preregistered_full"
        or not _json_exact(result.get("prerequisites"), prerequisites)
        or not isinstance(regimes, Mapping)
        or set(regimes) != {"gate_a", "gate_b"}
        or not all(isinstance(regimes[name], Mapping) for name in regimes)
        or not target_shape_valid
        or not isinstance(source_start, Mapping)
        or not _source_report_matches(source_start, head)
        or not isinstance(source_end, Mapping)
        or not _source_report_matches(source_end, head)
        or not isinstance(source_files, Mapping)
        or set(source_files) != set(_GATE_C_SOURCE_FILES)
        or not all(
            isinstance(source_files[path], str)
            and re.fullmatch(r"[0-9a-f]{64}", source_files[path])
            for path in _GATE_C_SOURCE_FILES
        )
        or not isinstance(environment, Mapping)
        or environment.get("image_digest") != image_id
        or environment.get("backend") != "gpu"
        or not isinstance(devices, list)
        or not devices
        or not all(isinstance(device, Mapping) for device in devices)
        or not any(device.get("platform") == "gpu" for device in devices)
    ):
        raise ProvenanceError(f"{target} result provenance/configuration is invalid. Set the named field to a value in the stated range, then rerun the operation.")
    qualification = result.get("qualification")
    if not isinstance(qualification, Mapping):
        raise ProvenanceError(f"{target} result qualification is missing. Provide the missing value or resource, then rerun the operation.")
    qualifier = (
        gate_c._qualification_report
        if formal
        else gate_c._gate_c_initialization_qualification
    )
    recomputed = qualifier(result, config=config)
    if not _json_exact(qualification, recomputed):
        raise ProvenanceError(f"{target} scientific qualification does not recompute. Fix the input condition named in the error, then rerun the operation.")
    return bool(recomputed["passed"])


def _validate_gate_c_controls_result_shape(
    result: Mapping[str, Any],
    *,
    target: Literal["gate_c2_controls", "gate_c3_controls"],
    head: str,
    image_id: str,
    prerequisites: Mapping[str, Any],
    schema_version: int,
    control: str,
    qualification_regime: str,
    deterministic_environment: Mapping[str, str] | None,
) -> None:
    expected_result_keys = {
        "schema_version",
        "control",
        "qualification_regime",
        "learner",
        "prerequisites",
        "regimes",
        "mechanism_oracle",
        "source_start",
        "source_end",
        "source_files",
        "environment",
        "qualification",
        "total_wall_seconds",
    }
    expected_prerequisite_names = {
        "gate_a",
        "gate_b",
        "gate_c_initialization",
    }
    expected_regime_keys = {
        "spec",
        "config",
        "schedule_identity",
        "paired_h0_operational_equivalence",
        "query_only_latent_no_read",
    }
    expected_environment_keys = {
        "backend",
        "devices",
        "image_digest",
        "jax",
        "python",
        "execution_and_update_evidence",
    }
    if deterministic_environment is not None:
        expected_environment_keys.add("deterministic_environment")
    expected_execution_keys = {
        "instrumented_training_entry_points",
        "trainer_factory_calls",
        "trainer_factory_call_count",
        "training_step_calls",
        "training_step_call_count",
        "optimizer_constructor_calls",
        "optimizer_instance_count",
        "optimizer_update_calls",
        "optimizer_update_call_count",
        "model_factory_calls",
        "model_constructor_calls",
        "materialized_roles",
        "complete",
    }
    source_start = result.get("source_start")
    source_end = result.get("source_end")
    source_files = result.get("source_files")
    regimes = result.get("regimes")
    mechanism_oracle = result.get("mechanism_oracle")
    environment = result.get("environment")
    devices = environment.get("devices") if isinstance(environment, Mapping) else None
    execution = (
        environment.get("execution_and_update_evidence")
        if isinstance(environment, Mapping)
        else None
    )
    retained_deterministic_environment = (
        environment.get("deterministic_environment")
        if isinstance(environment, Mapping)
        else None
    )
    deterministic_environment_valid = (
        retained_deterministic_environment is None
        if deterministic_environment is None
        else isinstance(retained_deterministic_environment, Mapping)
        and dict(retained_deterministic_environment)
        == dict(deterministic_environment)
    )
    regime_shape_valid = (
        isinstance(regimes, Mapping)
        and set(regimes) == {"gate_a", "gate_b"}
        and all(
            isinstance(regimes[name], Mapping)
            and set(regimes[name]) == expected_regime_keys
            and all(
                isinstance(regimes[name][field], Mapping)
                for field in expected_regime_keys
            )
            for name in ("gate_a", "gate_b")
        )
    )
    execution_shape_valid = (
        isinstance(execution, Mapping)
        and set(execution) == expected_execution_keys
        and isinstance(execution.get("instrumented_training_entry_points"), list)
        and all(
            isinstance(execution.get(field), list)
            for field in (
                "trainer_factory_calls",
                "training_step_calls",
                "optimizer_constructor_calls",
                "optimizer_update_calls",
                "model_factory_calls",
                "model_constructor_calls",
            )
        )
        and all(
            _is_integer(execution.get(field))
            and int(execution[field]) >= 0
            for field in (
                "trainer_factory_call_count",
                "training_step_call_count",
                "optimizer_instance_count",
                "optimizer_update_call_count",
            )
        )
        and isinstance(execution.get("materialized_roles"), Mapping)
        and isinstance(execution.get("complete"), bool)
    )
    if (
        set(result) != expected_result_keys
        or not isinstance(prerequisites, Mapping)
        or set(prerequisites) != expected_prerequisite_names
        or not _is_integer(result.get("schema_version"))
        or int(result["schema_version"]) != schema_version
        or result.get("control") != control
        or result.get("qualification_regime") != qualification_regime
        or result.get("learner") != "pp_prop_only"
        or not _json_exact(result.get("prerequisites"), prerequisites)
        or not regime_shape_valid
        or not isinstance(mechanism_oracle, Mapping)
        or not isinstance(source_start, Mapping)
        or not _source_report_matches(source_start, head)
        or not isinstance(source_end, Mapping)
        or not _source_report_matches(source_end, head)
        or not isinstance(source_files, Mapping)
        or set(source_files) != set(_GATE_C_SOURCE_FILES)
        or not all(
            isinstance(source_files[path], str)
            and re.fullmatch(r"[0-9a-f]{64}", source_files[path])
            for path in _GATE_C_SOURCE_FILES
        )
        or not isinstance(environment, Mapping)
        or set(environment) != expected_environment_keys
        or not deterministic_environment_valid
        or environment.get("image_digest") != image_id
        or environment.get("backend") != "gpu"
        or not isinstance(devices, list)
        or not devices
        or not all(isinstance(device, Mapping) for device in devices)
        or not any(device.get("platform") == "gpu" for device in devices)
        or not execution_shape_valid
        or not _is_finite_real(result.get("total_wall_seconds"))
        or float(result["total_wall_seconds"]) < 0.0
    ):
        raise ProvenanceError(
            f"{target} result provenance/configuration is invalid. Set the named field to a value in the stated range, then rerun the operation."
        )


def _validate_gate_c2_controls_scientific_result(
    result: Mapping[str, Any],
    *,
    head: str,
    image_id: str,
    prerequisites: Mapping[str, Any],
) -> bool:
    """Recompute the Gate C2 pretraining-control admission."""

    from examples.pp_prop import latent_workspace_ablation_gate as gate_c

    config = gate_c.GateCConfig()
    _validate_gate_c_controls_result_shape(
        result,
        target="gate_c2_controls",
        head=head,
        image_id=image_id,
        prerequisites=prerequisites,
        schema_version=gate_c.GATE_C2_CONTROLS_SCHEMA_VERSION,
        control=gate_c.GATE_C2_CONTROLS_CONTROL,
        qualification_regime="preregistered_gate_c2_pretraining_controls",
        deterministic_environment=None,
    )
    qualification = result.get("qualification")
    if not isinstance(qualification, Mapping):
        raise ProvenanceError("gate_c2_controls result qualification is missing. Provide the missing value or resource, then rerun the operation.")
    recomputed = gate_c._gate_c2_controls_qualification(result, config=config)
    if not _json_exact(qualification, recomputed):
        raise ProvenanceError(
            "gate_c2_controls scientific qualification does not recompute. Fix the input condition named in the error, then rerun the operation."
        )
    return bool(recomputed["passed"])


def _validate_gate_c3_controls_scientific_result(
    result: Mapping[str, Any],
    *,
    head: str,
    image_id: str,
    prerequisites: Mapping[str, Any],
) -> bool:
    """Recompute the Gate C3 pretraining-control admission."""

    from examples.pp_prop import latent_workspace_ablation_gate as gate_c

    config = gate_c.GateCConfig()
    _validate_gate_c_controls_result_shape(
        result,
        target="gate_c3_controls",
        head=head,
        image_id=image_id,
        prerequisites=prerequisites,
        schema_version=gate_c.GATE_C3_CONTROLS_SCHEMA_VERSION,
        control=gate_c.GATE_C3_CONTROLS_CONTROL,
        qualification_regime="preregistered_gate_c3_pretraining_controls",
        deterministic_environment=_GATE_C3_DETERMINISTIC_ENVIRONMENT,
    )
    qualification = result.get("qualification")
    expected_qualification_keys = {
        "valid",
        "passed",
        "criteria",
        "failures",
        "interpretation",
    }
    if not isinstance(qualification, Mapping) or set(
        qualification
    ) != expected_qualification_keys:
        raise ProvenanceError(
            "gate_c3_controls result qualification is missing or invalid. Provide the missing value or resource, then rerun the operation."
        )
    recomputed = gate_c._gate_c3_controls_qualification(result, config=config)
    if not isinstance(recomputed, Mapping) or not _json_exact(
        qualification, recomputed
    ):
        raise ProvenanceError(
            "gate_c3_controls scientific qualification does not recompute. Fix the input condition named in the error, then rerun the operation."
        )
    criteria = recomputed.get("criteria")
    failures = recomputed.get("failures")
    valid = recomputed.get("valid")
    passed = recomputed.get("passed")
    expected_failures = (
        sorted(name for name, value in criteria.items() if value is False)
        if isinstance(criteria, Mapping)
        else None
    )
    expected_interpretation = (
        "gate_c3_pretraining_controls_invalid_stop"
        if valid is False
        else (
            "gate_c3_pretraining_controls_passed"
            if passed is True
            else "gate_c3_pretraining_controls_failed_stop"
        )
    )
    if (
        set(recomputed) != expected_qualification_keys
        or not isinstance(valid, bool)
        or not isinstance(passed, bool)
        or not isinstance(criteria, Mapping)
        or set(criteria) != set(_GATE_C3_QUALIFICATION_CRITERIA)
        or not all(isinstance(value, bool) for value in criteria.values())
        or not isinstance(failures, list)
        or failures != expected_failures
        or recomputed.get("interpretation") != expected_interpretation
        or passed is not (valid and all(criteria.values()))
    ):
        raise ProvenanceError(
            "gate_c3_controls scientific qualification is invalid. Set the named field to a value in the stated range, then rerun the operation."
        )
    if valid is not True:
        raise ProvenanceError("gate_c3_controls result qualification is invalid. Set the named field to a value in the stated range, then rerun the operation.")
    return passed


def _validate_fixed_config(result: Mapping[str, Any], target: LaunchTarget) -> None:
    config = result.get("config")
    if not isinstance(config, Mapping):
        raise ProvenanceError("Result config is missing. Provide the missing value or resource, then rerun the operation.")
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
    gate_c_prerequisites: Mapping[str, Any] | None = None,
) -> bool:
    if target in _GATE_C_TARGETS:
        if (
            admission_manifests is not None
            or gate_a_prerequisite is not None
            or gate_b_init_bundle is not None
            or gate_c_prerequisites is None
        ):
            raise ProvenanceError("Gate C prerequisites are invalid. Set the named field to a value in the stated range, then rerun the operation.")
        if target == "gate_c2_controls":
            return _validate_gate_c2_controls_scientific_result(
                result,
                head=head,
                image_id=image_id,
                prerequisites=gate_c_prerequisites,
            )
        if target == "gate_c3_controls":
            return _validate_gate_c3_controls_scientific_result(
                result,
                head=head,
                image_id=image_id,
                prerequisites=gate_c_prerequisites,
            )
        return _validate_gate_c_scientific_result(
            result,
            target=target,
            head=head,
            image_id=image_id,
            prerequisites=gate_c_prerequisites,
        )
    if target in _GATE_B_TARGETS:
        if admission_manifests is not None:
            raise ProvenanceError("Gate B targets do not accept Gate A admissions. Fix the input condition named in the error, then rerun the operation.")
        prerequisites = result.get("prerequisites")
        embedded_gate_a = (
            prerequisites.get("gate_a")
            if isinstance(prerequisites, Mapping)
            else None
        )
        if gate_a_prerequisite is None or not _json_exact(
            embedded_gate_a, gate_a_prerequisite
        ):
            raise ProvenanceError("Gate B result does not bind Gate A prerequisite. Fix the input condition named in the error, then rerun the operation.")
        if target == "formal_gate_b":
            embedded_init = prerequisites.get("gate_b_initialization")
            if gate_b_init_bundle is None or not _json_exact(
                embedded_init, gate_b_init_bundle
            ):
                raise ProvenanceError(
                    "Formal Gate B result does not bind initialization manifest. Fix the input condition named in the error, then rerun the operation."
                )
        elif gate_b_init_bundle is not None:
            raise ProvenanceError("Gate B initialization received a circular prerequisite. Fix the input condition named in the error, then rerun the operation.")
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
        raise ProvenanceError("Result schema/control does not match launch target. Use matching values and structures.")
    if target != "formal_gate_a" and result.get("target") != target:
        raise ProvenanceError("Admission result target mismatch. Use matching values and structures.")
    _validate_fixed_config(result, target)
    source = result.get("source")
    source_end = result.get("source_end")
    if not isinstance(source, Mapping) or not _source_report_matches(source, head):
        raise ProvenanceError("Result start source evidence is not authenticated. Fix the input condition named in the error, then rerun the operation.")
    if not isinstance(source_end, Mapping) or not _source_report_matches(source_end, head):
        raise ProvenanceError("Result end source evidence is not authenticated. Fix the input condition named in the error, then rerun the operation.")
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
        raise ProvenanceError("Result image/backend evidence is not authenticated. Fix the input condition named in the error, then rerun the operation.")
    qualification = result.get("qualification")
    if not isinstance(qualification, Mapping) or not isinstance(
        qualification.get("passed"), bool
    ):
        raise ProvenanceError("Result qualification is incomplete. Fix the input condition named in the error, then rerun the operation.")
    if target != "formal_gate_a":
        admission = result.get("admission")
        if not isinstance(admission, Mapping):
            raise ProvenanceError("Admission result omits its inner evidence. Fix the input condition named in the error, then rerun the operation.")
        if not _json_exact(admission.get("config"), result.get("config")):
            raise ProvenanceError("Admission envelope config differs from inner evidence. Use matching values and structures.")
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
            raise ProvenanceError("Admission qualification does not recompute exactly. Fix the input condition named in the error, then rerun the operation.")
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
                "Formal result initialization/schedule differs from admissions. Use matching values and structures."
            )
        admissions = result.get("stage21_admissions")
        if (
            not isinstance(admissions, Mapping)
            or set(admissions) != {"one_update", "stability_256"}
            or admission_manifests is None
            or set(admission_manifests) != {"one_update", "stability_256"}
        ):
            raise ProvenanceError("Formal result omits authenticated admissions. Fix the input condition named in the error, then rerun the operation.")
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
                raise ProvenanceError(f"Formal result does not bind {name} manifest. Fix the input condition named in the error, then rerun the operation.")
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
            raise ProvenanceError("Formal scientific evidence is incomplete. Fix the input condition named in the error, then rerun the operation.") from error
        if (
            result.get("learner") != "pp_prop_only"
            or not _json_exact(qualification, recomputed)
            or result.get("interpretation") != recomputed["interpretation"]
        ):
            raise ProvenanceError("Formal scientific qualification does not recompute. Fix the input condition named in the error, then rerun the operation.")
        return bool(recomputed["passed"])
    raise AssertionError("Unreachable launch target. Fix the input condition named in the error, then rerun the operation.")


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
        raise ProvenanceError(f"Required {target} admission manifest path is invalid. Set the named field to a value in the stated range, then rerun the operation.")
    expected_stem = f"{head}-{target.replace('_', '-')}"
    expected_manifest = path.parent / f"{expected_stem}.manifest.json"
    expected_preflight = path.parent / f"{expected_stem}.preflight.json"
    expected_result = path.parent / f"{expected_stem}.json"
    if path != expected_manifest:
        raise ProvenanceError(f"Required {target} admission manifest path is not fixed. Fix the input condition named in the error, then rerun the operation.")
    for artifact in (expected_manifest, expected_preflight, expected_result):
        if not artifact.is_file() or not artifact.resolve().is_relative_to(root):
            raise ProvenanceError(f"Required {target} admission bundle is incomplete. Fix the input condition named in the error, then rerun the operation.")

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
        raise ProvenanceError(f"Required {target} admission manifest is invalid. Set the named field to a value in the stated range, then rerun the operation.")
    references = {
        "preflight": (value.get("preflight"), expected_preflight),
        "result": (value.get("result"), expected_result),
    }
    for label, (reference, expected_path) in references.items():
        if not isinstance(reference, Mapping):
            raise ProvenanceError(f"Required {target} {label} reference is missing. Provide the missing value or resource, then rerun the operation.")
        if (
            reference.get("path") != expected_path.relative_to(root).as_posix()
            or reference.get("repo_relative_path")
            != expected_path.relative_to(root).as_posix()
            or reference.get("sha256") != sha256_file(expected_path)
            or not _is_integer(reference.get("size_bytes"))
            or int(reference["size_bytes"]) != expected_path.stat().st_size
        ):
            raise ProvenanceError(f"Required {target} {label} digest/path mismatch. Use matching values and structures.")

    expected_bundle_sha256 = hashlib.sha256(
        (
            f"example21-launch-bundle-v1\0{target}\0{head}\0"
            f"{references['preflight'][0]['sha256']}\0"
            f"{references['result'][0]['sha256']}"
        ).encode("utf-8")
    ).hexdigest()
    if value.get("bundle_sha256") != expected_bundle_sha256:
        raise ProvenanceError(f"Required {target} bundle digest is invalid. Set the named field to a value in the stated range, then rerun the operation.")

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
        raise ProvenanceError(f"Required {target} admission did not pass. Fix the input condition named in the error, then rerun the operation.")
    admission = result.get("admission")
    if not isinstance(admission, Mapping):
        raise ProvenanceError(f"Required {target} admission report is missing. Provide the missing value or resource, then rerun the operation.")
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
            raise ProvenanceError("Required Gate B initialization manifest is missing. Provide the missing value or resource, then rerun the operation.")
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
        raise ProvenanceError("Required Gate B initialization manifest is invalid. Set the named field to a value in the stated range, then rerun the operation.")
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
        raise ProvenanceError("Required Gate B initialization bundle is invalid. Set the named field to a value in the stated range, then rerun the operation.")

    gate_a = _load_gate_a_prerequisite(config)
    preflight = load_strict_json(paths.preflight)
    prerequisites = preflight.get("gate_b_prerequisites")
    if (
        not isinstance(prerequisites, Mapping)
        or not _json_exact(prerequisites.get("gate_a"), gate_a)
        or prerequisites.get("gate_b_initialization") is not None
    ):
        raise ProvenanceError(
            "Gate B initialization preflight prerequisite binding is invalid. Set the named field to a value in the stated range, then rerun the operation."
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
        raise ProvenanceError("Required Gate B initialization admission did not pass. Fix the input condition named in the error, then rerun the operation.")
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


def load_authenticated_formal_gate_b(
    manifest_path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Validate and load the retained formal Gate B capability bundle.

    Parameters
    ----------
    manifest_path : str or pathlib.Path
        Exact retained formal Gate B manifest path.
    repo_root : str or pathlib.Path
        Worktree root containing Gate A, Gate B initialization, and formal
        Gate B artifacts.

    Returns
    -------
    dict
        Strictly authenticated manifest, preflight, result, prerequisite
        evidence, and retained content hashes for Gate C.
    """

    root = Path(repo_root).resolve()
    path = Path(manifest_path).resolve()
    config = LaunchConfig(
        target="formal_gate_b",
        repo_root=root,
        output_dir=root / _GATE_B_DIRECTORY,
    )
    paths = target_paths(config, _GATE_B_SOURCE_COMMIT, "formal_gate_b")
    if path != paths.manifest:
        raise ProvenanceError("Required formal Gate B manifest path is not fixed. Fix the input condition named in the error, then rerun the operation.")
    expected_hashes = {
        paths.preflight: _GATE_B_PREFLIGHT_SHA256,
        paths.result: _GATE_B_RESULT_SHA256,
        paths.manifest: _GATE_B_MANIFEST_SHA256,
    }
    for artifact, expected_sha256 in expected_hashes.items():
        if (
            not artifact.is_file()
            or not artifact.resolve().is_relative_to(root)
            or sha256_file(artifact) != expected_sha256
        ):
            raise ProvenanceError(
                "Required formal Gate B retained bytes are missing or changed. Provide the missing value or resource, then rerun the operation."
            )

    manifest = load_strict_json(paths.manifest)
    if (
        not _is_integer(manifest.get("schema_version"))
        or int(manifest["schema_version"]) != 1
        or manifest.get("kind") != "example21_authenticated_launch_manifest"
        or manifest.get("target") != "formal_gate_b"
        or manifest.get("source_head") != _GATE_B_SOURCE_COMMIT
        or manifest.get("bundle_valid") is not True
        or manifest.get("process_succeeded") is not True
        or manifest.get("artifact_schema_verified") is not True
        or manifest.get("scientific_qualification_passed") is not True
        or manifest.get("failure") is not None
    ):
        raise ProvenanceError("Required formal Gate B manifest is invalid. Set the named field to a value in the stated range, then rerun the operation.")
    preflight_reference = _validate_artifact_reference(
        manifest.get("preflight"),
        paths.preflight,
        repo_root=root,
        label="formal Gate B preflight",
    )
    result_reference = _validate_artifact_reference(
        manifest.get("result"),
        paths.result,
        repo_root=root,
        label="formal Gate B result",
    )
    bundle_sha256 = _launch_bundle_sha256(
        "formal_gate_b",
        _GATE_B_SOURCE_COMMIT,
        preflight_reference["sha256"],
        result_reference["sha256"],
    )
    if (
        bundle_sha256 != _GATE_B_BUNDLE_SHA256
        or manifest.get("bundle_sha256") != bundle_sha256
    ):
        raise ProvenanceError("Required formal Gate B bundle digest is invalid. Set the named field to a value in the stated range, then rerun the operation.")

    gate_a = _load_gate_a_prerequisite(config)
    gate_b_initialization = _load_gate_b_init_manifest(
        config,
        head=_GATE_B_SOURCE_COMMIT,
        image_id=_GATE_B_IMAGE_ID,
    )
    preflight = load_strict_json(paths.preflight)
    prerequisites = preflight.get("gate_b_prerequisites")
    if (
        not isinstance(prerequisites, Mapping)
        or not _json_exact(prerequisites.get("gate_a"), gate_a)
        or not _json_exact(
            prerequisites.get("gate_b_initialization"), gate_b_initialization
        )
    ):
        raise ProvenanceError("Formal Gate B preflight prerequisites are invalid. Set the named field to a value in the stated range, then rerun the operation.")
    _validate_preflight_semantics(
        preflight,
        target="formal_gate_b",
        head=_GATE_B_SOURCE_COMMIT,
        image_id=_GATE_B_IMAGE_ID,
        expected_result=paths.result,
        repo_root=root,
    )
    _validate_manifest_execution(
        manifest,
        preflight,
        head=_GATE_B_SOURCE_COMMIT,
        repo_root=root,
    )
    result = load_strict_json(paths.result)
    passed = _validate_target_result(
        result,
        target="formal_gate_b",
        head=_GATE_B_SOURCE_COMMIT,
        image_id=_GATE_B_IMAGE_ID,
        admission_manifests=None,
        repo_root=root,
        gate_a_prerequisite=gate_a,
        gate_b_init_bundle=gate_b_initialization,
    )
    if not passed:
        raise ProvenanceError("Required formal Gate B capability gate did not pass. Fix the input condition named in the error, then rerun the operation.")
    return {
        "target": "formal_gate_b",
        "source_head": _GATE_B_SOURCE_COMMIT,
        "image_digest": _GATE_B_IMAGE_ID,
        "bundle_sha256": bundle_sha256,
        "manifest_sha256": _GATE_B_MANIFEST_SHA256,
        "preflight_sha256": _GATE_B_PREFLIGHT_SHA256,
        "result_sha256": _GATE_B_RESULT_SHA256,
        "manifest": manifest,
        "preflight": preflight,
        "result": result,
        "gate_a": gate_a,
        "gate_b_initialization": gate_b_initialization,
    }


def _load_gate_c_prerequisites(config: LaunchConfig) -> dict[str, dict[str, Any]]:
    """Authenticate and compact the retained Gate A and Gate B bundles."""

    gate_a = _load_gate_a_prerequisite(config)
    gate_b_paths = _formal_gate_b_artifact_paths(config)
    gate_b_bundle = load_authenticated_formal_gate_b(
        gate_b_paths.manifest,
        repo_root=config.repo_root,
    )
    if (
        gate_b_bundle.get("target") != "formal_gate_b"
        or gate_b_bundle.get("source_head") != _GATE_B_SOURCE_COMMIT
    ):
        raise ProvenanceError("Authenticated formal Gate B identity is invalid. Set the named field to a value in the stated range, then rerun the operation.")
    gate_b = {
        "qualification_passed": True,
        "result_sha256": gate_b_bundle.get("result_sha256"),
        "manifest_sha256": gate_b_bundle.get("manifest_sha256"),
        "source_commit": gate_b_bundle.get("source_head"),
        "bundle_sha256": gate_b_bundle.get("bundle_sha256"),
        "preflight_sha256": gate_b_bundle.get("preflight_sha256"),
        "result_path": gate_b_paths.result.relative_to(
            config.repo_root
        ).as_posix(),
        "manifest_path": gate_b_paths.manifest.relative_to(
            config.repo_root
        ).as_posix(),
    }
    required = {
        "qualification_passed",
        "result_sha256",
        "manifest_sha256",
        "source_commit",
        "bundle_sha256",
        "preflight_sha256",
        "result_path",
        "manifest_path",
    }
    references = {"gate_a": dict(gate_a), "gate_b": gate_b}
    for name, reference in references.items():
        if set(reference) != required or reference["qualification_passed"] is not True:
            raise ProvenanceError(f"Authenticated {name} reference is incomplete. Fix the input condition named in the error, then rerun the operation.")
        if not _HEAD_PATTERN.fullmatch(str(reference["source_commit"])):
            raise ProvenanceError(f"Authenticated {name} source commit is invalid. Set the named field to a value in the stated range, then rerun the operation.")
        if any(
            not re.fullmatch(r"[0-9a-f]{64}", str(reference[field]))
            for field in (
                "result_sha256",
                "manifest_sha256",
                "bundle_sha256",
                "preflight_sha256",
            )
        ):
            raise ProvenanceError(f"Authenticated {name} digest is invalid. Set the named field to a value in the stated range, then rerun the operation.")
    return references


def _load_gate_c_init_manifest(
    config: LaunchConfig,
    *,
    head: str,
    image_id: str,
    base_prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the current-HEAD Gate C initialization bundle."""

    if config.target not in {
        "formal_gate_c",
        "gate_c2_controls",
        "gate_c3_controls",
    }:
        raise ProvenanceError("Gate C initialization is only a formal prerequisite. Fix the input condition named in the error, then rerun the operation.")
    if not isinstance(base_prerequisites, Mapping) or set(base_prerequisites) != {
        "gate_a",
        "gate_b",
    }:
        raise ProvenanceError("Gate C initialization base prerequisites are invalid. Set the named field to a value in the stated range, then rerun the operation.")
    paths = target_paths(config, head, "gate_c_init")
    for path in (paths.manifest, paths.preflight, paths.result):
        if not path.is_file() or not path.resolve().is_relative_to(config.repo_root):
            raise ProvenanceError("Required Gate C initialization bundle is incomplete. Fix the input condition named in the error, then rerun the operation.")

    manifest_sha256 = sha256_file(paths.manifest)
    manifest = load_strict_json(paths.manifest)
    if (
        set(manifest)
        != {
            "schema_version",
            "kind",
            "target",
            "source_head",
            "bundle_valid",
            "process_succeeded",
            "artifact_schema_verified",
            "scientific_qualification_passed",
            "preflight",
            "result",
            "gate_command",
            "postflight",
            "failure",
            "bundle_sha256",
        }
        or not _is_integer(manifest.get("schema_version"))
        or int(manifest["schema_version"]) != 1
        or manifest.get("kind") != "example21_authenticated_launch_manifest"
        or manifest.get("target") != "gate_c_init"
        or manifest.get("source_head") != head
        or manifest.get("bundle_valid") is not True
        or manifest.get("process_succeeded") is not True
        or manifest.get("artifact_schema_verified") is not True
        or manifest.get("scientific_qualification_passed") is not True
        or manifest.get("failure") is not None
    ):
        raise ProvenanceError("Required Gate C initialization manifest is invalid. Set the named field to a value in the stated range, then rerun the operation.")
    preflight_reference = _validate_artifact_reference(
        manifest.get("preflight"),
        paths.preflight,
        repo_root=config.repo_root,
        label="Gate C initialization preflight",
    )
    result_reference = _validate_artifact_reference(
        manifest.get("result"),
        paths.result,
        repo_root=config.repo_root,
        label="Gate C initialization result",
    )
    bundle_sha256 = _launch_bundle_sha256(
        "gate_c_init",
        head,
        preflight_reference["sha256"],
        result_reference["sha256"],
    )
    if manifest.get("bundle_sha256") != bundle_sha256:
        raise ProvenanceError("Required Gate C initialization bundle digest is invalid. Set the named field to a value in the stated range, then rerun the operation.")

    preflight = load_strict_json(paths.preflight)
    if (
        set(preflight)
        != {
            "schema_version",
            "kind",
            "target",
            "passed",
            "source",
            "image",
            "mounts",
            "container_source",
            "planned_gate",
            "admission_manifests",
            "gate_b_prerequisites",
            "gate_c_prerequisites",
        }
        or preflight.get("admission_manifests") is not None
        or preflight.get("gate_b_prerequisites") is not None
        or not _json_exact(
            preflight.get("gate_c_prerequisites"), base_prerequisites
        )
    ):
        raise ProvenanceError(
            "Gate C initialization preflight prerequisite binding is invalid. Set the named field to a value in the stated range, then rerun the operation."
        )
    _validate_preflight_semantics(
        preflight,
        target="gate_c_init",
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
        target="gate_c_init",
        head=head,
        image_id=image_id,
        admission_manifests=None,
        repo_root=config.repo_root,
        gate_c_prerequisites=base_prerequisites,
    )
    if not passed:
        raise ProvenanceError("Required Gate C initialization admission did not pass. Fix the input condition named in the error, then rerun the operation.")
    if (
        sha256_file(paths.manifest) != manifest_sha256
        or sha256_file(paths.preflight) != preflight_reference["sha256"]
        or sha256_file(paths.result) != result_reference["sha256"]
    ):
        raise ProvenanceError(
            "Gate C initialization bundle changed during authentication. Fix the input condition named in the error, then rerun the operation."
        )
    return {
        "target": "gate_c_init",
        "source_head": head,
        "image_digest": image_id,
        "bundle_sha256": bundle_sha256,
        "manifest_sha256": manifest_sha256,
        "preflight_sha256": preflight_reference["sha256"],
        "result_sha256": result_reference["sha256"],
        "admission": result,
    }


def _load_formal_gate_c_prerequisites(
    config: LaunchConfig,
    *,
    head: str,
    image_id: str,
) -> dict[str, Any]:
    """Authenticate the exact three prerequisites for formal Gate C."""

    if config.target != "formal_gate_c":
        raise ProvenanceError("Formal Gate C prerequisites require formal_gate_c. Provide the required value for Formal Gate C prerequisites.")
    base = _load_gate_c_prerequisites(config)
    initialization = _load_gate_c_init_manifest(
        config,
        head=head,
        image_id=image_id,
        base_prerequisites=base,
    )
    return {
        "gate_a": base["gate_a"],
        "gate_b": base["gate_b"],
        "gate_c_initialization": initialization,
    }


def _load_gate_c_controls_prerequisites(
    config: LaunchConfig,
    *,
    target: Literal["gate_c2_controls", "gate_c3_controls"],
    head: str,
    image_id: str,
) -> dict[str, Any]:
    if config.target != target:
        raise ProvenanceError(
            f"{target} prerequisites require target {target}. Provide the required value for {target} prerequisites."
        )
    base = _load_gate_c_prerequisites(config)
    initialization = _load_gate_c_init_manifest(
        config,
        head=head,
        image_id=image_id,
        base_prerequisites=base,
    )
    return {
        "gate_a": base["gate_a"],
        "gate_b": base["gate_b"],
        "gate_c_initialization": initialization,
    }


def _load_gate_c2_controls_prerequisites(
    config: LaunchConfig,
    *,
    head: str,
    image_id: str,
) -> dict[str, Any]:
    """Authenticate the exact inputs for the Gate C2 control admission."""

    return _load_gate_c_controls_prerequisites(
        config,
        target="gate_c2_controls",
        head=head,
        image_id=image_id,
    )


def _load_gate_c3_controls_prerequisites(
    config: LaunchConfig,
    *,
    head: str,
    image_id: str,
) -> dict[str, Any]:
    """Authenticate the exact inputs for the Gate C3 control admission."""

    return _load_gate_c_controls_prerequisites(
        config,
        target="gate_c3_controls",
        head=head,
        image_id=image_id,
    )


def _load_gate_c2_controls_manifest(
    config: LaunchConfig,
    *,
    head: str,
    image_id: str,
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the current-HEAD Gate C2 control admission bundle."""

    if config.target != "gate_c2_controls":
        raise ProvenanceError(
            "Gate C2 control admission requires gate_c2_controls. Provide the required value for Gate C2 control admission."
        )
    if not isinstance(prerequisites, Mapping) or set(prerequisites) != {
        "gate_a",
        "gate_b",
        "gate_c_initialization",
    }:
        raise ProvenanceError("Gate C2 control prerequisites are invalid. Set the named field to a value in the stated range, then rerun the operation.")
    paths = target_paths(config, head, "gate_c2_controls")
    for path in (paths.manifest, paths.preflight, paths.result):
        if not path.is_file() or not path.resolve().is_relative_to(config.repo_root):
            raise ProvenanceError("Required Gate C2 control bundle is incomplete. Fix the input condition named in the error, then rerun the operation.")

    manifest_sha256 = sha256_file(paths.manifest)
    manifest = load_strict_json(paths.manifest)
    if (
        set(manifest)
        != {
            "schema_version",
            "kind",
            "target",
            "source_head",
            "bundle_valid",
            "process_succeeded",
            "artifact_schema_verified",
            "scientific_qualification_passed",
            "preflight",
            "result",
            "gate_command",
            "postflight",
            "failure",
            "bundle_sha256",
        }
        or not _is_integer(manifest.get("schema_version"))
        or int(manifest["schema_version"]) != 1
        or manifest.get("kind") != "example21_authenticated_launch_manifest"
        or manifest.get("target") != "gate_c2_controls"
        or manifest.get("source_head") != head
        or manifest.get("bundle_valid") is not True
        or manifest.get("process_succeeded") is not True
        or manifest.get("artifact_schema_verified") is not True
        or manifest.get("scientific_qualification_passed") is not True
        or manifest.get("failure") is not None
    ):
        raise ProvenanceError("Required Gate C2 control manifest is invalid. Set the named field to a value in the stated range, then rerun the operation.")
    preflight_reference = _validate_artifact_reference(
        manifest.get("preflight"),
        paths.preflight,
        repo_root=config.repo_root,
        label="Gate C2 control preflight",
    )
    result_reference = _validate_artifact_reference(
        manifest.get("result"),
        paths.result,
        repo_root=config.repo_root,
        label="Gate C2 control result",
    )
    bundle_sha256 = _launch_bundle_sha256(
        "gate_c2_controls",
        head,
        preflight_reference["sha256"],
        result_reference["sha256"],
    )
    if manifest.get("bundle_sha256") != bundle_sha256:
        raise ProvenanceError("Required Gate C2 control bundle digest is invalid. Set the named field to a value in the stated range, then rerun the operation.")

    preflight = load_strict_json(paths.preflight)
    if (
        set(preflight)
        != {
            "schema_version",
            "kind",
            "target",
            "passed",
            "source",
            "image",
            "mounts",
            "container_source",
            "planned_gate",
            "admission_manifests",
            "gate_b_prerequisites",
            "gate_c_prerequisites",
        }
        or preflight.get("admission_manifests") is not None
        or preflight.get("gate_b_prerequisites") is not None
        or not _json_exact(preflight.get("gate_c_prerequisites"), prerequisites)
    ):
        raise ProvenanceError(
            "Gate C2 control preflight prerequisite binding is invalid. Set the named field to a value in the stated range, then rerun the operation."
        )
    _validate_preflight_semantics(
        preflight,
        target="gate_c2_controls",
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
        target="gate_c2_controls",
        head=head,
        image_id=image_id,
        admission_manifests=None,
        repo_root=config.repo_root,
        gate_c_prerequisites=prerequisites,
    )
    if not passed:
        raise ProvenanceError("Required Gate C2 control admission did not pass. Fix the input condition named in the error, then rerun the operation.")
    if (
        sha256_file(paths.manifest) != manifest_sha256
        or sha256_file(paths.preflight) != preflight_reference["sha256"]
        or sha256_file(paths.result) != result_reference["sha256"]
    ):
        raise ProvenanceError(
            "Gate C2 control bundle changed during authentication. Fix the input condition named in the error, then rerun the operation."
        )
    return {
        "target": "gate_c2_controls",
        "source_head": head,
        "image_digest": image_id,
        "bundle_sha256": bundle_sha256,
        "manifest_sha256": manifest_sha256,
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
            raise ProvenanceError(f"Required {target} admission manifest is missing. Provide the missing value or resource, then rerun the operation.")
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

    explicit = _container_environment(
        head,
        image_id,
        git_dir_in_container,
        target=config.target,
    )
    command = _docker_base(
        config,
        image_id=image_id,
        common_git_dir=(config.repo_root / ".git")
        if common_git_dir is None
        else common_git_dir,
        container_environment=explicit,
        gpu=True,
    )
    if config.target in _GATE_B_TARGETS:
        module = _DEPTH_GATE_MODULE
    elif config.target in _GATE_C_TARGETS:
        module = _ABLATION_GATE_MODULE
    else:
        module = _GATE_MODULE
    command.extend(["python", "-m", module])
    if config.target in _GATE_B_TARGETS:
        if admission_manifests is not None:
            raise ProvenanceError("Gate B targets do not accept Gate A admissions. Fix the input condition named in the error, then rerun the operation.")
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
    elif config.target in _GATE_C_TARGETS:
        if admission_manifests is not None:
            raise ProvenanceError("Gate C targets do not accept Gate A admissions. Fix the input condition named in the error, then rerun the operation.")
        gate_a = _gate_a_artifact_paths(config)
        gate_b = _formal_gate_b_artifact_paths(config)
        command.extend(
            [
                "--target",
                config.target,
                "--gate-a-result",
                str(gate_a.container_result),
                "--gate-a-manifest",
                str(_container_path(config, gate_a.manifest)),
                "--gate-b-manifest",
                str(_container_path(config, gate_b.manifest)),
            ]
        )
        if config.target in {
            "formal_gate_c",
            "gate_c2_controls",
            "gate_c3_controls",
        }:
            init_manifest = target_paths(config, head, "gate_c_init").manifest
            command.extend(
                [
                    "--gate-c-init-manifest",
                    str(_container_path(config, init_manifest)),
                ]
            )
    elif config.target == "formal_gate_a":
        if set(admission_manifests or {}) != {"one_update", "stability_256"}:
            raise ProvenanceError("Formal Gate A requires both admission manifests. Provide the required value for Formal Gate A.")
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
        raise ProvenanceError("Source is dirty before launch. Fix the input condition named in the error, then rerun the operation.")
    for path in (paths.result, paths.preflight, paths.manifest):
        if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
            raise ProvenanceError(f"Artifact already exists: {path}. Fix the input condition named in the error, then rerun the operation.")
    ignore = _run(
        command_runner,
        ["git", "check-ignore", "--quiet", str(paths.result)],
        cwd=config.repo_root,
        environment=_sanitized_host_environment(),
    )
    if ignore.returncode != 0:
        raise ProvenanceError("Output path is not ignored by Git. Fix the input condition named in the error, then rerun the operation.")

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
    gate_c_prerequisites: dict[str, Any] | None = None
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
            raise ProvenanceError("Container source preflight disagrees with clean HEAD. Use matching values and structures.")
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
        if config.target == "gate_c_init":
            gate_c_prerequisites = _load_gate_c_prerequisites(config)
        elif config.target == "formal_gate_c":
            gate_c_prerequisites = _load_formal_gate_c_prerequisites(
                config,
                head=source_start.head,
                image_id=image.image_id,
            )
        elif config.target == "gate_c2_controls":
            gate_c_prerequisites = _load_gate_c2_controls_prerequisites(
                config,
                head=source_start.head,
                image_id=image.image_id,
            )
        elif config.target == "gate_c3_controls":
            gate_c_prerequisites = _load_gate_c3_controls_prerequisites(
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
                    source_start.head,
                    image.image_id,
                    git_dir_in_container,
                    target=config.target,
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
            gate_c_prerequisites=(
                gate_c_prerequisites
                if config.target in _GATE_C_TARGETS
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
            raise ProvenanceError("Gate command did not create its declared result. Fix the input condition named in the error, then rerun the operation.")
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
            gate_c_prerequisites=gate_c_prerequisites,
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
                raise ProvenanceError("Source changed or became dirty during launch. Fix the input condition named in the error, then rerun the operation.")
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
                    "Preflight or result changed before manifest signing. Fix the input condition named in the error, then rerun the operation."
                )
            if config.target in _GATE_B_TARGETS:
                reloaded_gate_a = _load_gate_a_prerequisite(config)
                if not _json_exact(reloaded_gate_a, gate_a_prerequisite):
                    raise ProvenanceError(
                        "Gate A prerequisite changed before signing. Fix the input condition named in the error, then rerun the operation."
                    )
            if config.target == "formal_gate_b":
                reloaded_init = _load_gate_b_init_manifest(
                    config,
                    head=source_start.head,
                    image_id=image.image_id,
                )
                if not _json_exact(reloaded_init, gate_b_init_bundle):
                    raise ProvenanceError(
                        "Gate B initialization prerequisite changed before signing. Fix the input condition named in the error, then rerun the operation."
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
                gate_c_prerequisites=gate_c_prerequisites,
            )
            if rechecked_scientific is not scientific_passed:
                raise ProvenanceError(
                    "Scientific qualification changed before manifest signing. Fix the input condition named in the error, then rerun the operation."
                )
            if config.target == "gate_c_init":
                reloaded_gate_c = _load_gate_c_prerequisites(config)
                gate_c_names = ("gate_a", "gate_b")
            elif config.target == "formal_gate_c":
                reloaded_gate_c = _load_formal_gate_c_prerequisites(
                    config,
                    head=source_start.head,
                    image_id=image.image_id,
                )
                gate_c_names = ("gate_a", "gate_b", "gate_c_initialization")
            elif config.target == "gate_c2_controls":
                reloaded_gate_c = _load_gate_c2_controls_prerequisites(
                    config,
                    head=source_start.head,
                    image_id=image.image_id,
                )
                gate_c_names = ("gate_a", "gate_b", "gate_c_initialization")
            elif config.target == "gate_c3_controls":
                reloaded_gate_c = _load_gate_c3_controls_prerequisites(
                    config,
                    head=source_start.head,
                    image_id=image.image_id,
                )
                gate_c_names = ("gate_a", "gate_b", "gate_c_initialization")
            else:
                reloaded_gate_c = None
                gate_c_names = ()
            for name in gate_c_names:
                if reloaded_gate_c is None:
                    raise ProvenanceError("Gate C prerequisites were not reloaded. Fix the input condition named in the error, then rerun the operation.")
                if not _json_exact(
                    reloaded_gate_c.get(name),
                    (gate_c_prerequisites or {}).get(name),
                ):
                    raise ProvenanceError(
                        f"{name} prerequisite changed before signing. Fix the input condition named in the error, then rerun the operation."
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
        Zero for a valid passing bundle, two for a provenance failure, and three
        for a valid bundle that fails its scientific admission.
    """

    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    output = args.output_dir
    if output is None:
        if args.target in _GATE_B_TARGETS:
            output = Path("var/example21-depth-gate")
        elif args.target in _GATE_C_TARGETS:
            output = _GATE_C_DIRECTORY
        else:
            output = Path("var/example21-binding-gate")
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
        in {
            "one_update",
            "stability_256",
            "gate_b_init",
            "formal_gate_b",
            "gate_c_init",
            "formal_gate_c",
            "gate_c2_controls",
            "gate_c3_controls",
        }
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
