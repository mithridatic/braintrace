"""Tests for the immutable-image Example 21 Gate launcher."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pytest

from examples.pp_prop import latent_workspace_binding_gate as gate
from examples.pp_prop import latent_workspace_binding_gate_test as gate_fixtures
from examples.pp_prop import latent_workspace_binding_gate_launcher as launcher


_HEAD = "a" * 40
_IMAGE_ID = "sha256:" + "b" * 64


def _source() -> dict[str, object]:
    return {
        "commit": _HEAD,
        "asserted_commit": _HEAD,
        "asserted_commit_matches_head": True,
        "commit_is_valid_40_hex": True,
        "head_command_succeeded": True,
        "dirty": False,
        "asserted_dirty": False,
        "asserted_dirty_matches_worktree": True,
        "status_command_succeeded": True,
        "verified": True,
    }


def _strict_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, sort_keys=True), encoding="utf-8"
    )


def _result(target: str, manifests: dict[str, Any] | None = None) -> dict[str, Any]:
    fixed = (
        gate.BindingGateConfig.stage21_stability_config()
        if target == "stability_256"
        else gate.BindingGateConfig.stage21_one_update_config()
    )
    fixed_config: dict[str, Any] = {
        **dataclasses.asdict(fixed),
        "configuration_scale": fixed.configuration_scale,
    }
    if target == "formal_gate_a":
        model = gate._model_config(fixed, batch_size=fixed.batch_size)
        formal_config = {
            **fixed_config,
            "qualification_regime": fixed.qualification_regime,
            "configuration_sha256": gate._configuration_digest(fixed, model),
            "model": gate.legacy._json_ready(dataclasses.asdict(model)),
        }
        initialization = {
            "fresh_model": True,
            "model_seed": 2108,
            "parameter_sha256": gate.PREREGISTERED_GPU_INITIAL_PARAMETER_SHA256,
            "parameter_count": gate.PREREGISTERED_PARAMETER_COUNT,
        }
        data = {
            "training_schedule_sha256": gate.PREREGISTERED_TRAINING_SCHEDULE_SHA256,
            "validation_schedule_sha256": gate.PREREGISTERED_STABILITY_DIGESTS[
                "validation_schedule_sha256"
            ],
            "marginals": {"exact_marginal_equality": True},
        }
        environment = {
            "image_digest": _IMAGE_ID,
            "backend": "gpu",
            "devices": [{"id": 0, "platform": "gpu", "device_kind": "test GPU"}],
        }
        admissions = copy.deepcopy(manifests or {})
        architecture = gate_fixtures._passing_architecture()
        compiler = gate_fixtures._passing_compiler()
        training = {
            **gate_fixtures._passing_training(),
            "compiler": compiler,
        }
        evaluation = gate_fixtures._passing_evaluation()
        diagnostics = gate_fixtures._passing_diagnostics()
        gate_native = gate_fixtures._passing_separation(architecture)
        standard_arc = gate_fixtures._passing_separation(
            architecture, portable=True
        )
        qualification = gate._qualification_report(
            evaluation=evaluation,
            diagnostics=diagnostics,
            compiler=compiler,
            training=training,
            environment=environment,
            architecture=architecture,
            gate_native_separation=gate_native,
            standard_arc_separation=standard_arc,
            marginals=data["marginals"],
            source_start=_source(),
            source_end=_source(),
            one_update_admission=admissions.get("one_update", {}).get(
                "admission", {}
            ),
            stability_admission=admissions.get("stability_256", {}).get(
                "admission", {}
            ),
            initialization=initialization,
            data=data,
            config=fixed,
        )
        return {
            "schema_version": 3,
            "control": "example21_associative_workspace_binding_gate_a",
            "learner": "pp_prop_only",
            "config": formal_config,
            "stage21_admissions": admissions,
            "source": _source(),
            "source_end": _source(),
            "environment": environment,
            "initialization": initialization,
            "data": data,
            "training": training,
            "evaluation": evaluation,
            "diagnostics": diagnostics,
            "architecture": architecture,
            "gate_native_key_separation": gate_native,
            "standard_arc_key_separation": standard_arc,
            "qualification": qualification,
            "interpretation": qualification["interpretation"],
        }
    control = (
        "example21_stage21_one_update_admission"
        if target == "one_update"
        else "example21_stage21_stability_256_admission"
    )
    admission = (
        gate_fixtures._passing_one_update_admission()
        if target == "one_update"
        else gate_fixtures._passing_stability_admission()
    )
    qualification = (
        gate._one_update_admission_qualification(admission)
        if target == "one_update"
        else gate._stability_admission_qualification(admission)
    )
    admission["qualification"] = qualification
    return {
        "schema_version": 3,
        "control": control,
        "target": target,
        "learner": "pp_prop_only",
        "config": {
            **fixed_config,
            **(
                {"qualification_regime": fixed.qualification_regime}
                if target == "stability_256"
                else {}
            ),
        },
        "source": _source(),
        "source_end": _source(),
        "environment": {
            "image_digest": _IMAGE_ID,
            "backend": "gpu",
            "devices": [{"id": 0, "platform": "gpu", "device_kind": "test GPU"}],
        },
        "admission": admission,
        "qualification": qualification,
        "interpretation": qualification["interpretation"],
    }


class FakeRunner:
    def __init__(self, repo: Path, target: str) -> None:
        self.repo = repo
        self.target = target
        self.calls: list[list[str]] = []
        self.sidecar_seen_before_gate = False

    def __call__(
        self, command: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        argv = list(command)
        self.calls.append(argv)
        stdout = ""
        if argv[:3] == ["git", "rev-parse", "--show-toplevel"]:
            stdout = str(self.repo) + "\n"
        elif argv[:3] == ["git", "rev-parse", "--path-format=absolute"]:
            if argv[-1] == "--git-common-dir":
                stdout = str(self.repo / ".git-common") + "\n"
        elif argv[:3] == ["git", "rev-parse", "--absolute-git-dir"]:
            stdout = str(self.repo / ".git-common" / "worktrees" / "gate") + "\n"
        elif argv[:3] == ["git", "rev-parse", "HEAD"]:
            stdout = _HEAD + "\n"
        elif argv[:3] == ["git", "status", "--porcelain=v1"]:
            stdout = ""
        elif argv[:3] == ["git", "check-ignore", "--quiet"]:
            stdout = ""
        elif argv[:3] == ["docker", "image", "inspect"]:
            stdout = json.dumps(
                [
                    {
                        "Id": _IMAGE_ID,
                        "Config": {
                            "Labels": {"org.opencontainers.image.revision": _HEAD}
                        },
                    }
                ]
            )
        elif argv and argv[0] == "docker" and "python" in argv:
            result_arg = argv[argv.index("--output") + 1]
            relative = result_arg.removeprefix("/work/")
            destination = self.repo / Path(relative)
            sidecar = destination.with_suffix(".preflight.json")
            self.sidecar_seen_before_gate = sidecar.is_file()
            manifests: dict[str, Any] = {}
            for name, flag in (
                ("one_update", "--one-update-manifest"),
                ("stability_256", "--stability-manifest"),
            ):
                if flag in argv:
                    manifest_path = self.repo / Path(
                        argv[argv.index(flag) + 1].removeprefix("/work/")
                    )
                    bundle = launcher.load_authenticated_admission(
                        manifest_path,
                        target=name,
                        head=_HEAD,
                        image_id=_IMAGE_ID,
                        repo_root=self.repo,
                    )
                    manifests[name] = {
                        "target": name,
                        "source_head": _HEAD,
                        "image_digest": _IMAGE_ID,
                        "bundle_sha256": bundle["manifest"]["bundle_sha256"],
                        "manifest_sha256": bundle["manifest_sha256"],
                        "preflight_sha256": bundle["preflight_sha256"],
                        "result_sha256": bundle["result_sha256"],
                        "admission": bundle["admission"],
                    }
            _strict_write(destination, _result(self.target, manifests))
        elif argv and argv[0] == "docker":
            if argv[-2:] == ["git", "--version"]:
                stdout = "git version 2.50.1\n"
            elif argv[-3:] == ["git", "rev-parse", "HEAD"]:
                stdout = _HEAD + "\n"
            elif "status" in argv:
                stdout = ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")


def _config(tmp_path: Path, target: str) -> launcher.LaunchConfig:
    repo = tmp_path / "repo with spaces"
    (repo / ".git-common" / "worktrees" / "gate").mkdir(parents=True)
    return launcher.LaunchConfig(
        target=target,
        repo_root=repo,
        output_dir=repo / "var" / "example21-binding-gate",
    )


def _refresh_bundle_manifest(
    config: launcher.LaunchConfig, target: str
) -> Path:
    paths = launcher.target_paths(config, _HEAD, target)
    manifest = launcher.load_strict_json(paths.manifest)
    references = {
        "preflight": paths.preflight,
        "result": paths.result,
    }
    for label, path in references.items():
        manifest[label]["sha256"] = launcher.sha256_file(path)
        manifest[label]["size_bytes"] = path.stat().st_size
    manifest["bundle_sha256"] = hashlib.sha256(
        (
            f"example21-launch-bundle-v1\0{target}\0{_HEAD}\0"
            f"{manifest['preflight']['sha256']}\0{manifest['result']['sha256']}"
        ).encode("utf-8")
    ).hexdigest()
    launcher.write_strict_json(paths.manifest, manifest)
    return paths.manifest


def _authenticated_formal_inputs(
    tmp_path: Path,
) -> tuple[launcher.LaunchConfig, dict[str, Path], dict[str, Any]]:
    config = _config(tmp_path, "formal_gate_a")
    manifests: dict[str, Path] = {}
    evidence: dict[str, Any] = {}
    for name in ("one_update", "stability_256"):
        admission_config = launcher.LaunchConfig(
            target=name,
            repo_root=config.repo_root,
            output_dir=config.output_dir,
        )
        launcher.launch(
            admission_config,
            command_runner=FakeRunner(config.repo_root, name),
        )
        path = launcher.target_paths(config, _HEAD, name).manifest
        manifests[name] = path
        bundle = launcher.load_authenticated_admission(
            path,
            target=name,
            head=_HEAD,
            image_id=_IMAGE_ID,
            repo_root=config.repo_root,
        )
        evidence[name] = {
            "target": name,
            "source_head": _HEAD,
            "image_digest": _IMAGE_ID,
            "bundle_sha256": bundle["manifest"]["bundle_sha256"],
            "manifest_sha256": bundle["manifest_sha256"],
            "preflight_sha256": bundle["preflight_sha256"],
            "result_sha256": bundle["result_sha256"],
            "admission": bundle["admission"],
        }
    return config, manifests, evidence


@pytest.mark.parametrize("target", ["one_update", "stability_256"])
def test_launch_writes_preflight_before_immutable_image_gate_and_binds_result(
    tmp_path: Path, target: str
) -> None:
    config = _config(tmp_path, target)
    runner = FakeRunner(config.repo_root, target)

    manifest_path = launcher.launch(config, command_runner=runner)

    assert runner.sidecar_seen_before_gate is True
    manifest = launcher.load_strict_json(manifest_path)
    assert manifest["target"] == target
    assert manifest["bundle_valid"] is True
    assert manifest["process_succeeded"] is True
    assert manifest["scientific_qualification_passed"] is True
    assert manifest["preflight"]["sha256"] == launcher.sha256_file(
        Path(manifest["preflight"]["host_path"])
    )
    assert manifest["result"]["sha256"] == launcher.sha256_file(
        Path(manifest["result"]["host_path"])
    )
    gate_call = next(call for call in runner.calls if "python" in call)
    assert "--pull=never" in gate_call
    assert "--gpus" in gate_call
    assert _IMAGE_ID in gate_call
    assert "braintrace-gpu:0.11.0-py314" not in gate_call
    assert any("dst=/git-common,readonly" in item for item in gate_call)
    assert any(str(config.repo_root) in item for item in gate_call)


def test_formal_target_authenticates_and_binds_both_admission_manifests(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "formal_gate_a")
    for target in ("one_update", "stability_256"):
        runner = FakeRunner(config.repo_root, target)
        launcher.launch(
            launcher.LaunchConfig(
                target=target,
                repo_root=config.repo_root,
                output_dir=config.output_dir,
            ),
            command_runner=runner,
        )
    runner = FakeRunner(config.repo_root, "formal_gate_a")

    manifest_path = launcher.launch(config, command_runner=runner)

    manifest = launcher.load_strict_json(manifest_path)
    assert manifest["bundle_valid"] is True
    assert manifest["process_succeeded"] is True
    assert manifest["scientific_qualification_passed"] is True
    gate_call = next(call for call in runner.calls if "python" in call)
    assert "--one-update-manifest" in gate_call
    assert "--stability-manifest" in gate_call
    result = launcher.load_strict_json(Path(manifest["result"]["host_path"]))
    for target in ("one_update", "stability_256"):
        admission_manifest = launcher.target_paths(config, _HEAD, target).manifest
        expected = launcher.sha256_file(admission_manifest)
        assert result["stage21_admissions"][target]["manifest_sha256"] == expected


def test_revision_mismatch_stops_before_gate_and_retains_failed_preflight(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "one_update")
    base = FakeRunner(config.repo_root, "one_update")

    def mismatch(command: Sequence[str], **kwargs: object):
        result = base(command, **kwargs)
        if list(command)[:3] == ["docker", "image", "inspect"]:
            wrong = json.loads(result.stdout)
            wrong[0]["Config"]["Labels"]["org.opencontainers.image.revision"] = (
                "c" * 40
            )
            result = subprocess.CompletedProcess(
                list(command), 0, stdout=json.dumps(wrong), stderr=""
            )
        return result

    with pytest.raises(launcher.ProvenanceError, match="OCI revision"):
        launcher.launch(config, command_runner=mismatch)

    paths = launcher.target_paths(config, _HEAD, "one_update")
    assert paths.preflight.is_file()
    assert launcher.load_strict_json(paths.preflight)["passed"] is False
    assert not any("python" in call for call in base.calls)


def test_dirty_start_or_end_fails_closed(tmp_path: Path) -> None:
    config = _config(tmp_path, "one_update")
    base = FakeRunner(config.repo_root, "one_update")
    status_calls = 0

    def dirty(command: Sequence[str], **kwargs: object):
        nonlocal status_calls
        result = base(command, **kwargs)
        if list(command)[:3] == ["git", "status", "--porcelain=v1"]:
            status_calls += 1
            if status_calls >= 2:
                return subprocess.CompletedProcess(
                    list(command), 0, stdout=" M model.py\n", stderr=""
                )
        return result

    with pytest.raises(launcher.ProvenanceError, match="source changed"):
        launcher.launch(config, command_runner=dirty)

    paths = launcher.target_paths(config, _HEAD, "one_update")
    manifest = launcher.load_strict_json(paths.manifest)
    assert manifest["bundle_valid"] is False
    assert manifest["postflight"]["clean"] is False


def test_missing_or_tampered_admission_manifest_blocks_formal_target(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "formal_gate_a")
    with pytest.raises(launcher.ProvenanceError, match="admission manifest"):
        launcher.launch(config, command_runner=FakeRunner(config.repo_root, "formal_gate_a"))


def test_existing_artifacts_are_never_overwritten(tmp_path: Path) -> None:
    config = _config(tmp_path, "one_update")
    paths = launcher.target_paths(config, _HEAD, "one_update")
    paths.result.parent.mkdir(parents=True, exist_ok=True)
    paths.result.write_text("owned", encoding="utf-8")

    with pytest.raises(launcher.ProvenanceError, match="already exists"):
        launcher.launch(config, command_runner=FakeRunner(config.repo_root, "one_update"))

    assert paths.result.read_text(encoding="utf-8") == "owned"


def test_strict_json_and_atomic_writer_reject_nonfinite(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"
    with pytest.raises(ValueError):
        launcher.write_strict_json(destination, {"bad": float("nan")})
    assert not destination.exists()
    assert not destination.with_suffix(".json.tmp").exists()

    destination.write_text('{"bad": NaN}', encoding="utf-8")
    with pytest.raises(ValueError):
        launcher.load_strict_json(destination)


def test_target_commands_are_fixed_without_topology_or_budget_knobs(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "one_update")
    paths = launcher.target_paths(config, _HEAD, "one_update")
    command = launcher.gate_command(
        config,
        image_id=_IMAGE_ID,
        head=_HEAD,
        paths=paths,
        git_dir_in_container="/git-common/worktrees/gate",
        admission_manifests=None,
    )

    for forbidden in (
        "--training-updates",
        "--batch-size",
        "--neuron-count",
        "--recurrent-edges",
        "--context-memory-width",
        "--smoke",
    ):
        assert forbidden not in command
    assert command[-4:] == ["--target", "one_update", "--output", str(paths.container_result)]


@pytest.mark.parametrize(
    "target", ["one_update", "stability_256", "formal_gate_a"]
)
def test_gate_commands_use_importable_package_module_boundary(
    tmp_path: Path, target: str
) -> None:
    config = _config(tmp_path, target)
    paths = launcher.target_paths(config, _HEAD, target)
    admission_manifests = None
    if target == "formal_gate_a":
        admission_manifests = {
            name: launcher.target_paths(config, _HEAD, name).manifest
            for name in ("one_update", "stability_256")
        }

    command = launcher.gate_command(
        config,
        image_id=_IMAGE_ID,
        head=_HEAD,
        paths=paths,
        git_dir_in_container="/git-common/worktrees/gate",
        admission_manifests=admission_manifests,
    )
    python_index = command.index("python")

    assert command[python_index : python_index + 3] == [
        "python",
        "-m",
        "examples.pp_prop.latent_workspace_binding_gate",
    ]


def test_authenticated_preflight_revalidates_package_module_invocation(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "one_update")
    manifest = launcher.launch(
        config, command_runner=FakeRunner(config.repo_root, "one_update")
    )
    preflight = launcher.load_strict_json(
        launcher.target_paths(config, _HEAD, "one_update").preflight
    )
    command = preflight["planned_gate"]["argv"]
    python_index = command.index("python")
    assert command[python_index : python_index + 3] == [
        "python",
        "-m",
        "examples.pp_prop.latent_workspace_binding_gate",
    ]
    assert launcher.load_authenticated_admission(
        manifest,
        target="one_update",
        head=_HEAD,
        image_id=_IMAGE_ID,
        repo_root=config.repo_root,
    )["admission"]["target"] == "one_update"


@pytest.mark.parametrize(
    "mutation",
    [
        "worktree_mount",
        "common_git_mount",
        "common_git_read_only",
        "container_git_environment",
        "planned_git_environment",
        "planned_argv_pull_policy",
        "planned_argv_output",
        "source_root",
    ],
)
def test_rehashed_semantic_preflight_tampering_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    config = _config(tmp_path, "one_update")
    launcher.launch(config, command_runner=FakeRunner(config.repo_root, "one_update"))
    paths = launcher.target_paths(config, _HEAD, "one_update")
    preflight = launcher.load_strict_json(paths.preflight)

    if mutation == "worktree_mount":
        preflight["mounts"]["worktree"]["target"] = "/substitute"
    elif mutation == "common_git_mount":
        preflight["mounts"]["common_git"]["target"] = "/substitute"
    elif mutation == "common_git_read_only":
        preflight["mounts"]["common_git"]["read_only"] = False
    elif mutation == "container_git_environment":
        preflight["container_source"]["environment"]["GIT_OPTIONAL_LOCKS"] = "1"
    elif mutation == "planned_git_environment":
        preflight["planned_gate"]["environment"]["GIT_WORK_TREE"] = "/substitute"
    elif mutation == "planned_argv_pull_policy":
        preflight["planned_gate"]["argv"].remove("--pull=never")
    elif mutation == "planned_argv_output":
        preflight["planned_gate"]["argv"][-1] = "/work/substitute.json"
    elif mutation == "source_root":
        preflight["source"]["root"] = str(config.repo_root / "substitute")
    launcher.write_strict_json(paths.preflight, preflight)
    _refresh_bundle_manifest(config, "one_update")

    with pytest.raises(launcher.ProvenanceError, match="preflight|planned Gate"):
        launcher.load_authenticated_admission(
            paths.manifest,
            target="one_update",
            head=_HEAD,
            image_id=_IMAGE_ID,
            repo_root=config.repo_root,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("returncode", False), ("wall_seconds", True)],
)
def test_command_record_rejects_boolean_numeric_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    stdout = str(tmp_path) + "\n"
    record = {
        "argv": ["git", "rev-parse", "--show-toplevel"],
        "cwd": str(tmp_path),
        "environment": {"git_override_variables_removed": "true"},
        "returncode": 0,
        "stdout": stdout,
        "stderr": "",
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(b"").hexdigest(),
        "wall_seconds": 0.01,
    }
    validation = {
        "argv_tail": ("git", "rev-parse", "--show-toplevel"),
        "environment": {"git_override_variables_removed": "true"},
        "cwd": str(tmp_path),
    }
    launcher._validated_command_record(copy.deepcopy(record), **validation)
    record[field] = value

    with pytest.raises(launcher.ProvenanceError):
        launcher._validated_command_record(record, **validation)


@pytest.mark.parametrize("artifact", ["preflight", "manifest"])
def test_authenticated_bundle_rejects_boolean_schema_versions(
    tmp_path: Path, artifact: str
) -> None:
    config = _config(tmp_path, "one_update")
    launcher.launch(config, command_runner=FakeRunner(config.repo_root, "one_update"))
    paths = launcher.target_paths(config, _HEAD, "one_update")
    path = paths.preflight if artifact == "preflight" else paths.manifest
    value = launcher.load_strict_json(path)
    value["schema_version"] = True
    launcher.write_strict_json(path, value)
    if artifact == "preflight":
        _refresh_bundle_manifest(config, "one_update")

    with pytest.raises(launcher.ProvenanceError):
        launcher.load_authenticated_admission(
            paths.manifest,
            target="one_update",
            head=_HEAD,
            image_id=_IMAGE_ID,
            repo_root=config.repo_root,
        )


def test_authenticated_bundle_rejects_internally_consistent_alternate_root(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "one_update")
    launcher.launch(config, command_runner=FakeRunner(config.repo_root, "one_update"))
    paths = launcher.target_paths(config, _HEAD, "one_update")
    alternate = str(config.repo_root.parent / "colluding alternate root")
    original = str(config.repo_root)

    def replace_worktree_mount(argv: list[str]) -> list[str]:
        return [
            item.replace(
                f"type=bind,src={original},dst=/work",
                f"type=bind,src={alternate},dst=/work",
            )
            for item in argv
        ]

    def rewrite_record_cwd(record: dict[str, Any]) -> None:
        record["cwd"] = alternate
        record["argv"] = replace_worktree_mount(record["argv"])

    preflight = launcher.load_strict_json(paths.preflight)
    preflight["source"]["root"] = alternate
    preflight["mounts"]["worktree"]["source"] = alternate
    for record in preflight["source"]["commands"]:
        rewrite_record_cwd(record)
    root_record = preflight["source"]["commands"][0]
    root_record["stdout"] = alternate + "\n"
    root_record["stdout_sha256"] = hashlib.sha256(
        root_record["stdout"].encode("utf-8")
    ).hexdigest()
    rewrite_record_cwd(preflight["image"]["inspect_command"])
    for record in preflight["container_source"]["commands"]:
        rewrite_record_cwd(record)
    preflight["planned_gate"]["argv"] = replace_worktree_mount(
        preflight["planned_gate"]["argv"]
    )
    launcher.write_strict_json(paths.preflight, preflight)

    manifest = launcher.load_strict_json(paths.manifest)
    rewrite_record_cwd(manifest["gate_command"])
    for record in manifest["postflight"]["host_commands"]:
        rewrite_record_cwd(record)
    postflight_root = manifest["postflight"]["host_commands"][0]
    postflight_root["stdout"] = alternate + "\n"
    postflight_root["stdout_sha256"] = hashlib.sha256(
        postflight_root["stdout"].encode("utf-8")
    ).hexdigest()
    for record in manifest["postflight"]["container_commands"]:
        rewrite_record_cwd(record)
    launcher.write_strict_json(paths.manifest, manifest)
    _refresh_bundle_manifest(config, "one_update")

    with pytest.raises(launcher.ProvenanceError, match="root|worktree|preflight"):
        launcher.load_authenticated_admission(
            paths.manifest,
            target="one_update",
            head=_HEAD,
            image_id=_IMAGE_ID,
            repo_root=config.repo_root,
        )


def test_authenticated_bundle_digest_is_recomputed_not_trusted(tmp_path: Path) -> None:
    config = _config(tmp_path, "one_update")
    launcher.launch(config, command_runner=FakeRunner(config.repo_root, "one_update"))
    paths = launcher.target_paths(config, _HEAD, "one_update")
    manifest = launcher.load_strict_json(paths.manifest)
    manifest["bundle_sha256"] = "0" * 64
    launcher.write_strict_json(paths.manifest, manifest)

    with pytest.raises(launcher.ProvenanceError, match="bundle"):
        launcher.load_authenticated_admission(
            paths.manifest,
            target="one_update",
            head=_HEAD,
            image_id=_IMAGE_ID,
            repo_root=config.repo_root,
        )


@pytest.mark.parametrize("artifact", ["preflight", "result"])
def test_launch_detects_artifact_toctou_before_signing_manifest(
    tmp_path: Path, artifact: str
) -> None:
    config = _config(tmp_path, "one_update")

    class MutatingRunner(FakeRunner):
        def __init__(self) -> None:
            super().__init__(config.repo_root, "one_update")
            self.gate_completed = False
            self.mutated = False

        def __call__(
            self, command: Sequence[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            argv = list(command)
            if (
                artifact == "result"
                and self.gate_completed
                and not self.mutated
                and argv[:3] == ["git", "rev-parse", "--show-toplevel"]
            ):
                path = launcher.target_paths(config, _HEAD).result
                value = launcher.load_strict_json(path)
                value["tampered_after_validation"] = True
                launcher.write_strict_json(path, value)
                self.mutated = True
            completed = super().__call__(command, **kwargs)
            if argv and argv[0] == "docker" and "python" in argv:
                self.gate_completed = True
                if artifact == "preflight":
                    path = launcher.target_paths(config, _HEAD).preflight
                    value = launcher.load_strict_json(path)
                    value["tampered_during_gate"] = True
                    launcher.write_strict_json(path, value)
                    self.mutated = True
            return completed

    with pytest.raises(launcher.ProvenanceError, match="artifact|bundle|changed"):
        launcher.launch(config, command_runner=MutatingRunner())

    manifest = launcher.load_strict_json(launcher.target_paths(config, _HEAD).manifest)
    assert manifest["bundle_valid"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "target",
        "config",
        "source_start",
        "source_end",
        "environment",
        "qualification",
    ],
)
def test_target_result_validation_rejects_each_authenticated_boundary(
    tmp_path: Path, mutation: str,
) -> None:
    result = _result("one_update")
    if mutation == "schema":
        result["schema_version"] = 2
    elif mutation == "target":
        result["target"] = "stability_256"
    elif mutation == "config":
        result["config"]["neuron_count"] = 64
    elif mutation == "source_start":
        result["source"]["dirty"] = True
    elif mutation == "source_end":
        result["source_end"]["verified"] = False
    elif mutation == "environment":
        result["environment"]["image_digest"] = "sha256:" + "c" * 64
    elif mutation == "qualification":
        result["qualification"]["passed"] = "true"

    with pytest.raises(launcher.ProvenanceError):
        launcher._validate_target_result(
            result,
            target="one_update",
            head=_HEAD,
            image_id=_IMAGE_ID,
            admission_manifests=None,
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    "field", ["gap_steps", "gradient_chunk_size", "memory_decay"]
)
def test_admission_result_rejects_json_boolean_numeric_config(
    tmp_path: Path, field: str
) -> None:
    result = _result("one_update")
    assert launcher._validate_target_result(
        copy.deepcopy(result),
        target="one_update",
        head=_HEAD,
        image_id=_IMAGE_ID,
        admission_manifests=None,
        repo_root=tmp_path,
    ) is True
    result["config"][field] = True
    result["admission"]["config"][field] = True

    with pytest.raises(launcher.ProvenanceError):
        launcher._validate_target_result(
            result,
            target="one_update",
            head=_HEAD,
            image_id=_IMAGE_ID,
            admission_manifests=None,
            repo_root=tmp_path,
        )


def test_admission_outer_config_boolean_cannot_match_numeric_inner_config(
    tmp_path: Path,
) -> None:
    result = _result("one_update")
    assert launcher._validate_target_result(
        copy.deepcopy(result),
        target="one_update",
        head=_HEAD,
        image_id=_IMAGE_ID,
        admission_manifests=None,
        repo_root=tmp_path,
    ) is True
    result["config"]["memory_decay"] = True

    with pytest.raises(launcher.ProvenanceError):
        launcher._validate_target_result(
            result,
            target="one_update",
            head=_HEAD,
            image_id=_IMAGE_ID,
            admission_manifests=None,
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    "field", ["gap_steps", "gradient_chunk_size", "memory_decay"]
)
def test_formal_fixed_config_rejects_json_boolean_numeric_config(
    field: str,
) -> None:
    result = _result("formal_gate_a")
    launcher._validate_fixed_config(result, "formal_gate_a")
    result["config"][field] = True

    with pytest.raises(launcher.ProvenanceError):
        launcher._validate_fixed_config(result, "formal_gate_a")


@pytest.mark.parametrize(
    "mutation",
    [
        "training_missing",
        "evaluation_fabricated",
        "learner",
        "interpretation",
        "embedded_admission_config_bool",
    ],
)
def test_formal_result_recomputes_scientific_qualification_and_identity(
    tmp_path: Path, mutation: str
) -> None:
    config, manifests, evidence = _authenticated_formal_inputs(tmp_path)
    result = _result("formal_gate_a", evidence)
    assert result["qualification"]["passed"] is True
    assert launcher._validate_target_result(
        copy.deepcopy(result),
        target="formal_gate_a",
        head=_HEAD,
        image_id=_IMAGE_ID,
        admission_manifests=manifests,
        repo_root=config.repo_root,
    ) is True

    if mutation == "training_missing":
        result["training"] = {}
    elif mutation == "evaluation_fabricated":
        result["evaluation"]["intact"]["accuracy"] = 0.0
    elif mutation == "learner":
        result["learner"] = "bptt"
    elif mutation == "interpretation":
        result["interpretation"] = "gate_a_failed_stop_no_capability_conclusion"
    elif mutation == "embedded_admission_config_bool":
        result["stage21_admissions"]["one_update"]["admission"]["config"][
            "memory_decay"
        ] = True

    with pytest.raises(launcher.ProvenanceError):
        launcher._validate_target_result(
            result,
            target="formal_gate_a",
            head=_HEAD,
            image_id=_IMAGE_ID,
            admission_manifests=manifests,
            repo_root=config.repo_root,
        )


@pytest.mark.parametrize("mutation", ["initialization", "training", "validation"])
def test_formal_result_binds_initialization_and_schedules_to_admissions(
    tmp_path: Path, mutation: str
) -> None:
    config, manifests, evidence = _authenticated_formal_inputs(tmp_path)
    result = _result("formal_gate_a", evidence)
    result["qualification"]["passed"] = True
    assert launcher._validate_target_result(
        copy.deepcopy(result),
        target="formal_gate_a",
        head=_HEAD,
        image_id=_IMAGE_ID,
        admission_manifests=manifests,
        repo_root=config.repo_root,
    ) is True
    if mutation == "initialization":
        result["initialization"]["parameter_sha256"] = "0" * 64
    elif mutation == "training":
        result["data"]["training_schedule_sha256"] = "0" * 64
    elif mutation == "validation":
        result["data"]["validation_schedule_sha256"] = "0" * 64

    with pytest.raises(launcher.ProvenanceError, match="initialization|schedule"):
        launcher._validate_target_result(
            result,
            target="formal_gate_a",
            head=_HEAD,
            image_id=_IMAGE_ID,
            admission_manifests=manifests,
            repo_root=config.repo_root,
        )


@pytest.mark.parametrize(
    "value",
    [
        '[1, 2, 3]',
        '{"duplicate": 1, "duplicate": 2}',
    ],
)
def test_strict_json_rejects_nonobject_or_duplicate_key(
    tmp_path: Path, value: str
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(value, encoding="utf-8")
    with pytest.raises(ValueError):
        launcher.load_strict_json(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target", "unknown"),
        ("output_dir", Path("outside")),
        ("output_dir", None),
        ("image", " "),
        ("cache_volume", "bad/name"),
    ],
)
def test_launch_config_rejects_unsafe_scope(
    tmp_path: Path, field: str, value: object
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    kwargs: dict[str, object] = {
        "target": "one_update",
        "repo_root": repo,
        "output_dir": repo / "var" / "example21-binding-gate",
    }
    if field == "output_dir" and value == Path("outside"):
        kwargs[field] = tmp_path / "outside"
    elif field == "output_dir" and value is None:
        kwargs[field] = repo
    else:
        kwargs[field] = value
    with pytest.raises(ValueError):
        launcher.LaunchConfig(**kwargs)
