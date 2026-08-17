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
from examples.pp_prop import latent_workspace_depth_gate as depth
from examples.pp_prop import latent_workspace_depth_gate_test as depth_fixtures


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


def _source_at(commit: str) -> dict[str, object]:
    value = _source()
    value["commit"] = commit
    value["asserted_commit"] = commit
    return value


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


class DepthFakeRunner(FakeRunner):
    def __init__(
        self,
        repo: Path,
        target: str,
        *,
        gate_a: dict[str, Any],
        gate_b_init: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(repo, target)
        self.gate_a = copy.deepcopy(gate_a)
        self.gate_b_init = copy.deepcopy(gate_b_init)

    def __call__(
        self, command: Sequence[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        argv = list(command)
        if argv and argv[0] == "docker" and "python" in argv:
            self.calls.append(argv)
            result_arg = argv[argv.index("--output") + 1]
            destination = self.repo / Path(result_arg.removeprefix("/work/"))
            self.sidecar_seen_before_gate = destination.with_suffix(
                ".preflight.json"
            ).is_file()
            passing = depth_fixtures._passing_depth_report()
            if self.target == "gate_b_init":
                result = copy.deepcopy(
                    passing["prerequisites"]["gate_b_initialization"]["admission"]
                )
                result["prerequisites"]["gate_a"] = copy.deepcopy(self.gate_a)
                result["qualification"] = depth._gate_b_initialization_qualification(
                    result,
                    depth.DepthGateConfig(),
                )
            elif self.target == "formal_gate_b":
                if self.gate_b_init is None:
                    raise AssertionError("formal Gate B fixture requires initialization")
                result = passing
                result["prerequisites"] = {
                    "gate_a": copy.deepcopy(self.gate_a),
                    "gate_b_initialization": copy.deepcopy(self.gate_b_init),
                }
                result["qualification"] = depth._qualification_report(
                    result,
                    config=depth.DepthGateConfig(),
                )
            else:
                raise AssertionError(f"unsupported depth fixture target {self.target}")
            launcher.write_strict_json(destination, result)
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return super().__call__(command, **kwargs)


def _config(tmp_path: Path, target: str) -> launcher.LaunchConfig:
    repo = tmp_path / "repo with spaces"
    (repo / ".git-common" / "worktrees" / "gate").mkdir(parents=True)
    return launcher.LaunchConfig(
        target=target,
        repo_root=repo,
        output_dir=repo / "var" / "example21-binding-gate",
    )


def _depth_config(tmp_path: Path, target: str) -> launcher.LaunchConfig:
    repo = tmp_path / "repo with spaces"
    (repo / ".git-common" / "worktrees" / "gate").mkdir(parents=True)
    return launcher.LaunchConfig(
        target=target,
        repo_root=repo,
        output_dir=repo / "var" / "example21-depth-gate",
    )


def _install_gate_a_prerequisite(
    config: launcher.LaunchConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    paths = launcher._gate_a_artifact_paths(config)
    _strict_write(paths.preflight, {"retained": "authenticated Gate A preflight"})
    result = {
        "schema_version": 3,
        "control": "example21_associative_workspace_binding_gate_a",
        "learner": "pp_prop_only",
        "source": _source_at(launcher._GATE_A_SOURCE_COMMIT),
        "source_end": _source_at(launcher._GATE_A_SOURCE_COMMIT),
        "qualification": {"passed": True},
        "interpretation": "gate_a_passed_associative_binding",
    }
    _strict_write(paths.result, result)
    preflight = launcher._artifact_reference(
        paths.preflight,
        repo_root=config.repo_root,
    )
    retained_result = launcher._artifact_reference(
        paths.result,
        repo_root=config.repo_root,
    )
    bundle_sha256 = launcher._launch_bundle_sha256(
        "formal_gate_a",
        launcher._GATE_A_SOURCE_COMMIT,
        preflight["sha256"],
        retained_result["sha256"],
    )
    manifest = {
        "schema_version": 1,
        "kind": "example21_authenticated_launch_manifest",
        "target": "formal_gate_a",
        "source_head": launcher._GATE_A_SOURCE_COMMIT,
        "bundle_valid": True,
        "process_succeeded": True,
        "artifact_schema_verified": True,
        "scientific_qualification_passed": True,
        "failure": None,
        "preflight": preflight,
        "result": retained_result,
        "bundle_sha256": bundle_sha256,
    }
    _strict_write(paths.manifest, manifest)
    result_sha256 = launcher.sha256_file(paths.result)
    manifest_sha256 = launcher.sha256_file(paths.manifest)
    monkeypatch.setattr(launcher, "_GATE_A_RESULT_SHA256", result_sha256)
    monkeypatch.setattr(launcher, "_GATE_A_MANIFEST_SHA256", manifest_sha256)
    monkeypatch.setattr(launcher, "_GATE_A_BUNDLE_SHA256", bundle_sha256)
    monkeypatch.setattr(depth, "_GATE_A_RESULT_SHA256", result_sha256)
    monkeypatch.setattr(depth, "_GATE_A_MANIFEST_SHA256", manifest_sha256)
    return launcher._load_gate_a_prerequisite(config)


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
        "planned_environment_schema",
        "git_dir_escape",
        "host_command_count",
        "host_command_argv",
        "host_head_output",
        "host_root_output",
        "image_record_missing",
        "image_command_argv",
        "image_stdout_json",
        "image_identity",
        "container_command_count",
        "container_command_argv",
        "container_version_output",
        "planned_argv_schema",
        "planned_mount_count",
        "planned_mount_source",
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
    elif mutation == "planned_environment_schema":
        preflight["planned_gate"]["environment"] = []
    elif mutation == "git_dir_escape":
        preflight["planned_gate"]["environment"]["GIT_DIR"] = "/outside/repo"
        preflight["container_source"]["environment"]["GIT_DIR"] = "/outside/repo"
    elif mutation == "host_command_count":
        preflight["source"]["commands"].pop()
    elif mutation == "host_command_argv":
        preflight["source"]["commands"][0]["argv"].insert(0, "env")
    elif mutation == "host_head_output":
        record = preflight["source"]["commands"][3]
        record["stdout"] = "c" * 40 + "\n"
        record["stdout_sha256"] = launcher._sha256_text(record["stdout"])
    elif mutation == "host_root_output":
        record = preflight["source"]["commands"][0]
        record["stdout"] = str(config.repo_root / "substitute") + "\n"
        record["stdout_sha256"] = launcher._sha256_text(record["stdout"])
    elif mutation == "image_record_missing":
        preflight["image"]["inspect_command"] = None
    elif mutation == "image_command_argv":
        preflight["image"]["inspect_command"]["argv"].insert(0, "env")
    elif mutation == "image_stdout_json":
        record = preflight["image"]["inspect_command"]
        record["stdout"] = "[]"
        record["stdout_sha256"] = launcher._sha256_text(record["stdout"])
    elif mutation == "image_identity":
        record = preflight["image"]["inspect_command"]
        inspection = json.loads(record["stdout"])
        inspection[0]["Id"] = "sha256:" + "c" * 64
        record["stdout"] = json.dumps(inspection)
        record["stdout_sha256"] = launcher._sha256_text(record["stdout"])
    elif mutation == "container_command_count":
        preflight["container_source"]["commands"].pop()
    elif mutation == "container_command_argv":
        preflight["container_source"]["commands"][0]["argv"].insert(0, "env")
    elif mutation == "container_version_output":
        record = preflight["container_source"]["commands"][0]
        record["stdout"] = "not Git\n"
        record["stdout_sha256"] = launcher._sha256_text(record["stdout"])
        preflight["container_source"]["git_version"] = "not Git"
    elif mutation == "planned_argv_schema":
        preflight["planned_gate"]["argv"] = "docker run"
    elif mutation == "planned_mount_count":
        argv = preflight["planned_gate"]["argv"]
        index = max(index for index, value in enumerate(argv) if value == "--mount")
        del argv[index : index + 2]
    elif mutation == "planned_mount_source":
        argv = preflight["planned_gate"]["argv"]
        index = next(
            index
            for index, value in enumerate(argv)
            if isinstance(value, str) and "dst=/cache/jax" in value
        )
        argv[index] = "type=volume,src=bad/path,dst=/cache/jax"
    launcher.write_strict_json(paths.preflight, preflight)
    _refresh_bundle_manifest(config, "one_update")

    with pytest.raises(
        launcher.ProvenanceError,
        match="preflight|planned Gate|retained|environment|mount|command",
    ):
        launcher.load_authenticated_admission(
            paths.manifest,
            target="one_update",
            head=_HEAD,
            image_id=_IMAGE_ID,
            repo_root=config.repo_root,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "gate_command_missing",
        "gate_command_argv",
        "postflight_failure",
        "postflight_host_count",
        "postflight_host_argv",
        "postflight_host_head",
        "postflight_container_missing",
        "postflight_container_argv",
        "postflight_container_output",
    ],
)
def test_rehashed_manifest_execution_tampering_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    config = _config(tmp_path, "one_update")
    launcher.launch(config, command_runner=FakeRunner(config.repo_root, "one_update"))
    paths = launcher.target_paths(config, _HEAD, "one_update")
    manifest = launcher.load_strict_json(paths.manifest)

    if mutation == "gate_command_missing":
        manifest["gate_command"] = None
    elif mutation == "gate_command_argv":
        manifest["gate_command"]["argv"].insert(0, "env")
    elif mutation == "postflight_failure":
        manifest["failure"] = "tampered"
    elif mutation == "postflight_host_count":
        manifest["postflight"]["host_commands"].pop()
    elif mutation == "postflight_host_argv":
        manifest["postflight"]["host_commands"][0]["argv"].insert(0, "env")
    elif mutation == "postflight_host_head":
        record = manifest["postflight"]["host_commands"][3]
        record["stdout"] = "c" * 40 + "\n"
        record["stdout_sha256"] = launcher._sha256_text(record["stdout"])
    elif mutation == "postflight_container_missing":
        manifest["postflight"]["container"] = None
    elif mutation == "postflight_container_argv":
        manifest["postflight"]["container_commands"][0]["argv"].insert(0, "env")
    else:
        record = manifest["postflight"]["container_commands"][0]
        record["stdout"] = "not Git\n"
        record["stdout_sha256"] = launcher._sha256_text(record["stdout"])
        manifest["postflight"]["container"]["git_version"] = "not Git"
    launcher.write_strict_json(paths.manifest, manifest)

    with pytest.raises(launcher.ProvenanceError):
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


def test_launcher_target_extension_preserves_existing_targets() -> None:
    assert launcher._TARGETS == (
        "one_update",
        "stability_256",
        "formal_gate_a",
        "gate_b_init",
        "formal_gate_b",
    )


def test_gate_b_targets_pin_the_authenticated_gate_a_artifact_bytes() -> None:
    config = depth.DepthGateConfig()

    assert launcher._GATE_A_SOURCE_COMMIT == config.gate_a_source_commit
    assert launcher._GATE_A_RESULT_SHA256 == config.gate_a_result_sha256
    assert launcher._GATE_A_MANIFEST_SHA256 == config.gate_a_manifest_sha256
    assert launcher._GATE_A_RESULT_SHA256 == (
        "3a585e739715b31757082b50fe57b98ca50107891f7c79edaa7e5e54c90ad632"
    )
    assert launcher._GATE_A_MANIFEST_SHA256 == (
        "69d690daa5023f5b3ce22b0e65ea09a1a6706687d792e998651422f6d6ea15cf"
    )
    assert launcher._GATE_A_BUNDLE_SHA256 == (
        "ba850a205c4691d573facef7b8e90cabd4824905c73fcd4f6add29293cd95875"
    )


@pytest.mark.parametrize("target", ["gate_b_init", "formal_gate_b"])
def test_gate_b_target_paths_and_module_argv_are_exact(
    tmp_path: Path,
    target: str,
) -> None:
    config = _depth_config(tmp_path, target)
    paths = launcher.target_paths(config, _HEAD, target)

    assert paths.result == (
        config.output_dir / f"{_HEAD}-{target.replace('_', '-')}.json"
    )
    command = launcher.gate_command(
        config,
        image_id=_IMAGE_ID,
        head=_HEAD,
        paths=paths,
        git_dir_in_container="/git-common/worktrees/gate",
        admission_manifests=None,
    )
    python_index = command.index("python")
    gate_a_base = (
        "/work/var/example21-binding-gate/"
        f"{depth.DepthGateConfig().gate_a_source_commit}-formal-gate-a"
    )
    expected = [
        "python",
        "-m",
        "examples.pp_prop.latent_workspace_depth_gate",
        "--target",
        target,
        "--gate-a-result",
        f"{gate_a_base}.json",
        "--gate-a-manifest",
        f"{gate_a_base}.manifest.json",
    ]
    if target == "formal_gate_b":
        expected.extend(
            [
                "--gate-b-init-manifest",
                str(
                    launcher.target_paths(
                        config, _HEAD, "gate_b_init"
                    ).container_result
                ).replace(".json", ".manifest.json"),
            ]
        )
    expected.extend(["--output", str(paths.container_result)])

    assert command[python_index:] == expected
    assert "--training-updates" not in command
    assert "--neuron-count" not in command
    assert ("--gate-b-init-manifest" in command) is (target == "formal_gate_b")


def test_formal_gate_b_requires_authenticated_current_init_manifest(
    tmp_path: Path,
) -> None:
    config = _depth_config(tmp_path, "formal_gate_b")
    expected = launcher.target_paths(config, _HEAD, "gate_b_init").manifest

    assert expected.name == f"{_HEAD}-gate-b-init.manifest.json"
    with pytest.raises(launcher.ProvenanceError, match="Gate B initialization"):
        launcher._load_gate_b_init_manifest(
            config,
            head=_HEAD,
            image_id=_IMAGE_ID,
        )


@pytest.mark.parametrize(
    ("target", "changing_loader"),
    [
        ("gate_b_init", "_load_gate_a_prerequisite"),
        ("formal_gate_b", "_load_gate_b_init_manifest"),
    ],
)
def test_gate_b_launch_reauthenticates_prerequisites_before_signing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    changing_loader: str,
) -> None:
    config = _depth_config(tmp_path, target)
    stable = {
        "qualification_passed": True,
        "result_sha256": depth.DepthGateConfig().gate_a_result_sha256,
        "manifest_sha256": depth.DepthGateConfig().gate_a_manifest_sha256,
        "source_commit": depth.DepthGateConfig().gate_a_source_commit,
    }
    calls = 0

    def changed_after_gate(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls > 1:
            changed = copy.deepcopy(stable)
            changed["bundle_sha256"] = "0" * 64
            return changed
        return copy.deepcopy(stable)

    def stable_loader(*args: object, **kwargs: object) -> dict[str, object]:
        return copy.deepcopy(stable)

    monkeypatch.setattr(
        launcher,
        "_load_gate_a_prerequisite",
        (
            changed_after_gate
            if changing_loader == "_load_gate_a_prerequisite"
            else stable_loader
        ),
        raising=False,
    )
    monkeypatch.setattr(
        launcher,
        "_load_gate_b_init_manifest",
        (
            changed_after_gate
            if changing_loader == "_load_gate_b_init_manifest"
            else stable_loader
        ),
        raising=False,
    )
    monkeypatch.setattr(
        launcher,
        "_validate_target_result",
        lambda *args, **kwargs: True,
    )

    with pytest.raises(launcher.ProvenanceError, match="changed before signing"):
        launcher.launch(
            config,
            command_runner=FakeRunner(config.repo_root, target),
        )

    assert calls == 2


def test_formal_gate_b_recomputes_qualification_from_retained_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = depth_fixtures._passing_depth_report()
    retained = {
        "passed": True,
        "criteria": {"retained_evidence_recomputed": True},
        "interpretation": "gate_b_passed_demonstrated_depth_application",
    }
    report["qualification"] = copy.deepcopy(retained)
    original_accuracy = report["evaluation"]["efforts"]["8"]["intact"][
        "accuracy"
    ]

    def recompute(
        candidate: dict[str, Any],
        *,
        config: depth.DepthGateConfig,
    ) -> dict[str, Any]:
        del config
        if (
            candidate["evaluation"]["efforts"]["8"]["intact"]["accuracy"]
            == original_accuracy
        ):
            return copy.deepcopy(retained)
        return {
            "passed": False,
            "criteria": {"retained_evidence_recomputed": False},
            "interpretation": "gate_b_failed_stop_no_capability_conclusion",
        }

    monkeypatch.setattr(depth, "_qualification_report", recompute)
    assert launcher._validate_gate_b_scientific_result(
        copy.deepcopy(report),
        target="formal_gate_b",
        head=_HEAD,
        image_id=_IMAGE_ID,
    ) is True

    report["evaluation"]["efforts"]["8"]["intact"]["accuracy"] = 0.0
    with pytest.raises(launcher.ProvenanceError, match="qualification"):
        launcher._validate_gate_b_scientific_result(
            report,
            target="formal_gate_b",
            head=_HEAD,
            image_id=_IMAGE_ID,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_result",
        "manifest_bytes",
        "result_bytes",
        "manifest_semantics",
        "missing_reference",
        "bundle_digest",
        "result_semantics",
    ],
)
def test_gate_a_prerequisite_loader_fails_closed_on_each_authenticated_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config = _depth_config(tmp_path, "gate_b_init")
    _install_gate_a_prerequisite(config, monkeypatch)
    paths = launcher._gate_a_artifact_paths(config)

    if mutation == "missing_result":
        paths.result.unlink()
    elif mutation == "manifest_bytes":
        paths.manifest.write_bytes(paths.manifest.read_bytes() + b" ")
    elif mutation == "result_bytes":
        paths.result.write_bytes(paths.result.read_bytes() + b" ")
    elif mutation in {
        "manifest_semantics",
        "missing_reference",
        "bundle_digest",
    }:
        manifest = launcher.load_strict_json(paths.manifest)
        if mutation == "manifest_semantics":
            manifest["target"] = "substitute"
        elif mutation == "missing_reference":
            manifest["preflight"] = None
        else:
            manifest["bundle_sha256"] = "0" * 64
        launcher.write_strict_json(paths.manifest, manifest)
        monkeypatch.setattr(
            launcher,
            "_GATE_A_MANIFEST_SHA256",
            launcher.sha256_file(paths.manifest),
        )
    else:
        result = launcher.load_strict_json(paths.result)
        result["learner"] = "bptt"
        launcher.write_strict_json(paths.result, result)
        result_sha256 = launcher.sha256_file(paths.result)
        monkeypatch.setattr(launcher, "_GATE_A_RESULT_SHA256", result_sha256)
        manifest = launcher.load_strict_json(paths.manifest)
        manifest["result"] = launcher._artifact_reference(
            paths.result,
            repo_root=config.repo_root,
        )
        bundle_sha256 = launcher._launch_bundle_sha256(
            "formal_gate_a",
            launcher._GATE_A_SOURCE_COMMIT,
            manifest["preflight"]["sha256"],
            result_sha256,
        )
        manifest["bundle_sha256"] = bundle_sha256
        launcher.write_strict_json(paths.manifest, manifest)
        monkeypatch.setattr(launcher, "_GATE_A_BUNDLE_SHA256", bundle_sha256)
        monkeypatch.setattr(
            launcher,
            "_GATE_A_MANIFEST_SHA256",
            launcher.sha256_file(paths.manifest),
        )

    with pytest.raises(launcher.ProvenanceError):
        launcher._load_gate_a_prerequisite(config)


def test_authenticated_gate_b_init_and_formal_bundle_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_config = _depth_config(tmp_path, "gate_b_init")
    gate_a = _install_gate_a_prerequisite(init_config, monkeypatch)
    init_runner = DepthFakeRunner(
        init_config.repo_root,
        "gate_b_init",
        gate_a=gate_a,
    )

    init_manifest_path = launcher.launch(
        init_config,
        command_runner=init_runner,
    )

    assert init_runner.sidecar_seen_before_gate is True
    init_manifest = launcher.load_strict_json(init_manifest_path)
    assert init_manifest["bundle_valid"] is True
    assert init_manifest["scientific_qualification_passed"] is True
    init_bundle = launcher._load_gate_b_init_manifest(
        init_config,
        head=_HEAD,
        image_id=_IMAGE_ID,
    )
    assert set(init_bundle) == {
        "target",
        "source_head",
        "image_digest",
        "bundle_sha256",
        "manifest_sha256",
        "preflight_sha256",
        "result_sha256",
        "admission",
    }
    assert init_bundle["admission"]["qualification"]["passed"] is True

    formal_config = launcher.LaunchConfig(
        target="formal_gate_b",
        repo_root=init_config.repo_root,
        output_dir=init_config.output_dir,
    )
    formal_runner = DepthFakeRunner(
        formal_config.repo_root,
        "formal_gate_b",
        gate_a=gate_a,
        gate_b_init=init_bundle,
    )

    formal_manifest_path = launcher.launch(
        formal_config,
        command_runner=formal_runner,
    )

    formal_manifest = launcher.load_strict_json(formal_manifest_path)
    assert formal_runner.sidecar_seen_before_gate is True
    assert formal_manifest["bundle_valid"] is True
    assert formal_manifest["process_succeeded"] is True
    assert formal_manifest["scientific_qualification_passed"] is True
    formal_result = launcher.load_strict_json(
        launcher.target_paths(formal_config, _HEAD).result
    )
    assert formal_result["prerequisites"]["gate_a"] == gate_a
    assert formal_result["prerequisites"]["gate_b_initialization"] == init_bundle


def test_authenticated_formal_gate_b_bundle_reloads_for_gate_c(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_config = _depth_config(tmp_path, "gate_b_init")
    gate_a = _install_gate_a_prerequisite(init_config, monkeypatch)
    launcher.launch(
        init_config,
        command_runner=DepthFakeRunner(
            init_config.repo_root,
            "gate_b_init",
            gate_a=gate_a,
        ),
    )
    init_bundle = launcher._load_gate_b_init_manifest(
        init_config,
        head=_HEAD,
        image_id=_IMAGE_ID,
    )
    formal_config = launcher.LaunchConfig(
        target="formal_gate_b",
        repo_root=init_config.repo_root,
        output_dir=init_config.output_dir,
    )
    manifest_path = launcher.launch(
        formal_config,
        command_runner=DepthFakeRunner(
            formal_config.repo_root,
            "formal_gate_b",
            gate_a=gate_a,
            gate_b_init=init_bundle,
        ),
    )
    paths = launcher.target_paths(formal_config, _HEAD, "formal_gate_b")
    manifest = launcher.load_strict_json(manifest_path)
    monkeypatch.setattr(launcher, "_GATE_B_SOURCE_COMMIT", _HEAD)
    monkeypatch.setattr(launcher, "_GATE_B_IMAGE_ID", _IMAGE_ID)
    monkeypatch.setattr(
        launcher,
        "_GATE_B_PREFLIGHT_SHA256",
        launcher.sha256_file(paths.preflight),
    )
    monkeypatch.setattr(
        launcher,
        "_GATE_B_RESULT_SHA256",
        launcher.sha256_file(paths.result),
    )
    monkeypatch.setattr(
        launcher,
        "_GATE_B_MANIFEST_SHA256",
        launcher.sha256_file(paths.manifest),
    )
    monkeypatch.setattr(
        launcher,
        "_GATE_B_BUNDLE_SHA256",
        manifest["bundle_sha256"],
    )

    bundle = launcher.load_authenticated_formal_gate_b(
        manifest_path,
        repo_root=formal_config.repo_root,
    )

    argv = bundle["preflight"]["planned_gate"]["argv"]
    python_index = argv.index("python")
    assert argv[python_index : python_index + 3] == [
        "python",
        "-m",
        "examples.pp_prop.latent_workspace_depth_gate",
    ]
    assert "--gate-b-init-manifest" in argv
    assert bundle["result"]["prerequisites"]["gate_b_initialization"] == (
        init_bundle
    )
    assert bundle["source_head"] == _HEAD
    assert bundle["image_digest"] == _IMAGE_ID
    assert bundle["preflight_sha256"] == launcher.sha256_file(paths.preflight)
    assert bundle["result_sha256"] == launcher.sha256_file(paths.result)
    assert bundle["manifest_sha256"] == launcher.sha256_file(paths.manifest)
    assert bundle["bundle_sha256"] == manifest["bundle_sha256"]


def test_formal_gate_b_retained_bundle_identity_is_pinned() -> None:
    assert launcher._GATE_B_SOURCE_COMMIT == (
        "dafa64a8b4c3848241baa117affa55b632518a8e"
    )
    assert launcher._GATE_B_IMAGE_ID == (
        "sha256:35349cb07c49e275b15c5c563a8d75fa08b49d4b0829d86939c1c09fb1ef6d16"
    )
    assert launcher._GATE_B_PREFLIGHT_SHA256 == (
        "91e86d92670cd33d3f4206ff3d5096e3721104996a9506223a9e34c082dd052f"
    )
    assert launcher._GATE_B_RESULT_SHA256 == (
        "6456537ea108cea8892d00c8a71c1f647217e074b525bc9ed01b64aef9001766"
    )
    assert launcher._GATE_B_MANIFEST_SHA256 == (
        "99c42985e203413eb0600a5dabe321188776eff8058500dc86f4a1618b413eab"
    )
    assert launcher._GATE_B_BUNDLE_SHA256 == (
        "be07e8c92d8deaa94508f34dcee45f5feb09740cb2804778d6280a2fa3c64851"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "manifest_admission",
        "result_reference",
        "bundle_digest",
        "preflight_prerequisite",
        "preflight_command",
        "result_prerequisite",
        "scientific_failure",
    ],
)
def test_gate_b_init_loader_fails_closed_at_each_authenticated_layer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    config = _depth_config(tmp_path, "gate_b_init")
    gate_a = _install_gate_a_prerequisite(config, monkeypatch)
    launcher.launch(
        config,
        command_runner=DepthFakeRunner(
            config.repo_root,
            "gate_b_init",
            gate_a=gate_a,
        ),
    )
    paths = launcher.target_paths(config, _HEAD, "gate_b_init")

    if mutation in {"manifest_admission", "result_reference", "bundle_digest"}:
        manifest = launcher.load_strict_json(paths.manifest)
        if mutation == "manifest_admission":
            manifest["scientific_qualification_passed"] = False
        elif mutation == "result_reference":
            manifest["result"]["size_bytes"] = False
        else:
            manifest["bundle_sha256"] = "0" * 64
        launcher.write_strict_json(paths.manifest, manifest)
    elif mutation in {"preflight_prerequisite", "preflight_command"}:
        preflight = launcher.load_strict_json(paths.preflight)
        if mutation == "preflight_prerequisite":
            preflight["gate_b_prerequisites"]["gate_a"]["bundle_sha256"] = "0" * 64
        else:
            command = preflight["planned_gate"]["argv"]
            gate_a_index = command.index("--gate-a-result") + 1
            command[gate_a_index] = "/work/substitute-gate-a.json"
        launcher.write_strict_json(paths.preflight, preflight)
        _refresh_bundle_manifest(config, "gate_b_init")
    else:
        result = launcher.load_strict_json(paths.result)
        if mutation == "result_prerequisite":
            result["prerequisites"]["gate_a"]["bundle_sha256"] = "0" * 64
        else:
            failed = {
                "passed": False,
                "interpretation": "gate_b_initialization_failed",
            }
            monkeypatch.setattr(
                depth,
                "_gate_b_initialization_qualification",
                lambda candidate, config: copy.deepcopy(failed),
            )
            result["qualification"] = failed
        launcher.write_strict_json(paths.result, result)
        _refresh_bundle_manifest(config, "gate_b_init")

    with pytest.raises(launcher.ProvenanceError):
        launcher._load_gate_b_init_manifest(
            config,
            head=_HEAD,
            image_id=_IMAGE_ID,
        )


@pytest.mark.parametrize(
    ("target", "initialization", "message"),
    [
        ("gate_b_init", {"unexpected": True}, "circular prerequisite"),
        ("formal_gate_b", {"bundle": "changed"}, "initialization manifest"),
    ],
)
def test_gate_b_result_rejects_wrong_initialization_binding_before_science(
    tmp_path: Path,
    target: str,
    initialization: dict[str, Any],
    message: str,
) -> None:
    gate_a = {"qualification_passed": True}
    embedded_initialization = {"bundle": "retained"}
    result = {
        "prerequisites": {
            "gate_a": gate_a,
            "gate_b_initialization": embedded_initialization,
        }
    }

    with pytest.raises(launcher.ProvenanceError, match=message):
        launcher._validate_target_result(
            result,
            target=target,
            head=_HEAD,
            image_id=_IMAGE_ID,
            admission_manifests=None,
            repo_root=tmp_path,
            gate_a_prerequisite=gate_a,
            gate_b_init_bundle=initialization,
        )


@pytest.mark.parametrize(
    ("target", "explicit_output", "scientific_passed", "expected_code", "directory"),
    [
        ("gate_b_init", None, True, 0, "example21-depth-gate"),
        ("formal_gate_a", None, True, 0, "example21-binding-gate"),
        ("one_update", "custom-output", False, 3, "custom-output"),
    ],
)
def test_cli_selects_scoped_defaults_and_reports_scientific_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    target: str,
    explicit_output: str | None,
    scientific_passed: bool,
    expected_code: int,
    directory: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    captured: list[launcher.LaunchConfig] = []

    def fake_launch(config: launcher.LaunchConfig) -> Path:
        captured.append(config)
        destination = config.output_dir / "cli.manifest.json"
        launcher.write_strict_json(
            destination,
            {"scientific_qualification_passed": scientific_passed},
        )
        return destination

    monkeypatch.setattr(launcher, "launch", fake_launch)
    argv = ["--target", target, "--repo-root", str(repo)]
    if explicit_output is not None:
        argv.extend(["--output-dir", explicit_output])

    assert launcher.main(argv) == expected_code

    assert len(captured) == 1
    expected_output = (
        repo / explicit_output
        if explicit_output is not None
        else repo / "var" / directory
    )
    assert captured[0].output_dir == expected_output
    streams = capsys.readouterr()
    assert "cli.manifest.json" in streams.out
    if expected_code == 3:
        assert "scientific admission failed" in streams.err
    else:
        assert streams.err == ""


def test_cli_returns_provenance_error_without_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    def fail_launch(config: launcher.LaunchConfig) -> Path:
        del config
        raise launcher.ProvenanceError("authenticated launch rejected")

    monkeypatch.setattr(launcher, "launch", fail_launch)

    assert launcher.main(
        ["--target", "formal_gate_b", "--repo-root", str(repo)]
    ) == 2
    streams = capsys.readouterr()
    assert streams.out == ""
    assert "authenticated launch rejected" in streams.err


def test_missing_child_result_writes_a_complete_failure_manifest(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "one_update")

    class MissingResultRunner(FakeRunner):
        def __call__(
            self, command: Sequence[str], **kwargs: object
        ) -> subprocess.CompletedProcess[str]:
            argv = list(command)
            if argv and argv[0] == "docker" and "python" in argv:
                self.calls.append(argv)
                result_arg = argv[argv.index("--output") + 1]
                destination = self.repo / Path(result_arg.removeprefix("/work/"))
                self.sidecar_seen_before_gate = destination.with_suffix(
                    ".preflight.json"
                ).is_file()
                return subprocess.CompletedProcess(
                    argv,
                    0,
                    stdout="",
                    stderr="",
                )
            return super().__call__(command, **kwargs)

    with pytest.raises(launcher.ProvenanceError, match="did not create"):
        launcher.launch(
            config,
            command_runner=MissingResultRunner(config.repo_root, "one_update"),
        )

    manifest = launcher.load_strict_json(
        launcher.target_paths(config, _HEAD).manifest
    )
    assert manifest["bundle_valid"] is False
    assert manifest["process_succeeded"] is True
    assert manifest["artifact_schema_verified"] is False
    assert manifest["scientific_qualification_passed"] is False
    assert manifest["result"] is None
    assert manifest["gate_command"] is not None
    assert manifest["postflight"]["clean"] is True
    assert "did not create" in manifest["failure"]


def test_unexpected_result_validator_error_is_manifested_and_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, "one_update")

    def fail_validation(*args: object, **kwargs: object) -> bool:
        raise RuntimeError("validator exploded")

    monkeypatch.setattr(launcher, "_validate_target_result", fail_validation)

    with pytest.raises(launcher.ProvenanceError, match="validator exploded") as caught:
        launcher.launch(
            config,
            command_runner=FakeRunner(config.repo_root, "one_update"),
        )

    assert isinstance(caught.value.__cause__, RuntimeError)
    manifest = launcher.load_strict_json(
        launcher.target_paths(config, _HEAD).manifest
    )
    assert manifest["bundle_valid"] is False
    assert manifest["process_succeeded"] is True
    assert manifest["result"] is None
    assert manifest["failure"] == "RuntimeError: validator exploded"
