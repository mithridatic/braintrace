"""Tests for the curriculum-adoption driver CLI."""

import msgspec_json

import temporal_benchmark_curriculum_adoption as adoption_cli
from temporal_benchmark_example15_control_run import fixed_config_document


def test_driver_requires_and_preserves_static_control_and_provenance(tmp_path) -> None:
    values = adoption_cli._parser().parse_args(
        [
            "--container-image-digest",
            "sha256:test",
            "--source-commit",
            "abcdef",
            "--source-dirty",
            "true",
            "--example15-accuracy-change",
            "-0.005",
            "--output-directory",
            str(tmp_path),
        ]
    )

    settings = adoption_cli._settings(values)

    assert settings.source_dirty is True
    assert settings.example15_accuracy_change == -0.005
    assert settings.output_directory == tmp_path.resolve()


def test_driver_prefers_validated_example15_static_control_artifact(tmp_path) -> None:
    control = tmp_path / "static-control.json"
    control.write_text(
        msgspec_json.dumps(
            {
                "schema_version": 1,
                "kind": "temporal_credit_example15_static_control",
                "status": "completed",
                "development_only": True,
                "sealed_test": False,
                "fixed_config": fixed_config_document(),
                "baseline": {
                    "artifact_sha256": "a" * 64,
                    "provenance": {
                        "source_commit": "baseline",
                        "source_dirty": False,
                        "container_image_digest": "sha256:baseline",
                    },
                    "mean_accuracy": 0.96,
                    "acceptance_passed": True,
                    "accepted_baseline": True,
                },
                "current": {
                    "artifact_sha256": "b" * 64,
                    "provenance": {
                        "source_commit": "current",
                        "source_dirty": True,
                        "container_image_digest": "sha256:current",
                    },
                    "mean_accuracy": 0.955,
                },
                "example15_accuracy_change": -0.005,
                "static_control_gate_passed": True,
            }
        ),
        encoding="utf-8",
    )
    values = adoption_cli._parser().parse_args(
        [
            "--source-commit",
            "abcdef",
            "--source-dirty",
            "true",
            "--container-image-digest",
            "sha256:test",
            "--example15-static-control-result",
            str(control),
        ]
    )

    assert adoption_cli._settings(values).example15_accuracy_change == -0.005
