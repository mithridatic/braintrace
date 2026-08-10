"""Tests for the paired-bundle command-line configuration."""

import temporal_benchmark_curriculum_adoption_bundle as bundle_cli


def test_bundle_cli_builds_unsealed_independent_trace_configuration(tmp_path) -> None:
    values = bundle_cli._parser().parse_args(
        [
            "--json-output",
            str(tmp_path / "raw.json"),
            "--device",
            "cpu",
            "--short-trace-half-life-x-steps",
            "5",
            "--short-trace-half-life-f-steps",
            "10",
            "--recurrent-clip-norm",
            "disabled",
        ]
    )

    config = bundle_cli._config(values)

    assert config.curriculum is True
    assert config.sealed_test is False
    assert config.curriculum_trace_half_lives.short.x == 5.0
    assert config.curriculum_trace_half_lives.short.f == 10.0
    assert config.gradient_clip_norms.recurrent is None
