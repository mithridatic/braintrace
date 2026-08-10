"""Tests for the thin recurrent-weight-decay search CLI."""

from pathlib import Path

from temporal_benchmark_weight_decay_search import _parser, _settings


def test_weight_decay_cli_exposes_execution_not_scientific_knobs(
    tmp_path: Path,
) -> None:
    parser = _parser()
    destinations = {action.dest for action in parser._actions}
    values = parser.parse_args(
        [
            "--output-directory",
            str(tmp_path),
            "--container-image-digest",
            "sha256:test-image",
            "--source-commit",
            "0123456789abcdef",
            "--device",
            "cpu",
        ]
    )

    settings = _settings(values)

    assert settings.output_directory == tmp_path.resolve()
    assert settings.container_image_digest == "sha256:test-image"
    assert settings.source_commit == "0123456789abcdef"
    assert settings.device == "cpu"
    assert "gain" not in destinations
    assert "readout_learning_rate" not in destinations
    assert "trace_half_life_x_steps" not in destinations
    assert "clip_norm" not in destinations
    assert "recurrent_weight_decay" not in destinations
