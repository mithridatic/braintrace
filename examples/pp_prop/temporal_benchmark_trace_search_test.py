"""Tests for the thin trace half-life search CLI."""

from pathlib import Path

from temporal_benchmark_trace_search import _parser, _settings


def test_cli_exposes_pending_weight_decay_but_not_frozen_scientific_knobs(
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
            "--recurrent-weight-decay",
            "0.00001",
            "--device",
            "cpu",
        ]
    )

    settings = _settings(values)

    assert settings.output_directory == tmp_path.resolve()
    assert settings.recurrent_weight_decay == 1e-5
    assert settings.device == "cpu"
    assert "recurrent_weight_decay" in destinations
    assert "gain" not in destinations
    assert "readout_learning_rate" not in destinations
    assert "trace_half_life_x_steps" not in destinations
    assert "clip_norm" not in destinations
