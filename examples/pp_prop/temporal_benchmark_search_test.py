"""Tests for the thin development optimizer-search CLI."""

from pathlib import Path

from temporal_benchmark_search import _parser, _settings


def test_cli_defaults_to_all_group_clipping_without_running_model(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("BRAINTRACE_IMAGE_DIGEST", "sha256:test-image")
    monkeypatch.setenv("BRAINTRACE_SOURCE_COMMIT", "0123456789abcdef")
    values = _parser().parse_args(
        [
            "--output-directory",
            str(tmp_path),
            "--clip-norm",
            "2.0",
            "--device",
            "cpu",
        ]
    )

    settings = _settings(values)

    assert settings.output_directory == tmp_path.resolve()
    assert settings.device == "cpu"
    assert settings.gradient_clip_norms.readout == 2.0
    assert settings.gradient_clip_norms.feedforward == 2.0
    assert settings.gradient_clip_norms.recurrent == 2.0
    assert settings.container_image_digest == "sha256:test-image"
    assert settings.source_commit == "0123456789abcdef"
