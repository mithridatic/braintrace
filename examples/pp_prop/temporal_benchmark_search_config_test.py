"""Tests for pure optimizer-search configuration and command planning."""

from pathlib import Path

from temporal_benchmark_search_config import (
    DEVELOPMENT_BUNDLES,
    ORDERED_LEARNING_RATE_GRID,
    SearchSettings,
    benchmark_command,
    expected_benchmark_config,
    ordered_candidates,
)


def _settings(tmp_path: Path) -> SearchSettings:
    return SearchSettings(
        source_root=tmp_path,
        output_directory=tmp_path / "search",
        benchmark_script=tmp_path / "17-temporal-credit-benchmark.py",
        manifest_path=tmp_path / "manifest.json",
        python_executable="python-test",
        container_image_digest="sha256:test-image",
        source_commit="0123456789abcdef",
    )


def _value(command: tuple[str, ...], option: str) -> str:
    return command[command.index(option) + 1]


def test_local_ordered_grid_matches_optimizer_grid() -> None:
    from temporal_benchmark_optimizer import successive_halving_grid

    assert ORDERED_LEARNING_RATE_GRID == successive_halving_grid()
    assert len(set(ORDERED_LEARNING_RATE_GRID)) == 27


def test_command_is_fresh_development_validation_only_config(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    candidate = ordered_candidates()[7]
    command = benchmark_command(
        settings, candidate, DEVELOPMENT_BUNDLES[1], 300, tmp_path / "raw.json"
    )
    expected = expected_benchmark_config(
        settings, candidate, DEVELOPMENT_BUNDLES[1], 300
    )

    assert command[:2] == ("python-test", str(settings.benchmark_script))
    assert _value(command, "--updates") == "300"
    assert _value(command, "--bundle-id") == DEVELOPMENT_BUNDLES[1]
    assert _value(command, "--readout-learning-rate") == repr(candidate.readout)
    assert _value(command, "--readout-clip-norm") == "1.0"
    assert _value(command, "--feedforward-clip-norm") == "1.0"
    assert _value(command, "--recurrent-clip-norm") == "1.0"
    assert "--no-curriculum" in command
    assert "--no-gradient-evidence" in command
    assert "--no-sealed-test" in command
    assert "--sealed-test" not in command
    assert "--allow-dirty" in command
    assert expected.sealed_test is False
    assert expected.gradient_evidence is False
    assert expected.curriculum is False
    assert expected.gradient_clip_norms.readout == 1.0
