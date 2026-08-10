"""Tests for fixed trace half-life coordinate-search configuration."""

from pathlib import Path

from temporal_benchmark_search_config import DEVELOPMENT_BUNDLES
from temporal_benchmark_trace_search_config import (
    FIXED_TRACE_SEARCH_GAIN,
    FIXED_TRACE_SEARCH_RATES,
    HORIZON_TRACE_GRIDS,
    TraceSearchSettings,
    coordinate_candidates,
    expected_trace_benchmark_config,
    trace_benchmark_command,
)


def _settings(tmp_path: Path) -> TraceSearchSettings:
    return TraceSearchSettings(
        source_root=tmp_path,
        output_directory=tmp_path / "trace-search",
        benchmark_script=tmp_path / "17-temporal-credit-benchmark.py",
        manifest_path=tmp_path / "manifest.json",
        python_executable="python-test",
        container_image_digest="sha256:test-image",
        source_commit="0123456789abcdef",
        device="cpu",
        recurrent_weight_decay=1e-5,
    )


def _value(command: tuple[str, ...], option: str) -> str:
    return command[command.index(option) + 1]


def test_coordinate_grids_and_order_match_the_frozen_spec() -> None:
    assert [item.horizon for item in HORIZON_TRACE_GRIDS] == [
        "short",
        "medium",
        "long",
    ]
    assert [item.updates for item in HORIZON_TRACE_GRIDS] == [200, 400, 800]
    assert [item.half_lives for item in HORIZON_TRACE_GRIDS] == [
        (5.0, 10.0),
        (10.0, 20.0, 30.0),
        (30.0, 60.0, 100.0),
    ]
    assert [item.provisional_f for item in HORIZON_TRACE_GRIDS] == [10, 20, 60]


def test_expected_config_uses_selected_optimizer_and_coordinate_order(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    grid = HORIZON_TRACE_GRIDS[1]
    x_candidate = coordinate_candidates(grid, "x", grid.provisional_f)[2]
    x_config = expected_trace_benchmark_config(
        settings, grid, x_candidate, DEVELOPMENT_BUNDLES[0]
    )
    f_candidate = coordinate_candidates(grid, "f", x_config.trace_half_life_x_steps)[0]
    f_config = expected_trace_benchmark_config(
        settings, grid, f_candidate, DEVELOPMENT_BUNDLES[0]
    )

    assert (x_config.trace_half_life_x_steps, x_config.trace_half_life_f_steps) == (
        30,
        20,
    )
    assert (f_config.trace_half_life_x_steps, f_config.trace_half_life_f_steps) == (
        30,
        10,
    )
    assert x_config.learning_rates == FIXED_TRACE_SEARCH_RATES
    assert x_config.gain == FIXED_TRACE_SEARCH_GAIN == 0.8
    assert x_config.updates == 400
    assert x_config.recurrent_weight_decay == 1e-5
    assert x_config.gradient_clip_norms.readout == 1.0
    assert x_config.curriculum is False
    assert x_config.gradient_evidence is False
    assert x_config.sealed_test is False


def test_command_can_only_materialize_development_validation_evidence(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    grid = HORIZON_TRACE_GRIDS[0]
    candidate = coordinate_candidates(grid, "x", grid.provisional_f)[0]
    command = trace_benchmark_command(
        settings, grid, candidate, DEVELOPMENT_BUNDLES[1], tmp_path / "raw.json"
    )

    assert command[:2] == ("python-test", str(settings.benchmark_script))
    assert _value(command, "--horizon") == "short"
    assert _value(command, "--updates") == "200"
    assert _value(command, "--trace-half-life-x-steps") == "5.0"
    assert _value(command, "--trace-half-life-f-steps") == "10.0"
    assert _value(command, "--readout-learning-rate") == "0.01"
    assert _value(command, "--feedforward-learning-rate") == "0.001"
    assert _value(command, "--recurrent-learning-rate") == "0.001"
    assert _value(command, "--recurrent-weight-decay") == "1e-05"
    assert "--no-curriculum" in command
    assert "--no-gradient-evidence" in command
    assert "--no-sealed-test" in command
    assert "--sealed-test" not in command
    assert "--allow-dirty" in command
