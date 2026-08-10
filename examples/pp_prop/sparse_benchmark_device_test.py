"""Tests for sparse pp-prop benchmark device selection and memory accounting."""

import os

import pytest

from sparse_benchmark_device import (
    PLATFORM_VARIABLE,
    apply_device_selection,
    device_memory_peak_bytes,
    verify_device_selection,
)


class _Device:
    def __init__(self, statistics):
        self._statistics = statistics

    def memory_stats(self):
        return self._statistics


def test_cpu_selection_pins_the_host_platform() -> None:
    environment: dict[str, str] = {}

    apply_device_selection("cpu", environment)

    assert environment == {PLATFORM_VARIABLE: "cpu"}


@pytest.mark.parametrize("device", ["auto", "gpu"])
def test_non_cpu_selection_leaves_the_platform_unset(device: str) -> None:
    environment = {PLATFORM_VARIABLE: "supplied"}

    apply_device_selection(device, environment)

    assert environment == {PLATFORM_VARIABLE: "supplied"}


def test_cpu_selection_defaults_to_the_process_environment(monkeypatch) -> None:
    monkeypatch.setenv(PLATFORM_VARIABLE, "supplied")

    apply_device_selection("cpu")

    assert os.environ[PLATFORM_VARIABLE] == "cpu"


@pytest.mark.parametrize(
    ("device", "platform"),
    [
        ("auto", "cpu"),
        ("auto", "gpu"),
        ("cpu", "cpu"),
        ("gpu", "gpu"),
        ("gpu", "cuda"),
        ("gpu", "rocm"),
    ],
)
def test_satisfiable_requests_are_accepted(device: str, platform: str) -> None:
    verify_device_selection(device, platform)


@pytest.mark.parametrize("platform", ["cpu", "tpu", ""])
def test_a_requested_gpu_refuses_another_backend(platform: str) -> None:
    with pytest.raises(RuntimeError, match="requested device gpu"):
        verify_device_selection("gpu", platform)


def test_refusal_names_the_backend_that_was_bound() -> None:
    with pytest.raises(RuntimeError, match="bound backend is cpu"):
        verify_device_selection("gpu", "cpu")


def test_allocator_peak_is_read_from_the_device() -> None:
    device = _Device({"peak_bytes_in_use": 2**30, "bytes_in_use": 1})

    assert device_memory_peak_bytes(device) == 2**30


@pytest.mark.parametrize(
    "statistics", [None, {}, {"peak_bytes_in_use": None}, "unavailable"]
)
def test_backends_without_the_statistic_report_nothing(statistics: object) -> None:
    assert device_memory_peak_bytes(_Device(statistics)) is None


def test_a_device_without_statistics_reports_nothing() -> None:
    assert device_memory_peak_bytes(object()) is None
