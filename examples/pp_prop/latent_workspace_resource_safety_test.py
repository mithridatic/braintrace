"""Tests for Example 21 resource-safety assessments."""

from __future__ import annotations

import msgspec_json
import math

import pytest

try:
    import examples.pp_prop.latent_workspace_resource_safety as safety
except ModuleNotFoundError as error:
    if error.name not in {
        "examples",
        "examples.pp_prop",
        "examples.pp_prop.latent_workspace_resource_safety",
    }:
        raise
    import latent_workspace_resource_safety as safety


assess_gpu_memory_safety = safety.assess_gpu_memory_safety
assess_recurrent_edge_budget = safety.assess_recurrent_edge_budget
require_full_gpu_memory_safety = safety.require_full_gpu_memory_safety
require_recurrent_edge_budget = safety.require_recurrent_edge_budget
assess_pre_device_gpu_environment = safety.assess_pre_device_gpu_environment
assess_gpu_runtime_safety = safety.assess_gpu_runtime_safety
require_pre_device_gpu_environment = safety.require_pre_device_gpu_environment
require_full_gpu_runtime_safety = safety.require_full_gpu_runtime_safety


def test_edge_budget_accepts_exact_default_policy_cap() -> None:
    report = assess_recurrent_edge_budget(
        neuron_count=4_096,
        recurrent_edge_count=4_194_304,
    )

    assert report.safe
    assert report.max_edges_per_neuron == 1_024
    assert report.policy_edge_cap == 4_194_304
    assert report.no_self_edge_cap == 16_773_120
    assert report.edge_cap == 4_194_304
    assert report.edges_per_neuron == 1_024.0
    assert report.budget_utilization == 1.0
    assert report.violations == ()
    assert require_recurrent_edge_budget(4_096, 4_194_304) == report


def test_edge_budget_rejects_one_edge_over_policy_cap() -> None:
    report = assess_recurrent_edge_budget(4, 4_097)

    assert not report.safe
    assert report.edges_per_neuron == 1_024.25
    assert report.budget_utilization > 1.0
    assert "policy_edge_cap_exceeded" in report.violations
    with pytest.raises(safety.ResourceSafetyError, match="(?i)recurrent edge budget"):
        require_recurrent_edge_budget(4, 4_097)


def test_edge_budget_applies_no_self_capacity_when_it_is_tighter() -> None:
    at_capacity = assess_recurrent_edge_budget(3, 6)
    over_capacity = assess_recurrent_edge_budget(3, 7)

    assert at_capacity.safe
    assert at_capacity.policy_edge_cap == 3_072
    assert at_capacity.no_self_edge_cap == 6
    assert at_capacity.edge_cap == 6
    assert at_capacity.budget_utilization == 1.0
    assert not over_capacity.safe
    assert over_capacity.violations == ("no_self_edge_cap_exceeded",)


def test_zero_edges_for_one_neuron_has_zero_utilization() -> None:
    report = assess_recurrent_edge_budget(1, 0)

    assert report.safe
    assert report.edge_cap == 0
    assert report.edges_per_neuron == 0.0
    assert report.budget_utilization == 0.0


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"neuron_count": 0, "recurrent_edge_count": 0}, "neuron_count"),
        ({"neuron_count": True, "recurrent_edge_count": 0}, "neuron_count"),
        ({"neuron_count": 2, "recurrent_edge_count": -1}, "recurrent_edge_count"),
        ({"neuron_count": 2, "recurrent_edge_count": 1.5}, "recurrent_edge_count"),
        (
            {
                "neuron_count": 2,
                "recurrent_edge_count": 1,
                "max_edges_per_neuron": 0,
            },
            "max_edges_per_neuron",
        ),
    ],
)
def test_edge_budget_rejects_invalid_configuration(
    arguments: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        assess_recurrent_edge_budget(**arguments)  # type: ignore[arg-type]


def test_edge_report_is_json_safe() -> None:
    report = assess_recurrent_edge_budget(8, 32, max_edges_per_neuron=5)

    payload = report.to_dict()

    assert payload["safe"] is True
    assert payload["edge_cap"] == 40
    msgspec_json.dumps(payload, allow_nan=False)


def test_full_gpu_assessment_accepts_complete_evidence_within_limits() -> None:
    gib = 1024**3
    report = assess_gpu_memory_safety(
        run_scope="full",
        peak_device_bytes=2 * gib,
        physical_device_bytes=16 * gib,
        allocator_target_fraction=0.80,
    )

    assert report.status == "safe"
    assert report.evidence_complete
    assert report.within_limits
    assert report.full_qualification_safe
    assert report.observed_physical_fraction == 0.125
    assert report.violations == ()
    assert (
        require_full_gpu_memory_safety(
            peak_device_bytes=2 * gib,
            physical_device_bytes=16 * gib,
            allocator_target_fraction=0.80,
        )
        == report
    )


def test_full_gpu_assessment_accepts_physical_limit_boundary() -> None:
    report = assess_gpu_memory_safety(
        run_scope="full",
        peak_device_bytes=85,
        physical_device_bytes=100,
        allocator_target_fraction=0.85,
    )

    assert report.full_qualification_safe
    assert report.observed_physical_fraction == pytest.approx(0.85)


def test_full_gpu_assessment_rejects_peak_over_physical_limit() -> None:
    report = assess_gpu_memory_safety(
        run_scope="full",
        peak_device_bytes=86,
        physical_device_bytes=100,
        allocator_target_fraction=0.80,
    )

    assert report.status == "unsafe"
    assert report.evidence_complete
    assert not report.within_limits
    assert not report.full_qualification_safe
    assert report.violations == ("physical_memory_fraction_exceeded",)


def test_full_gpu_assessment_rejects_allocator_target_over_limit() -> None:
    report = assess_gpu_memory_safety(
        run_scope="full",
        peak_device_bytes=8,
        physical_device_bytes=100,
        allocator_target_fraction=0.850_001,
    )

    assert report.status == "unsafe"
    assert report.violations == ("allocator_target_fraction_exceeded",)


@pytest.mark.parametrize(
    ("peak", "physical", "allocator", "violation"),
    [
        (None, 100, 0.8, "peak_device_bytes_missing"),
        (10, None, 0.8, "physical_device_bytes_missing"),
        (10, 100, None, "allocator_target_fraction_missing"),
        (math.nan, 100, 0.8, "peak_device_bytes_invalid"),
        (10, math.inf, 0.8, "physical_device_bytes_invalid"),
        (10, 100, math.nan, "allocator_target_fraction_invalid"),
        (True, 100, 0.8, "peak_device_bytes_invalid"),
        (10, -100, 0.8, "physical_device_bytes_invalid"),
        (0, 100, 0.8, "peak_device_bytes_invalid"),
    ],
)
def test_full_gpu_assessment_fails_closed_for_missing_or_invalid_evidence(
    peak: object,
    physical: object,
    allocator: object,
    violation: str,
) -> None:
    report = assess_gpu_memory_safety(
        run_scope="full",
        peak_device_bytes=peak,
        physical_device_bytes=physical,
        allocator_target_fraction=allocator,
    )

    assert report.status == "insufficient_evidence"
    assert not report.evidence_complete
    assert not report.within_limits
    assert not report.full_qualification_safe
    assert violation in report.violations
    with pytest.raises(safety.ResourceSafetyError, match="(?i)full GPU qualification"):
        report.require_full_qualification_safe()


def test_peak_larger_than_physical_capacity_is_invalid_evidence() -> None:
    report = assess_gpu_memory_safety(
        run_scope="full",
        peak_device_bytes=101,
        physical_device_bytes=100,
        allocator_target_fraction=0.8,
    )

    assert report.status == "insufficient_evidence"
    assert not report.evidence_complete
    assert report.violations == ("peak_exceeds_physical_capacity",)


def test_smoke_run_is_distinct_from_full_qualification() -> None:
    report = assess_gpu_memory_safety(
        run_scope="smoke",
        peak_device_bytes=10,
        physical_device_bytes=100,
        allocator_target_fraction=0.8,
    )

    assert report.status == "smoke_within_limits"
    assert report.evidence_complete
    assert report.within_limits
    assert not report.full_qualification_safe
    with pytest.raises(safety.ResourceSafetyError, match="not a full qualification"):
        report.require_full_qualification_safe()


def test_smoke_run_with_missing_evidence_is_not_safe() -> None:
    report = assess_gpu_memory_safety(
        run_scope="smoke",
        peak_device_bytes=None,
        physical_device_bytes=None,
        allocator_target_fraction=None,
    )

    assert report.status == "smoke_insufficient_evidence"
    assert not report.within_limits
    assert not report.full_qualification_safe


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("run_scope", "benchmark", "run_scope"),
        ("max_physical_fraction", 0.0, "max_physical_fraction"),
        ("max_physical_fraction", 0.850_001, "max_physical_fraction"),
        ("max_physical_fraction", 1.01, "max_physical_fraction"),
        ("max_allocator_target_fraction", math.nan, "max_allocator"),
        ("max_allocator_target_fraction", 0.86, "max_allocator"),
    ],
)
def test_gpu_assessment_rejects_invalid_policy_configuration(
    keyword: str, value: object, message: str
) -> None:
    arguments: dict[str, object] = {
        "run_scope": "full",
        "peak_device_bytes": 10,
        "physical_device_bytes": 100,
        "allocator_target_fraction": 0.8,
    }
    arguments[keyword] = value

    with pytest.raises(ValueError, match=message):
        assess_gpu_memory_safety(**arguments)  # type: ignore[arg-type]


def test_gpu_report_is_json_safe_without_fabricating_missing_evidence() -> None:
    report = assess_gpu_memory_safety(
        run_scope="full",
        peak_device_bytes=None,
        physical_device_bytes=100,
        allocator_target_fraction=0.8,
    )

    payload = report.to_dict()

    assert payload["peak_device_bytes"] is None
    assert payload["observed_physical_fraction"] is None
    assert payload["full_qualification_safe"] is False
    msgspec_json.dumps(payload, allow_nan=False)


def test_pre_device_environment_accepts_recommended_allocator_target() -> None:
    environment = {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.80", "OTHER": "kept"}
    original = dict(environment)

    report = assess_pre_device_gpu_environment(environment)

    assert report.configured_fraction == 0.8
    assert report.recommended_fraction == 0.8
    assert report.max_fraction == 0.85
    assert report.uses_recommended_fraction
    assert report.safe
    assert report.violations == ()
    assert environment == original
    assert require_pre_device_gpu_environment(environment) == report


def test_pre_device_environment_accepts_exact_hard_limit() -> None:
    report = assess_pre_device_gpu_environment(
        {"XLA_PYTHON_CLIENT_MEM_FRACTION": " 0.85 "}
    )

    assert report.safe
    assert not report.uses_recommended_fraction


@pytest.mark.parametrize(
    ("environment", "violation"),
    [
        ({}, "xla_memory_fraction_missing"),
        ({"XLA_PYTHON_CLIENT_MEM_FRACTION": ""}, "xla_memory_fraction_invalid"),
        ({"XLA_PYTHON_CLIENT_MEM_FRACTION": "nope"}, "xla_memory_fraction_invalid"),
        ({"XLA_PYTHON_CLIENT_MEM_FRACTION": "nan"}, "xla_memory_fraction_invalid"),
        ({"XLA_PYTHON_CLIENT_MEM_FRACTION": "inf"}, "xla_memory_fraction_invalid"),
        ({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0"}, "xla_memory_fraction_invalid"),
        (
            {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.850001"},
            "xla_memory_fraction_exceeded",
        ),
    ],
)
def test_pre_device_environment_fails_closed(
    environment: dict[str, str], violation: str
) -> None:
    report = assess_pre_device_gpu_environment(environment)

    assert not report.safe
    assert violation in report.violations
    with pytest.raises(safety.ResourceSafetyError, match="(?i)pre-device GPU"):
        report.require_safe()


def test_pre_device_environment_rejects_loosened_policy() -> None:
    environment = {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8"}

    with pytest.raises(ValueError, match="max_fraction"):
        assess_pre_device_gpu_environment(environment, max_fraction=0.86)
    with pytest.raises(ValueError, match="recommended_fraction"):
        assess_pre_device_gpu_environment(environment, recommended_fraction=0.86)


def test_pre_device_report_is_json_safe() -> None:
    report = assess_pre_device_gpu_environment({})

    payload = report.to_dict()

    assert payload["configured_fraction"] is None
    assert payload["safe"] is False
    msgspec_json.dumps(payload, allow_nan=False)


def test_full_runtime_safety_normalizes_allocator_and_process_evidence() -> None:
    gib = 1024**3
    report = assess_gpu_runtime_safety(
        run_scope="full",
        environment={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.80"},
        allocator_peak_bytes=2 * gib,
        allocator_limit_bytes=12 * gib,
        physical_device_bytes=16 * gib,
        process_peak_bytes=3 * gib,
    )

    assert report.allocator_peak_bytes == 2 * gib
    assert report.allocator_limit_bytes == 12 * gib
    assert report.process_peak_bytes == 3 * gib
    assert report.normalized_peak_device_bytes == 3 * gib
    assert report.observed_allocator_fraction == pytest.approx(1 / 6)
    assert report.memory.peak_device_bytes == 3 * gib
    assert report.evidence_complete
    assert report.within_limits
    assert report.full_qualification_safe
    assert report.status == "safe"
    assert report.violations == ()
    assert (
        require_full_gpu_runtime_safety(
            environment={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.80"},
            allocator_peak_bytes=2 * gib,
            allocator_limit_bytes=12 * gib,
            physical_device_bytes=16 * gib,
            process_peak_bytes=3 * gib,
        )
        == report
    )


def test_runtime_safety_uses_larger_allocator_peak_for_physical_gate() -> None:
    report = assess_gpu_runtime_safety(
        run_scope="full",
        environment={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8"},
        allocator_peak_bytes=86,
        allocator_limit_bytes=90,
        physical_device_bytes=100,
        process_peak_bytes=20,
    )

    assert report.normalized_peak_device_bytes == 86
    assert report.status == "unsafe"
    assert report.violations == ("physical_memory_fraction_exceeded",)


@pytest.mark.parametrize(
    ("keyword", "value", "violation"),
    [
        ("allocator_peak_bytes", None, "allocator_peak_bytes_missing"),
        ("allocator_limit_bytes", None, "allocator_limit_bytes_missing"),
        ("process_peak_bytes", None, "process_peak_bytes_missing"),
        ("allocator_peak_bytes", math.nan, "allocator_peak_bytes_invalid"),
        ("allocator_limit_bytes", math.inf, "allocator_limit_bytes_invalid"),
        ("process_peak_bytes", -1, "process_peak_bytes_invalid"),
    ],
)
def test_full_runtime_safety_fails_closed_for_incomplete_runtime_evidence(
    keyword: str, value: object, violation: str
) -> None:
    arguments: dict[str, object] = {
        "run_scope": "full",
        "environment": {"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8"},
        "allocator_peak_bytes": 20,
        "allocator_limit_bytes": 80,
        "physical_device_bytes": 100,
        "process_peak_bytes": 30,
    }
    arguments[keyword] = value

    report = assess_gpu_runtime_safety(**arguments)  # type: ignore[arg-type]

    assert report.status == "insufficient_evidence"
    assert not report.evidence_complete
    assert not report.within_limits
    assert not report.full_qualification_safe
    assert violation in report.violations
    with pytest.raises(safety.ResourceSafetyError, match="(?i)full GPU runtime"):
        report.require_full_qualification_safe()


@pytest.mark.parametrize(
    ("allocator_peak", "allocator_limit", "physical", "violation"),
    [
        (81, 80, 100, "allocator_peak_exceeds_limit"),
        (20, 101, 100, "allocator_limit_exceeds_physical_capacity"),
    ],
)
def test_full_runtime_safety_rejects_incoherent_allocator_evidence(
    allocator_peak: int,
    allocator_limit: int,
    physical: int,
    violation: str,
) -> None:
    report = assess_gpu_runtime_safety(
        run_scope="full",
        environment={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8"},
        allocator_peak_bytes=allocator_peak,
        allocator_limit_bytes=allocator_limit,
        physical_device_bytes=physical,
        process_peak_bytes=30,
    )

    assert report.status == "insufficient_evidence"
    assert violation in report.violations


def test_runtime_safety_rejects_unsafe_environment_before_memory_evidence() -> None:
    report = assess_gpu_runtime_safety(
        run_scope="full",
        environment={},
        allocator_peak_bytes=20,
        allocator_limit_bytes=80,
        physical_device_bytes=100,
        process_peak_bytes=30,
    )

    assert report.status == "insufficient_evidence"
    assert report.violations == ("xla_memory_fraction_missing",)
    assert not report.full_qualification_safe


def test_runtime_safety_classifies_configured_fraction_over_limit_as_unsafe() -> None:
    report = assess_gpu_runtime_safety(
        run_scope="full",
        environment={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.86"},
        allocator_peak_bytes=20,
        allocator_limit_bytes=80,
        physical_device_bytes=100,
        process_peak_bytes=30,
    )

    assert report.evidence_complete
    assert report.status == "unsafe"
    assert report.violations == ("xla_memory_fraction_exceeded",)
    assert not report.full_qualification_safe


def test_smoke_runtime_evidence_can_never_qualify_full_run() -> None:
    report = assess_gpu_runtime_safety(
        run_scope="smoke",
        environment={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8"},
        allocator_peak_bytes=20,
        allocator_limit_bytes=80,
        physical_device_bytes=100,
        process_peak_bytes=30,
    )

    assert report.status == "smoke_within_limits"
    assert report.within_limits
    assert not report.full_qualification_safe
    with pytest.raises(safety.ResourceSafetyError, match="not a full qualification"):
        report.require_full_qualification_safe()


def test_runtime_safety_report_is_json_safe() -> None:
    report = assess_gpu_runtime_safety(
        run_scope="full",
        environment={"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.8"},
        allocator_peak_bytes=20,
        allocator_limit_bytes=80,
        physical_device_bytes=100,
        process_peak_bytes=30,
    )

    payload = report.to_dict()

    assert payload["environment"]["safe"] is True  # type: ignore[index]
    assert payload["memory"]["full_qualification_safe"] is True  # type: ignore[index]
    msgspec_json.dumps(payload, allow_nan=False)
