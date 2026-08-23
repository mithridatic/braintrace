"""Pure resource-safety assessments for Example 21 qualification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal, Mapping


DEFAULT_MAX_EDGES_PER_NEURON = 1_024
DEFAULT_MAX_PHYSICAL_MEMORY_FRACTION = 0.85
DEFAULT_MAX_ALLOCATOR_TARGET_FRACTION = 0.85
DEFAULT_ALLOCATOR_TARGET_FRACTION = 0.80
XLA_MEMORY_FRACTION_VARIABLE = "XLA_PYTHON_CLIENT_MEM_FRACTION"

RunScope = Literal["full", "smoke"]
GpuSafetyStatus = Literal[
    "safe",
    "unsafe",
    "insufficient_evidence",
    "smoke_within_limits",
    "smoke_over_limit",
    "smoke_insufficient_evidence",
]


class ResourceSafetyError(RuntimeError):
    """Signal that a requested resource-safety gate did not pass."""


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}. Set {name} to an integer greater than or equal to {minimum}."
        )
    result = int(value)
    if result < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}. Set {name} to an integer greater than or equal to {minimum}."
        )
    return result


def _fraction_policy(
    value: object,
    name: str,
    *,
    upper_bound: float = 1.0,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and in (0, {upper_bound}]. Set {name} to a finite value in (0, {upper_bound}].")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= upper_bound:
        raise ValueError(f"{name} must be finite and in (0, {upper_bound}]. Set {name} to a finite value in (0, {upper_bound}].")
    return result


def _positive_integer_evidence(
    value: object,
    name: str,
) -> tuple[int | None, str | None]:
    if value is None:
        return None, f"{name}_missing"
    if isinstance(value, bool) or not isinstance(value, Real):
        return None, f"{name}_invalid"
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0 or not numeric.is_integer():
        return None, f"{name}_invalid"
    return int(numeric), None


def _fraction_evidence(
    value: object,
    name: str,
) -> tuple[float | None, str | None]:
    if value is None:
        return None, f"{name}_missing"
    if isinstance(value, bool) or not isinstance(value, Real):
        return None, f"{name}_invalid"
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        return None, f"{name}_invalid"
    return result, None


@dataclass(frozen=True)
class RecurrentEdgeBudgetAssessment:
    """Describe a validated recurrent-edge budget decision.

    Parameters
    ----------
    neuron_count : int
        Number of recurrent neurons.
    recurrent_edge_count : int
        Number of directed recurrent edges.
    max_edges_per_neuron : int
        Policy limit on mean recurrent edges per neuron.
    policy_edge_cap : int
        Product of ``neuron_count`` and ``max_edges_per_neuron``.
    no_self_edge_cap : int
        Maximum directed edge count when self-connections are prohibited.
    edge_cap : int
        Effective cap, the smaller of the policy and no-self caps.
    edges_per_neuron : float
        Mean directed recurrent edges per neuron.
    budget_utilization : float or None
        Fraction of the effective cap used. This is ``None`` only when a
        positive edge count is compared with a zero no-self capacity.
    safe : bool
        Whether both policy and no-self constraints pass.
    violations : tuple of str
        Stable machine-readable violation codes.
    """

    neuron_count: int
    recurrent_edge_count: int
    max_edges_per_neuron: int
    policy_edge_cap: int
    no_self_edge_cap: int
    edge_cap: int
    edges_per_neuron: float
    budget_utilization: float | None
    safe: bool
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation of the assessment.

        Returns
        -------
        dict
            Edge counts, caps, utilization, decision, and violation codes.
        """
        return {
            "neuron_count": self.neuron_count,
            "recurrent_edge_count": self.recurrent_edge_count,
            "max_edges_per_neuron": self.max_edges_per_neuron,
            "policy_edge_cap": self.policy_edge_cap,
            "no_self_edge_cap": self.no_self_edge_cap,
            "edge_cap": self.edge_cap,
            "edges_per_neuron": self.edges_per_neuron,
            "budget_utilization": self.budget_utilization,
            "safe": self.safe,
            "violations": list(self.violations),
        }

    def require_safe(self) -> None:
        """Raise if the recurrent-edge budget is not safe.

        Raises
        ------
        ResourceSafetyError
            If either the policy cap or no-self capacity is exceeded.
        """
        if not self.safe:
            detail = ", ".join(self.violations)
            raise ResourceSafetyError(f"Recurrent edge budget is unsafe: {detail}. Fix the input condition named in the error, then rerun the operation.")


def assess_recurrent_edge_budget(
    neuron_count: int,
    recurrent_edge_count: int,
    *,
    max_edges_per_neuron: int = DEFAULT_MAX_EDGES_PER_NEURON,
) -> RecurrentEdgeBudgetAssessment:
    """Assess recurrent edges against policy and no-self graph capacity.

    Parameters
    ----------
    neuron_count : int
        Positive number of recurrent neurons.
    recurrent_edge_count : int
        Nonnegative number of directed recurrent edges.
    max_edges_per_neuron : int, default=1024
        Positive policy cap on average recurrent edges per neuron.

    Returns
    -------
    RecurrentEdgeBudgetAssessment
        Validated caps, utilization, and safety decision.

    Raises
    ------
    ValueError
        If a count or policy value is not an integer in its allowed range.
    """
    neurons = _integer(neuron_count, "neuron_count", minimum=1)
    edges = _integer(recurrent_edge_count, "recurrent_edge_count", minimum=0)
    per_neuron = _integer(
        max_edges_per_neuron,
        "max_edges_per_neuron",
        minimum=1,
    )
    policy_cap = neurons * per_neuron
    no_self_cap = neurons * (neurons - 1)
    edge_cap = min(policy_cap, no_self_cap)

    violations: list[str] = []
    if edges > policy_cap:
        violations.append("policy_edge_cap_exceeded")
    if edges > no_self_cap:
        violations.append("no_self_edge_cap_exceeded")

    if edge_cap > 0:
        utilization: float | None = edges / edge_cap
    elif edges == 0:
        utilization = 0.0
    else:
        utilization = None

    return RecurrentEdgeBudgetAssessment(
        neuron_count=neurons,
        recurrent_edge_count=edges,
        max_edges_per_neuron=per_neuron,
        policy_edge_cap=policy_cap,
        no_self_edge_cap=no_self_cap,
        edge_cap=edge_cap,
        edges_per_neuron=edges / neurons,
        budget_utilization=utilization,
        safe=not violations,
        violations=tuple(violations),
    )


def require_recurrent_edge_budget(
    neuron_count: int,
    recurrent_edge_count: int,
    *,
    max_edges_per_neuron: int = DEFAULT_MAX_EDGES_PER_NEURON,
) -> RecurrentEdgeBudgetAssessment:
    """Assess and enforce the recurrent-edge budget.

    Parameters
    ----------
    neuron_count : int
        Positive number of recurrent neurons.
    recurrent_edge_count : int
        Nonnegative number of directed recurrent edges.
    max_edges_per_neuron : int, default=1024
        Positive policy cap on average recurrent edges per neuron.

    Returns
    -------
    RecurrentEdgeBudgetAssessment
        The passing assessment.

    Raises
    ------
    ValueError
        If a count or policy value is invalid.
    ResourceSafetyError
        If the edge count exceeds either enforced cap.
    """
    report = assess_recurrent_edge_budget(
        neuron_count,
        recurrent_edge_count,
        max_edges_per_neuron=max_edges_per_neuron,
    )
    report.require_safe()
    return report


@dataclass(frozen=True)
class GpuMemorySafetyAssessment:
    """Describe GPU memory evidence and its qualification status.

    Parameters
    ----------
    run_scope : {"full", "smoke"}
        Whether the evidence comes from a full qualification or a smoke run.
    peak_device_bytes : int or None
        Measured peak GPU device bytes, normalized to ``None`` when invalid.
    physical_device_bytes : int or None
        Physical GPU capacity, normalized to ``None`` when invalid.
    allocator_target_fraction : float or None
        Requested allocator fraction, normalized to ``None`` when invalid.
    max_physical_fraction : float
        Maximum allowed ratio of peak bytes to physical bytes.
    max_allocator_target_fraction : float
        Maximum allowed allocator target fraction.
    observed_physical_fraction : float or None
        Measured peak-to-physical ratio when the byte evidence is coherent.
    evidence_complete : bool
        Whether every required measurement is present, finite, and valid.
    within_limits : bool
        Whether complete evidence satisfies both memory limits.
    full_qualification_safe : bool
        Whether a full run, specifically, has complete passing evidence.
    status : str
        Machine-readable result that preserves full versus smoke scope.
    violations : tuple of str
        Stable missing-evidence, invalid-evidence, or limit violation codes.
    """

    run_scope: RunScope
    peak_device_bytes: int | None
    physical_device_bytes: int | None
    allocator_target_fraction: float | None
    max_physical_fraction: float
    max_allocator_target_fraction: float
    observed_physical_fraction: float | None
    evidence_complete: bool
    within_limits: bool
    full_qualification_safe: bool
    status: GpuSafetyStatus
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation of the assessment.

        Returns
        -------
        dict
            Scope, normalized evidence, thresholds, decisions, and violations.
        """
        return {
            "run_scope": self.run_scope,
            "peak_device_bytes": self.peak_device_bytes,
            "physical_device_bytes": self.physical_device_bytes,
            "allocator_target_fraction": self.allocator_target_fraction,
            "max_physical_fraction": self.max_physical_fraction,
            "max_allocator_target_fraction": self.max_allocator_target_fraction,
            "observed_physical_fraction": self.observed_physical_fraction,
            "evidence_complete": self.evidence_complete,
            "within_limits": self.within_limits,
            "full_qualification_safe": self.full_qualification_safe,
            "status": self.status,
            "violations": list(self.violations),
        }

    def require_full_qualification_safe(self) -> None:
        """Raise unless complete full-run evidence passes both limits.

        Raises
        ------
        ResourceSafetyError
            If the assessment is a smoke run, lacks evidence, or exceeds a
            configured memory limit.
        """
        if self.run_scope != "full":
            raise ResourceSafetyError("Assessment is not a full qualification. Free resources or reduce the allocation.")
        if not self.full_qualification_safe:
            detail = ", ".join(self.violations)
            raise ResourceSafetyError(
                f"Full GPU qualification is not resource-safe: {detail}. Free resources or reduce the allocation."
            )


def assess_gpu_memory_safety(
    *,
    run_scope: RunScope,
    peak_device_bytes: object,
    physical_device_bytes: object,
    allocator_target_fraction: object,
    max_physical_fraction: float = DEFAULT_MAX_PHYSICAL_MEMORY_FRACTION,
    max_allocator_target_fraction: float = DEFAULT_MAX_ALLOCATOR_TARGET_FRACTION,
) -> GpuMemorySafetyAssessment:
    """Assess GPU memory evidence without treating missing data as safe.

    Parameters
    ----------
    run_scope : {"full", "smoke"}
        Evidence scope. A smoke result can be within limits but can never pass
        the full-qualification gate.
    peak_device_bytes : object
        Measured positive integer peak GPU device bytes, or ``None``.
    physical_device_bytes : object
        Positive integer physical GPU capacity in bytes, or ``None``.
    allocator_target_fraction : object
        Configured allocator fraction in ``(0, 1]``, or ``None``.
    max_physical_fraction : float, default=0.85
        Maximum peak-to-physical fraction for full qualification. This policy
        cannot be loosened above 0.85.
    max_allocator_target_fraction : float, default=0.85
        Maximum accepted allocator target fraction. This policy cannot be
        loosened above 0.85.

    Returns
    -------
    GpuMemorySafetyAssessment
        Normalized evidence, explicit scope, and a fail-closed decision.

    Raises
    ------
    ValueError
        If the run scope or a policy threshold is invalid. Invalid measurement
        evidence is reported as insufficient rather than raising.
    """
    if run_scope not in ("full", "smoke"):
        raise ValueError("run_scope must be 'full' or 'smoke'. Set run_scope to 'full' or 'smoke'.")
    physical_limit = _fraction_policy(
        max_physical_fraction,
        "max_physical_fraction",
        upper_bound=DEFAULT_MAX_PHYSICAL_MEMORY_FRACTION,
    )
    allocator_limit = _fraction_policy(
        max_allocator_target_fraction,
        "max_allocator_target_fraction",
        upper_bound=DEFAULT_MAX_ALLOCATOR_TARGET_FRACTION,
    )

    peak, peak_issue = _positive_integer_evidence(
        peak_device_bytes,
        "peak_device_bytes",
    )
    physical, physical_issue = _positive_integer_evidence(
        physical_device_bytes,
        "physical_device_bytes",
    )
    allocator, allocator_issue = _fraction_evidence(
        allocator_target_fraction,
        "allocator_target_fraction",
    )
    evidence_issues = [
        issue
        for issue in (peak_issue, physical_issue, allocator_issue)
        if issue is not None
    ]

    observed: float | None = None
    if peak is not None and physical is not None:
        if peak > physical:
            evidence_issues.append("peak_exceeds_physical_capacity")
        else:
            observed = peak / physical

    limit_violations: list[str] = []
    evidence_complete = not evidence_issues
    if evidence_complete:
        assert observed is not None
        assert allocator is not None
        if observed > physical_limit:
            limit_violations.append("physical_memory_fraction_exceeded")
        if allocator > allocator_limit:
            limit_violations.append("allocator_target_fraction_exceeded")

    violations = tuple([*evidence_issues, *limit_violations])
    within_limits = evidence_complete and not limit_violations
    full_safe = run_scope == "full" and within_limits
    if run_scope == "full":
        if not evidence_complete:
            status: GpuSafetyStatus = "insufficient_evidence"
        elif within_limits:
            status = "safe"
        else:
            status = "unsafe"
    elif not evidence_complete:
        status = "smoke_insufficient_evidence"
    elif within_limits:
        status = "smoke_within_limits"
    else:
        status = "smoke_over_limit"

    return GpuMemorySafetyAssessment(
        run_scope=run_scope,
        peak_device_bytes=peak,
        physical_device_bytes=physical,
        allocator_target_fraction=allocator,
        max_physical_fraction=physical_limit,
        max_allocator_target_fraction=allocator_limit,
        observed_physical_fraction=observed,
        evidence_complete=evidence_complete,
        within_limits=within_limits,
        full_qualification_safe=full_safe,
        status=status,
        violations=violations,
    )


def require_full_gpu_memory_safety(
    *,
    peak_device_bytes: object,
    physical_device_bytes: object,
    allocator_target_fraction: object,
    max_physical_fraction: float = DEFAULT_MAX_PHYSICAL_MEMORY_FRACTION,
    max_allocator_target_fraction: float = DEFAULT_MAX_ALLOCATOR_TARGET_FRACTION,
) -> GpuMemorySafetyAssessment:
    """Assess and enforce full GPU qualification memory safety.

    Parameters
    ----------
    peak_device_bytes : object
        Measured positive integer peak GPU device bytes.
    physical_device_bytes : object
        Positive integer physical GPU capacity in bytes.
    allocator_target_fraction : object
        Configured allocator fraction in ``(0, 1]``.
    max_physical_fraction : float, default=0.85
        Maximum peak-to-physical fraction, capped at 0.85.
    max_allocator_target_fraction : float, default=0.85
        Maximum allocator target fraction, capped at 0.85.

    Returns
    -------
    GpuMemorySafetyAssessment
        The passing full-run assessment.

    Raises
    ------
    ValueError
        If a policy threshold is invalid.
    ResourceSafetyError
        If evidence is missing or invalid, or either limit is exceeded.
    """
    report = assess_gpu_memory_safety(
        run_scope="full",
        peak_device_bytes=peak_device_bytes,
        physical_device_bytes=physical_device_bytes,
        allocator_target_fraction=allocator_target_fraction,
        max_physical_fraction=max_physical_fraction,
        max_allocator_target_fraction=max_allocator_target_fraction,
    )
    report.require_full_qualification_safe()
    return report


@dataclass(frozen=True)
class PreDeviceGpuEnvironmentAssessment:
    """Describe the pre-device XLA allocator environment gate.

    Parameters
    ----------
    variable_name : str
        Name of the XLA memory-fraction environment variable.
    raw_value : str or None
        Unmodified injected value, or ``None`` when absent.
    configured_fraction : float or None
        Parsed finite positive fraction, or ``None`` when parsing failed.
    recommended_fraction : float
        Recommended allocator target for the approved run.
    max_fraction : float
        Hard maximum accepted allocator target.
    uses_recommended_fraction : bool
        Whether the configured value equals the recommended target.
    safe : bool
        Whether a valid value is configured at or below the hard maximum.
    violations : tuple of str
        Stable missing, invalid, or over-limit violation codes.
    """

    variable_name: str
    raw_value: str | None
    configured_fraction: float | None
    recommended_fraction: float
    max_fraction: float
    uses_recommended_fraction: bool
    safe: bool
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation of the environment gate.

        Returns
        -------
        dict
            Injected setting, normalized fraction, policy, and decision.
        """
        return {
            "variable_name": self.variable_name,
            "raw_value": self.raw_value,
            "configured_fraction": self.configured_fraction,
            "recommended_fraction": self.recommended_fraction,
            "max_fraction": self.max_fraction,
            "uses_recommended_fraction": self.uses_recommended_fraction,
            "safe": self.safe,
            "violations": list(self.violations),
        }

    def require_safe(self) -> None:
        """Raise unless the allocator fraction is safely configured.

        Raises
        ------
        ResourceSafetyError
            If the injected environment omits, invalidates, or exceeds the
            XLA allocator fraction policy.
        """
        if not self.safe:
            detail = ", ".join(self.violations)
            raise ResourceSafetyError(
                f"Pre-device GPU environment is not resource-safe: {detail}. Fix the input condition named in the error, then rerun the operation."
            )


def assess_pre_device_gpu_environment(
    environment: Mapping[str, str],
    *,
    recommended_fraction: float = DEFAULT_ALLOCATOR_TARGET_FRACTION,
    max_fraction: float = DEFAULT_MAX_ALLOCATOR_TARGET_FRACTION,
) -> PreDeviceGpuEnvironmentAssessment:
    """Validate the XLA allocator setting before any JAX device access.

    This function only reads the injected mapping. It does not inspect or
    mutate the process environment and does not import or call JAX.

    Parameters
    ----------
    environment : mapping of str to str
        Environment snapshot supplied by the caller.
    recommended_fraction : float, default=0.80
        Recommended target for the approved run. It cannot exceed either the
        selected hard maximum or the immutable 0.85 policy ceiling.
    max_fraction : float, default=0.85
        Selected hard maximum. It cannot exceed 0.85.

    Returns
    -------
    PreDeviceGpuEnvironmentAssessment
        Parsed setting and a fail-closed decision.

    Raises
    ------
    ValueError
        If a policy fraction is invalid or attempts to loosen the 0.85 ceiling.
    """
    hard_limit = _fraction_policy(
        max_fraction,
        "max_fraction",
        upper_bound=DEFAULT_MAX_ALLOCATOR_TARGET_FRACTION,
    )
    recommended = _fraction_policy(
        recommended_fraction,
        "recommended_fraction",
        upper_bound=DEFAULT_MAX_ALLOCATOR_TARGET_FRACTION,
    )
    if recommended > hard_limit:
        raise ValueError("recommended_fraction cannot exceed max_fraction. Fix the input condition named in the error, then rerun the operation.")

    injected = environment.get(XLA_MEMORY_FRACTION_VARIABLE)
    raw_value = injected if isinstance(injected, str) else None
    configured: float | None = None
    violations: list[str] = []
    if injected is None:
        violations.append("xla_memory_fraction_missing")
    elif not isinstance(injected, str):
        violations.append("xla_memory_fraction_invalid")
    else:
        try:
            parsed = float(injected.strip())
        except ValueError:
            parsed = math.nan
        if not math.isfinite(parsed) or parsed <= 0.0 or parsed > 1.0:
            violations.append("xla_memory_fraction_invalid")
        else:
            configured = parsed
            if configured > hard_limit:
                violations.append("xla_memory_fraction_exceeded")

    return PreDeviceGpuEnvironmentAssessment(
        variable_name=XLA_MEMORY_FRACTION_VARIABLE,
        raw_value=raw_value,
        configured_fraction=configured,
        recommended_fraction=recommended,
        max_fraction=hard_limit,
        uses_recommended_fraction=configured == recommended,
        safe=not violations,
        violations=tuple(violations),
    )


def require_pre_device_gpu_environment(
    environment: Mapping[str, str],
    *,
    recommended_fraction: float = DEFAULT_ALLOCATOR_TARGET_FRACTION,
    max_fraction: float = DEFAULT_MAX_ALLOCATOR_TARGET_FRACTION,
) -> PreDeviceGpuEnvironmentAssessment:
    """Enforce the XLA allocator setting before JAX device work.

    Parameters
    ----------
    environment : mapping of str to str
        Environment snapshot supplied by the caller.
    recommended_fraction : float, default=0.80
        Recommended allocator target.
    max_fraction : float, default=0.85
        Selected maximum, capped by the immutable 0.85 policy.

    Returns
    -------
    PreDeviceGpuEnvironmentAssessment
        The passing pre-device assessment.

    Raises
    ------
    ValueError
        If a policy fraction is invalid.
    ResourceSafetyError
        If the setting is missing, invalid, or above the hard maximum.
    """
    report = assess_pre_device_gpu_environment(
        environment,
        recommended_fraction=recommended_fraction,
        max_fraction=max_fraction,
    )
    report.require_safe()
    return report


@dataclass(frozen=True)
class GpuRuntimeSafetyAssessment:
    """Combine allocator and process evidence into one GPU safety gate.

    Parameters
    ----------
    run_scope : {"full", "smoke"}
        Whether evidence covers the full qualification or a smoke run.
    environment : PreDeviceGpuEnvironmentAssessment
        Pre-device XLA allocator configuration decision.
    memory : GpuMemorySafetyAssessment
        Existing physical-memory assessment using the larger runtime peak.
    allocator_peak_bytes : int or None
        JAX allocator peak, normalized to ``None`` when invalid.
    allocator_limit_bytes : int or None
        JAX allocator byte limit, normalized to ``None`` when invalid.
    process_peak_bytes : int or None
        Peak GPU process use, normalized to ``None`` when invalid.
    normalized_peak_device_bytes : int or None
        Larger of valid allocator and process peaks.
    observed_allocator_fraction : float or None
        Allocator peak divided by allocator limit when both are valid.
    evidence_complete : bool
        Whether both evidence sources are complete and coherent.
    within_limits : bool
        Whether complete evidence and the pre-device gate satisfy policy.
    full_qualification_safe : bool
        Whether complete passing evidence comes from a full run.
    status : str
        Machine-readable scope-aware result.
    violations : tuple of str
        Stable environment, evidence, coherence, and limit codes.
    """

    run_scope: RunScope
    environment: PreDeviceGpuEnvironmentAssessment
    memory: GpuMemorySafetyAssessment
    allocator_peak_bytes: int | None
    allocator_limit_bytes: int | None
    process_peak_bytes: int | None
    normalized_peak_device_bytes: int | None
    observed_allocator_fraction: float | None
    evidence_complete: bool
    within_limits: bool
    full_qualification_safe: bool
    status: GpuSafetyStatus
    violations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe runtime qualification report.

        Returns
        -------
        dict
            Nested environment and memory reports plus normalized evidence.
        """
        return {
            "run_scope": self.run_scope,
            "environment": self.environment.to_dict(),
            "memory": self.memory.to_dict(),
            "allocator_peak_bytes": self.allocator_peak_bytes,
            "allocator_limit_bytes": self.allocator_limit_bytes,
            "process_peak_bytes": self.process_peak_bytes,
            "normalized_peak_device_bytes": self.normalized_peak_device_bytes,
            "observed_allocator_fraction": self.observed_allocator_fraction,
            "evidence_complete": self.evidence_complete,
            "within_limits": self.within_limits,
            "full_qualification_safe": self.full_qualification_safe,
            "status": self.status,
            "violations": list(self.violations),
        }

    def require_full_qualification_safe(self) -> None:
        """Raise unless full-run runtime evidence passes every gate.

        Raises
        ------
        ResourceSafetyError
            If evidence is smoke-only, incomplete, incoherent, or over limit.
        """
        if self.run_scope != "full":
            raise ResourceSafetyError("Assessment is not a full qualification. Free resources or reduce the allocation.")
        if not self.full_qualification_safe:
            detail = ", ".join(self.violations)
            raise ResourceSafetyError(
                f"Full GPU runtime qualification is not resource-safe: {detail}. Free resources or reduce the allocation."
            )


def _append_unique(target: list[str], values: tuple[str, ...] | list[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def assess_gpu_runtime_safety(
    *,
    run_scope: RunScope,
    environment: Mapping[str, str],
    allocator_peak_bytes: object,
    allocator_limit_bytes: object,
    physical_device_bytes: object,
    process_peak_bytes: object,
    recommended_allocator_target_fraction: float = DEFAULT_ALLOCATOR_TARGET_FRACTION,
    max_physical_fraction: float = DEFAULT_MAX_PHYSICAL_MEMORY_FRACTION,
    max_allocator_target_fraction: float = DEFAULT_MAX_ALLOCATOR_TARGET_FRACTION,
) -> GpuRuntimeSafetyAssessment:
    """Assess full runtime GPU evidence without silently filling gaps.

    The larger of the JAX allocator peak and the sampled GPU process peak is
    normalized into :func:`assess_gpu_memory_safety`. Both independent sources
    remain mandatory for a full qualification.

    Parameters
    ----------
    run_scope : {"full", "smoke"}
        Scope covered by the supplied evidence.
    environment : mapping of str to str
        Pre-device environment snapshot.
    allocator_peak_bytes : object
        Peak live bytes reported by the JAX allocator.
    allocator_limit_bytes : object
        Byte limit reported by the JAX allocator.
    physical_device_bytes : object
        Physical GPU capacity reported by the device monitor.
    process_peak_bytes : object
        Peak GPU bytes attributed to the qualification process.
    recommended_allocator_target_fraction : float, default=0.80
        Recommended XLA allocator fraction.
    max_physical_fraction : float, default=0.85
        Hard observed-use limit, capped at 0.85.
    max_allocator_target_fraction : float, default=0.85
        Hard XLA allocator target, capped at 0.85.

    Returns
    -------
    GpuRuntimeSafetyAssessment
        Normalized, scope-aware, fail-closed report.

    Raises
    ------
    ValueError
        If scope or a policy threshold is invalid.
    """
    if run_scope not in ("full", "smoke"):
        raise ValueError("run_scope must be 'full' or 'smoke'. Set run_scope to 'full' or 'smoke'.")
    environment_report = assess_pre_device_gpu_environment(
        environment,
        recommended_fraction=recommended_allocator_target_fraction,
        max_fraction=max_allocator_target_fraction,
    )
    allocator_peak, allocator_peak_issue = _positive_integer_evidence(
        allocator_peak_bytes,
        "allocator_peak_bytes",
    )
    allocator_limit, allocator_limit_issue = _positive_integer_evidence(
        allocator_limit_bytes,
        "allocator_limit_bytes",
    )
    process_peak, process_peak_issue = _positive_integer_evidence(
        process_peak_bytes,
        "process_peak_bytes",
    )
    physical, physical_issue = _positive_integer_evidence(
        physical_device_bytes,
        "physical_device_bytes",
    )
    violations: list[str] = list(environment_report.violations)
    _append_unique(
        violations,
        [
            issue
            for issue in (
                allocator_peak_issue,
                allocator_limit_issue,
                process_peak_issue,
                physical_issue,
            )
            if issue is not None
        ],
    )
    runtime_evidence_complete = all(
        issue is None
        for issue in (
            allocator_peak_issue,
            allocator_limit_issue,
            process_peak_issue,
            physical_issue,
        )
    )
    allocator_evidence_coherent = True

    if allocator_peak is not None and allocator_limit is not None:
        observed_allocator_fraction: float | None = allocator_peak / allocator_limit
        if allocator_peak > allocator_limit:
            violations.append("allocator_peak_exceeds_limit")
            allocator_evidence_coherent = False
    else:
        observed_allocator_fraction = None
    if allocator_limit is not None and physical is not None:
        if allocator_limit > physical:
            violations.append("allocator_limit_exceeds_physical_capacity")
            allocator_evidence_coherent = False

    valid_peaks = [peak for peak in (allocator_peak, process_peak) if peak is not None]
    normalized_peak = max(valid_peaks) if valid_peaks else None
    memory = assess_gpu_memory_safety(
        run_scope=run_scope,
        peak_device_bytes=normalized_peak,
        physical_device_bytes=physical,
        allocator_target_fraction=environment_report.configured_fraction,
        max_physical_fraction=max_physical_fraction,
        max_allocator_target_fraction=max_allocator_target_fraction,
    )
    preliminary_complete = (
        environment_report.configured_fraction is not None
        and runtime_evidence_complete
        and allocator_evidence_coherent
    )
    if preliminary_complete:
        _append_unique(
            violations,
            [
                violation
                for violation in memory.violations
                if violation != "allocator_target_fraction_exceeded"
            ],
        )
    evidence_complete = preliminary_complete and memory.evidence_complete
    within_limits = (
        evidence_complete and environment_report.safe and memory.within_limits
    )
    full_safe = run_scope == "full" and within_limits
    if run_scope == "full":
        if not evidence_complete:
            status: GpuSafetyStatus = "insufficient_evidence"
        elif within_limits:
            status = "safe"
        else:
            status = "unsafe"
    elif not evidence_complete:
        status = "smoke_insufficient_evidence"
    elif within_limits:
        status = "smoke_within_limits"
    else:
        status = "smoke_over_limit"

    return GpuRuntimeSafetyAssessment(
        run_scope=run_scope,
        environment=environment_report,
        memory=memory,
        allocator_peak_bytes=allocator_peak,
        allocator_limit_bytes=allocator_limit,
        process_peak_bytes=process_peak,
        normalized_peak_device_bytes=normalized_peak,
        observed_allocator_fraction=observed_allocator_fraction,
        evidence_complete=evidence_complete,
        within_limits=within_limits,
        full_qualification_safe=full_safe,
        status=status,
        violations=tuple(violations),
    )


def require_full_gpu_runtime_safety(
    *,
    environment: Mapping[str, str],
    allocator_peak_bytes: object,
    allocator_limit_bytes: object,
    physical_device_bytes: object,
    process_peak_bytes: object,
    recommended_allocator_target_fraction: float = DEFAULT_ALLOCATOR_TARGET_FRACTION,
    max_physical_fraction: float = DEFAULT_MAX_PHYSICAL_MEMORY_FRACTION,
    max_allocator_target_fraction: float = DEFAULT_MAX_ALLOCATOR_TARGET_FRACTION,
) -> GpuRuntimeSafetyAssessment:
    """Assess and enforce the complete full-run GPU runtime gate.

    Parameters
    ----------
    environment : mapping of str to str
        Pre-device environment snapshot.
    allocator_peak_bytes : object
        Peak live bytes reported by the JAX allocator.
    allocator_limit_bytes : object
        Byte limit reported by the JAX allocator.
    physical_device_bytes : object
        Physical GPU capacity reported by the device monitor.
    process_peak_bytes : object
        Peak GPU bytes attributed to the qualification process.
    recommended_allocator_target_fraction : float, default=0.80
        Recommended XLA allocator fraction.
    max_physical_fraction : float, default=0.85
        Hard observed-use limit, capped at 0.85.
    max_allocator_target_fraction : float, default=0.85
        Hard allocator target, capped at 0.85.

    Returns
    -------
    GpuRuntimeSafetyAssessment
        The passing full-run assessment.

    Raises
    ------
    ValueError
        If a policy value is invalid.
    ResourceSafetyError
        If any pre-device, evidence, coherence, scope, or memory gate fails.
    """
    report = assess_gpu_runtime_safety(
        run_scope="full",
        environment=environment,
        allocator_peak_bytes=allocator_peak_bytes,
        allocator_limit_bytes=allocator_limit_bytes,
        physical_device_bytes=physical_device_bytes,
        process_peak_bytes=process_peak_bytes,
        recommended_allocator_target_fraction=(recommended_allocator_target_fraction),
        max_physical_fraction=max_physical_fraction,
        max_allocator_target_fraction=max_allocator_target_fraction,
    )
    report.require_full_qualification_safe()
    return report
