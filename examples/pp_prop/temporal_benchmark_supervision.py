"""Arm policies and construction-equivalence guards for Example 17."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from temporal_benchmark_config import ArmName


@dataclass(frozen=True)
class ArmPolicy:
    """Declare active recurrence and trainable parameter groups for one arm."""

    algorithm: str
    recurrence_module: bool
    recurrence_active: bool
    train_readout: bool
    train_feedforward: bool
    train_recurrent: bool


ARM_POLICIES: dict[ArmName, ArmPolicy] = {
    "readout_only": ArmPolicy("pp_prop", True, True, True, False, False),
    "feedforward_readout_recurrence_zero": ArmPolicy(
        "pp_prop", True, False, True, True, False
    ),
    "recurrent_readout": ArmPolicy("pp_prop", True, True, True, False, True),
    "all_pp_prop": ArmPolicy("pp_prop", True, True, True, True, True),
    "no_recurrent_module": ArmPolicy("pp_prop", False, False, True, True, False),
    "frozen_random_recurrence": ArmPolicy("pp_prop", True, True, True, True, False),
    "all_bptt": ArmPolicy("bptt", True, True, True, True, True),
}


def policy_for_arm(arm: ArmName) -> ArmPolicy:
    """Return the immutable policy for a benchmark arm."""
    return ARM_POLICIES[arm]


def parameter_group(path: tuple[object, ...]) -> str | None:
    """Map a BrainState parameter path onto a benchmark optimizer group."""
    names = {str(part) for part in path}
    if "readout" in names:
        return "readout"
    if "ff_syn" in names:
        return "feedforward"
    if "rec_syn" in names:
        return "recurrent"
    return None


def assert_construction_equivalence(
    recurrence_zero_logits: np.ndarray,
    no_recurrence_logits: np.ndarray,
    *,
    atol: float = 1e-7,
) -> None:
    """Require arms 2 and 5 to agree numerically."""
    np.testing.assert_allclose(
        recurrence_zero_logits, no_recurrence_logits, rtol=0.0, atol=atol
    )


def algorithm_label(policy: ArmPolicy) -> str:
    """Return an unambiguous public algorithm label."""
    return (
        "full_window_reverse_mode_bptt_oracle"
        if policy.algorithm == "bptt"
        else "single_step_pp_prop"
    )
