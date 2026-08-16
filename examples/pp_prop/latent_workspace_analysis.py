"""One-shot latent-geometry and linear-probe analysis for Example 21."""

from __future__ import annotations

from numbers import Integral, Real
from typing import Sequence

import numpy as np


_HUTCHINSON_PROBE_COUNT = 16
_CG_MIN_ITERATIONS = 64
_CG_ITERATION_FACTOR = 2
_CG_MAX_ITERATIONS = 2048
_CG_RELATIVE_TOLERANCE = 1e-8
_CG_RESIDUAL_ACCEPTANCE = 5e-8
_SPLITMIX_INCREMENT = np.uint64(0x9E3779B97F4A7C15)
_SPLITMIX_MULTIPLIER_1 = np.uint64(0xBF58476D1CE4E5B9)
_SPLITMIX_MULTIPLIER_2 = np.uint64(0x94D049BB133111EB)


def _float_array(value: np.ndarray, name: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular numeric array") from error


def _latent_array(states: np.ndarray) -> np.ndarray:
    array = _float_array(states, "states")
    if array.ndim != 3 or any(size == 0 for size in array.shape):
        raise ValueError(
            "states shape must be (episodes, iterations, width) with nonempty axes; "
            f"got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"states shape {array.shape} contains non-finite values")
    return array


def _feature_array(features: np.ndarray, name: str = "features") -> np.ndarray:
    array = _float_array(features, name)
    if array.ndim != 2 or any(size == 0 for size in array.shape):
        raise ValueError(
            f"{name} shape must be (episodes, width) with nonempty axes; "
            f"got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} shape {array.shape} contains non-finite values")
    return array


def _factor_array(
    factors: np.ndarray,
    name: str,
    state_shape: tuple[int, int, int],
) -> np.ndarray:
    array = _float_array(factors, name)
    episode_count, _, width = state_shape
    if (
        array.ndim != 3
        or any(size == 0 for size in array.shape)
        or array.shape[0] != episode_count
        or array.shape[2] != width
    ):
        raise ValueError(
            f"{name} shape must be (episodes, slots, width) and match states; "
            f"got {name} {array.shape}, states {state_shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} shape {array.shape} contains non-finite values")
    return array


def _class_count(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"class_count must be an integer of at least 2; got {value!r}")
    count = int(value)
    if count < 2:
        raise ValueError(f"class_count must be at least 2; got {count}")
    return count


def _label_array(
    labels: np.ndarray,
    episode_count: int,
    class_count: int,
    name: str = "labels",
    reference_name: str = "features",
) -> np.ndarray:
    try:
        array = np.asarray(labels)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a rectangular integer array") from error
    if array.ndim not in (1, 2) or array.shape[0] != episode_count:
        raise ValueError(
            f"{name} shape must be (episodes,) or (episodes, outputs) and match "
            f"{reference_name}; got {name} {array.shape}, "
            f"{reference_name} episodes {episode_count}"
        )
    if array.ndim == 2 and array.shape[1] == 0:
        raise ValueError(f"{name} shape {array.shape} has no output symbols")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(
            f"{name} shape {array.shape} must contain integer class indices"
        )
    integer_labels = array.astype(np.int64, copy=False)
    if np.any(integer_labels < 0) or np.any(integer_labels >= class_count):
        raise ValueError(
            f"{name} classes must be in [0, {class_count}); got shape {array.shape}"
        )
    return integer_labels


def _index_array(
    name: str, indices: Sequence[int] | np.ndarray, episode_count: int
) -> np.ndarray:
    array = np.asarray(indices)
    if array.ndim != 1:
        raise ValueError(f"{name} shape must be one-dimensional; got {array.shape}")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty; got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise ValueError(f"{name} must contain integer episode indices")
    result = array.astype(np.intp, copy=False)
    if np.any(result < 0) or np.any(result >= episode_count):
        raise ValueError(
            f"{name} contains an index outside episode_count {episode_count}"
        )
    if np.unique(result).size != result.size:
        raise ValueError(f"{name} contains duplicate episode indices")
    return result


def _probe_split(
    fit_indices: Sequence[int] | np.ndarray,
    score_indices: Sequence[int] | np.ndarray,
    episode_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    fit = _index_array("fit_indices", fit_indices, episode_count)
    score = _index_array("score_indices", score_indices, episode_count)
    overlap = np.intersect1d(fit, score)
    if overlap.size:
        raise ValueError(
            "fit_indices and score_indices must be disjoint; "
            f"overlap={overlap.tolist()}"
        )
    return fit, score


def _splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        mixed = values + _SPLITMIX_INCREMENT
        mixed = (mixed ^ (mixed >> np.uint64(30))) * _SPLITMIX_MULTIPLIER_1
        mixed = (mixed ^ (mixed >> np.uint64(27))) * _SPLITMIX_MULTIPLIER_2
    return mixed ^ (mixed >> np.uint64(31))


def _fixed_rademacher_probes(width: int) -> np.ndarray:
    rows = np.arange(_HUTCHINSON_PROBE_COUNT, dtype=np.uint64)[:, None]
    columns = np.arange(width, dtype=np.uint64)[None, :]
    if width <= _HUTCHINSON_PROBE_COUNT:
        parity = np.bitwise_and(rows, columns)
        for shift in (32, 16, 8, 4, 2, 1):
            parity = np.bitwise_xor(parity, np.right_shift(parity, shift))
        bits = np.bitwise_and(parity, 1)
    else:
        with np.errstate(over="ignore"):
            keys = columns + rows * _SPLITMIX_INCREMENT
        bits = _splitmix64(keys) >> np.uint64(63)
    return np.where(bits == 0, 1.0, -1.0)


def participation_ratio(states: np.ndarray) -> np.ndarray:
    """Estimate effective dimensionality at every latent iteration.

    Parameters
    ----------
    states : numpy.ndarray
        Latent states with shape ``(episodes, iterations, width)``.

    Returns
    -------
    numpy.ndarray
        Participation ratios with shape ``(iterations,)``. The denominator is
        a deterministic 16-probe Hutchinson estimate of ``trace(C**2)``; no
        random state is consumed and no width-by-width covariance is formed.
        Widths up to 16 use a complete orthogonal probe set. A state with no
        variation across episodes has participation ratio zero. Wider states
        use a fixed SplitMix64-derived sketch. Like every fixed finite sketch
        below the state width, it has a nontrivial nullspace and is an estimate,
        not a rank oracle.

    Raises
    ------
    ValueError
        If ``states`` has the wrong shape or contains non-finite values.
    """
    array = _latent_array(states)
    scale = np.max(np.abs(array), axis=(0, 2), keepdims=True)
    scaled = np.divide(array, scale, out=np.zeros_like(array), where=scale > 0.0)
    centered = scaled - np.mean(scaled, axis=0, keepdims=True)
    probes = _fixed_rademacher_probes(array.shape[2])
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        trace = np.sum(np.square(centered), axis=(0, 2))
        projected = np.einsum("eiw,pw->eip", centered, probes, optimize=True)
        covariance_probes = np.einsum(
            "eiw,eip->iwp", centered, projected, optimize=True
        )
        trace_square = np.mean(np.sum(np.square(covariance_probes), axis=1), axis=1)
        result = np.square(
            np.divide(
                trace,
                np.sqrt(trace_square),
                out=np.zeros_like(trace),
                where=trace_square > 0.0,
            )
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(
            f"states shape {array.shape} produced non-finite participation_ratio"
        )
    maximum_rank = min(max(array.shape[0] - 1, 0), array.shape[2])
    return np.clip(result, 0.0, float(maximum_rank))


def trajectory_step_norm(states: np.ndarray) -> np.ndarray:
    """Calculate mean episode-wise latent change at every iteration.

    Parameters
    ----------
    states : numpy.ndarray
        Latent states with shape ``(episodes, iterations, width)``.

    Returns
    -------
    numpy.ndarray
        Mean Euclidean step norms with shape ``(iterations,)``. Index zero is
        defined as zero because no preceding latent state exists.

    Raises
    ------
    ValueError
        If ``states`` has the wrong shape or contains non-finite values.
    """
    array = _latent_array(states)
    result = np.zeros(array.shape[1], dtype=np.float64)
    if array.shape[1] > 1:
        with np.errstate(over="ignore", invalid="ignore"):
            differences = np.diff(array, axis=1)
            result[1:] = np.mean(np.linalg.norm(differences, axis=2), axis=0)
    if not np.all(np.isfinite(result)):
        raise ValueError(
            f"states shape {array.shape} produced non-finite trajectory_step_norm"
        )
    return result


def _ridge_coefficient(ridge: float) -> float:
    if isinstance(ridge, (bool, np.bool_)) or not isinstance(ridge, Real):
        raise ValueError(f"ridge must be a positive finite number; got {ridge!r}")
    coefficient = float(ridge)
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError(f"ridge must be positive and finite; got {coefficient}")
    return coefficient


def _matrix_free_ridge_weights(
    centered_features: np.ndarray,
    centered_targets: np.ndarray,
    coefficient: float,
    feature_name: str,
) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        right_hand_side = centered_features.T @ centered_targets
    if not np.all(np.isfinite(right_hand_side)):
        raise ValueError(f"{feature_name} produced a non-finite probe system")

    weights = np.zeros_like(right_hand_side)
    residual = right_hand_side.copy()
    direction = residual.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        initial_residual_square = np.sum(np.square(residual), axis=0)
    if not np.all(np.isfinite(initial_residual_square)):
        raise ValueError(f"{feature_name} produced a non-finite probe residual")
    residual_square = initial_residual_square.copy()
    threshold = np.square(_CG_RELATIVE_TOLERANCE) * np.maximum(
        initial_residual_square, np.finfo(np.float64).tiny
    )
    active = residual_square > threshold
    effective_dimension = min(centered_features.shape)
    iteration_limit = min(
        _CG_MAX_ITERATIONS,
        max(_CG_MIN_ITERATIONS, _CG_ITERATION_FACTOR * effective_dimension),
    )

    for _ in range(iteration_limit):
        if not np.any(active):
            break
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            projected = centered_features @ direction
            action = centered_features.T @ projected + coefficient * direction
            denominator = np.sum(direction * action, axis=0)
        if np.any(~np.isfinite(denominator[active])) or np.any(
            denominator[active] <= 0.0
        ):
            raise ValueError(f"{feature_name} produced an invalid probe system")
        step = np.divide(
            residual_square,
            denominator,
            out=np.zeros_like(residual_square),
            where=active,
        )
        weights += direction * step
        residual -= action * step
        with np.errstate(over="ignore", invalid="ignore"):
            next_square = np.sum(np.square(residual), axis=0)
        next_active = next_square > threshold
        ratio = np.divide(
            next_square,
            residual_square,
            out=np.zeros_like(next_square),
            where=active & (residual_square > 0.0),
        )
        direction = residual + direction * ratio
        direction[:, ~next_active] = 0.0
        residual_square = next_square
        active = next_active

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected = centered_features @ weights
        checked_action = centered_features.T @ projected + coefficient * weights
        checked_residual = right_hand_side - checked_action
        checked_norm = np.sqrt(np.sum(np.square(checked_residual), axis=0))
        initial_norm = np.sqrt(initial_residual_square)
        relative_residual = np.divide(
            checked_norm,
            initial_norm,
            out=np.zeros_like(checked_norm),
            where=initial_norm > 0.0,
        )
    if not np.all(np.isfinite(relative_residual)):
        raise ValueError(f"{feature_name} produced a non-finite probe residual")
    maximum_residual = float(np.max(relative_residual, initial=0.0))
    if maximum_residual > _CG_RESIDUAL_ACCEPTANCE:
        raise ValueError(
            f"{feature_name} ridge probe failed to reach bounded relative residual "
            f"{_CG_RESIDUAL_ACCEPTANCE:.1e} within {iteration_limit} matrix-free "
            f"iterations; maximum residual was {maximum_residual:.3e}"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError(f"{feature_name} produced non-finite probe weights")
    return weights


def _linear_probe_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    fit_indices: Sequence[int] | np.ndarray,
    score_indices: Sequence[int] | np.ndarray,
    class_count: int,
    ridge: float = 1e-6,
    *,
    feature_name: str = "features",
    label_name: str = "labels",
) -> np.ndarray:
    feature_array = _feature_array(features, feature_name)
    count = _class_count(class_count)
    label_array = _label_array(
        labels,
        feature_array.shape[0],
        count,
        label_name,
        feature_name,
    )
    fit, score = _probe_split(fit_indices, score_indices, feature_array.shape[0])
    coefficient = _ridge_coefficient(ridge)
    target_shape = label_array.shape[1:] + (count,)
    target_classes = np.arange(count, dtype=np.int64)
    targets = np.equal(label_array[fit][..., None], target_classes).astype(
        np.float64, copy=False
    )
    targets = targets.reshape(fit.size, -1)

    with np.errstate(over="ignore", invalid="ignore"):
        feature_mean = np.mean(feature_array[fit], axis=0, keepdims=True)
        target_mean = np.mean(targets, axis=0, keepdims=True)
        centered_features = feature_array[fit] - feature_mean
        centered_targets = targets - target_mean
    if not np.all(np.isfinite(centered_features)):
        raise ValueError(f"{feature_name} produced non-finite centered features")
    weights = _matrix_free_ridge_weights(
        centered_features, centered_targets, coefficient, feature_name
    )
    with np.errstate(over="ignore", invalid="ignore"):
        scores = (feature_array[score] - feature_mean) @ weights + target_mean
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{feature_name} produced non-finite probe scores")
    return np.argmax(scores.reshape((score.size,) + target_shape), axis=-1)


def linear_probe_accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    fit_indices: Sequence[int] | np.ndarray,
    score_indices: Sequence[int] | np.ndarray,
    class_count: int,
    ridge: float = 1e-6,
) -> float:
    """Fit a deterministic ridge classifier and score held-out episodes.

    One-dimensional labels produce ordinary answer accuracy. Two-dimensional
    labels are decoded independently along the second axis, and the returned
    score is mean per-symbol accuracy rather than permutation-class accuracy.

    Parameters
    ----------
    features : numpy.ndarray
        Probe inputs with shape ``(episodes, width)``.
    labels : numpy.ndarray
        Integer labels with shape ``(episodes,)`` or ``(episodes, outputs)``.
    fit_indices : sequence of int
        Unique episode indices used only to fit the probe.
    score_indices : sequence of int
        Unique episode indices used only to score the probe.
    class_count : int
        Number of target classes.
    ridge : float, default=1e-6
        Positive L2 coefficient applied to weights but not the intercept.

    Returns
    -------
    float
        Held-out element-wise classification accuracy.

    Raises
    ------
    ValueError
        If shapes, labels, classes, the ridge, or the disjoint split are invalid.
    """
    feature_array = _feature_array(features)
    count = _class_count(class_count)
    label_array = _label_array(labels, feature_array.shape[0], count)
    _, score = _probe_split(fit_indices, score_indices, feature_array.shape[0])
    predictions = _linear_probe_predictions(
        feature_array,
        label_array,
        fit_indices,
        score_indices,
        count,
        ridge,
    )
    return float(np.mean(predictions == label_array[score]))


def _accuracy_value(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [0, 1]; got {value!r}")
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]; got {result}")
    return result


def memory_final_comparison(memory_accuracy: float, final_accuracy: float) -> str:
    """Describe whether latent iteration improved answer decodability.

    Parameters
    ----------
    memory_accuracy : float
        Held-out answer decodability from the contextual-memory read.
    final_accuracy : float
        Held-out answer decodability from the final workspace state.

    Returns
    -------
    str
        A plain-English comparison that retains a null separation result.

    Raises
    ------
    ValueError
        If either accuracy is non-finite or outside the unit interval.
    """
    memory = _accuracy_value(memory_accuracy, "memory_accuracy")
    final = _accuracy_value(final_accuracy, "final_accuracy")
    if memory >= final:
        return (
            f"Memory-only answer decodability ({memory:.3f}) matched or exceeded "
            f"final-workspace decodability ({final:.3f}); the two-state separation "
            "did not add decodable information at this scale."
        )
    return (
        f"Final-workspace answer decodability ({final:.3f}) exceeded memory-only "
        f"decodability ({memory:.3f}) by {final - memory:.3f} on this held-out "
        "probe split."
    )


def _probe_scores(
    features: np.ndarray,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    score_indices: np.ndarray,
    class_count: int,
    ridge: float,
    feature_name: str,
    label_name: str,
) -> tuple[float, float]:
    predictions = _linear_probe_predictions(
        features,
        labels,
        fit_indices,
        score_indices,
        class_count,
        ridge,
        feature_name=feature_name,
        label_name=label_name,
    )
    scored_labels = labels[score_indices]
    per_symbol = float(np.mean(predictions == scored_labels))
    if labels.ndim == 1:
        return per_symbol, per_symbol
    exact_rule = float(np.mean(np.all(predictions == scored_labels, axis=1)))
    return per_symbol, exact_rule


def analyze_latent_workspace(
    states: np.ndarray,
    memory_read: np.ndarray,
    answers: np.ndarray,
    rules: np.ndarray,
    fit_indices: Sequence[int] | np.ndarray,
    score_indices: Sequence[int] | np.ndarray,
    ridge: float = 1e-6,
    *,
    memory_values: np.ndarray | None = None,
    memory_keys: np.ndarray | None = None,
) -> dict[str, object]:
    """Produce the complete held-out latent-workspace analysis.

    Parameters
    ----------
    states : numpy.ndarray
        Workspace states shaped ``(episodes, iterations, width)``.
    memory_read : numpy.ndarray
        Query-conditioned memory reads shaped ``(episodes, width)``.
    answers : numpy.ndarray
        Query-answer class indices shaped ``(episodes,)``.
    rules : numpy.ndarray
        Per-episode permutations shaped ``(episodes, symbol_count)``.
    fit_indices : sequence of int
        Episode indices reserved for fitting all probes.
    score_indices : sequence of int
        Disjoint episode indices reserved for scoring all probes.
    ridge : float, default=1e-6
        Positive ridge coefficient used by every probe.
    memory_values : numpy.ndarray, optional
        Raw contextual-memory value factors shaped
        ``(episodes, slots, width)``. Must be supplied with ``memory_keys``.
    memory_keys : numpy.ndarray, optional
        Raw contextual-memory key factors shaped
        ``(episodes, slots, width)``. Must be supplied with ``memory_values``.

    Returns
    -------
    dict
        JSON-friendly geometry, answer and rule probe scores, split details,
        and the memory-versus-final comparison sentence.

    Raises
    ------
    ValueError
        If shapes, permutations, labels, or the probe split are invalid.
    """
    state_array = _latent_array(states)
    memory_array = _feature_array(memory_read, "memory_read")
    episode_count, iteration_count, width = state_array.shape
    if memory_array.shape != (episode_count, width):
        raise ValueError(
            "states and memory_read shapes must share episodes and width; got "
            f"states {state_array.shape}, memory_read {memory_array.shape}"
        )

    rule_array = np.asarray(rules)
    if rule_array.ndim != 2 or rule_array.shape[0] != episode_count:
        raise ValueError(
            "rules shape must be (episodes, symbol_count) and match states; got "
            f"rules {rule_array.shape}, states {state_array.shape}"
        )
    symbol_count = _class_count(rule_array.shape[1])
    rule_array = _label_array(
        rule_array, episode_count, symbol_count, "rules", "states"
    )
    expected = np.arange(symbol_count, dtype=np.int64)
    if np.any(np.sort(rule_array, axis=1) != expected):
        raise ValueError(
            f"rules shape {rule_array.shape} must contain one permutation per episode"
        )

    answer_array = np.asarray(answers)
    if answer_array.shape != (episode_count,):
        raise ValueError(
            "answers shape must be (episodes,) and match states; got "
            f"answers {answer_array.shape}, states {state_array.shape}"
        )
    answer_array = _label_array(
        answer_array, episode_count, symbol_count, "answers", "states"
    )
    fit, score = _probe_split(fit_indices, score_indices, episode_count)
    coefficient = _ridge_coefficient(ridge)

    answer_workspace: list[float] = []
    rule_workspace: list[float] = []
    rule_symbol_workspace: list[float] = []
    for iteration in range(iteration_count):
        answer_score, _ = _probe_scores(
            state_array[:, iteration],
            answer_array,
            fit,
            score,
            symbol_count,
            coefficient,
            f"states[:, {iteration}]",
            "answers",
        )
        rule_symbol_score, rule_exact_score = _probe_scores(
            state_array[:, iteration],
            rule_array,
            fit,
            score,
            symbol_count,
            coefficient,
            f"states[:, {iteration}]",
            "rules",
        )
        answer_workspace.append(answer_score)
        rule_workspace.append(rule_exact_score)
        rule_symbol_workspace.append(rule_symbol_score)

    answer_memory, _ = _probe_scores(
        memory_array,
        answer_array,
        fit,
        score,
        symbol_count,
        coefficient,
        "memory_read",
        "answers",
    )
    rule_symbol_memory, rule_memory = _probe_scores(
        memory_array,
        rule_array,
        fit,
        score,
        symbol_count,
        coefficient,
        "memory_read",
        "rules",
    )

    report: dict[str, object] = {
        "participation_ratio": participation_ratio(state_array).tolist(),
        "participation_ratio_method": {
            "name": "deterministic_hutchinson",
            "probe_count": _HUTCHINSON_PROBE_COUNT,
            "exact_through_width": _HUTCHINSON_PROBE_COUNT,
            "limitation": (
                "Above the exact-width threshold this deterministic finite sketch "
                "has a nontrivial nullspace."
            ),
        },
        "trajectory_step_norm": trajectory_step_norm(state_array).tolist(),
        "answer_decodability": {
            "workspace_per_iteration": answer_workspace,
            "memory_read": answer_memory,
            "final_workspace": answer_workspace[-1],
        },
        "rule_decodability": {
            "workspace_per_iteration": rule_workspace,
            "memory_read": rule_memory,
            "final_workspace": rule_workspace[-1],
        },
        "rule_per_symbol_decodability": {
            "workspace_per_iteration": rule_symbol_workspace,
            "memory_read": rule_symbol_memory,
            "final_workspace": rule_symbol_workspace[-1],
        },
        "probe_split": {
            "fit_count": int(fit.size),
            "score_count": int(score.size),
            "fit_indices": fit.tolist(),
            "score_indices": score.tolist(),
        },
        "comparison": memory_final_comparison(answer_memory, answer_workspace[-1]),
    }

    if (memory_values is None) != (memory_keys is None):
        provided = "memory_values" if memory_values is not None else "memory_keys"
        missing = "memory_keys" if memory_values is not None else "memory_values"
        raise ValueError(f"{provided} was provided but {missing} is missing")
    if memory_values is not None and memory_keys is not None:
        value_array = _factor_array(
            memory_values, "memory_values", tuple(state_array.shape)
        )
        key_array = _factor_array(memory_keys, "memory_keys", tuple(state_array.shape))
        if value_array.shape != key_array.shape:
            raise ValueError(
                "memory_values and memory_keys shapes must match; got "
                f"memory_values {value_array.shape}, memory_keys {key_array.shape}"
            )
        raw_factors = np.concatenate(
            (
                value_array.reshape(episode_count, -1),
                key_array.reshape(episode_count, -1),
            ),
            axis=1,
        )
        raw_answer, _ = _probe_scores(
            raw_factors,
            answer_array,
            fit,
            score,
            symbol_count,
            coefficient,
            "raw_memory_factors",
            "answers",
        )
        raw_rule_symbol, raw_rule_exact = _probe_scores(
            raw_factors,
            rule_array,
            fit,
            score,
            symbol_count,
            coefficient,
            "raw_memory_factors",
            "rules",
        )
        report["raw_memory_factor_decodability"] = {
            "answer": raw_answer,
            "full_rule_exact": raw_rule_exact,
            "rule_per_symbol": raw_rule_symbol,
        }
    return report
