"""Tests for Example 21 latent-workspace analysis."""

from __future__ import annotations

import json

import numpy as np
import pytest

try:
    import examples.pp_prop.latent_workspace_analysis as analysis
except ModuleNotFoundError as error:
    if error.name not in {
        "examples",
        "examples.pp_prop",
        "examples.pp_prop.latent_workspace_analysis",
    }:
        raise
    import latent_workspace_analysis as analysis

analyze_latent_workspace = analysis.analyze_latent_workspace
linear_probe_accuracy = analysis.linear_probe_accuracy
memory_final_comparison = analysis.memory_final_comparison
participation_ratio = analysis.participation_ratio
trajectory_step_norm = analysis.trajectory_step_norm


def _probe_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.vstack((np.eye(3), np.eye(3)))
    answers = np.array([0, 1, 2, 0, 1, 2])
    rules = np.array(
        [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
        ]
    )
    return features, answers, rules


def _dense_probe_predictions(
    features: np.ndarray,
    labels: np.ndarray,
    fit: np.ndarray,
    score: np.ndarray,
    class_count: int,
    ridge: float,
) -> np.ndarray:
    fit_features = features[fit]
    feature_mean = np.mean(fit_features, axis=0, keepdims=True)
    centered_features = fit_features - feature_mean
    targets = (
        labels[fit][..., None] == np.arange(class_count, dtype=np.int64)
    ).reshape(fit.size, -1)
    target_mean = np.mean(targets, axis=0, keepdims=True)
    centered_targets = targets - target_mean
    gram = centered_features.T @ centered_features
    weights = np.linalg.solve(
        gram + ridge * np.eye(features.shape[1]),
        centered_features.T @ centered_targets,
    )
    scores = (features[score] - feature_mean) @ weights + target_mean
    target_shape = labels.shape[1:] + (class_count,)
    return np.argmax(scores.reshape((score.size,) + target_shape), axis=-1)


def test_participation_ratio_recovers_known_ranks_and_zero_rank() -> None:
    states = np.zeros((4, 3, 3))
    states[:, 1, 0] = [-3.0, -1.0, 1.0, 3.0]
    states[:, 2, :2] = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]

    ratios = participation_ratio(states)

    np.testing.assert_allclose(ratios, [0.0, 1.0, 2.0], atol=1e-12)


def test_participation_ratio_matches_dense_trace_oracle_below_probe_count() -> None:
    values = np.sin(np.arange(70, dtype=np.float64).reshape(7, 2, 5) * 0.37)
    centered = values - np.mean(values, axis=0, keepdims=True)
    covariance = np.einsum("eiw,eiv->iwv", centered, centered)
    trace = np.trace(covariance, axis1=1, axis2=2)
    expected = np.square(trace) / np.sum(np.square(covariance), axis=(1, 2))

    np.testing.assert_allclose(participation_ratio(values), expected, rtol=1e-12)


def test_participation_ratio_width_seventeen_does_not_alias_period_sixteen() -> None:
    values = np.array([-3.0, -1.0, 1.0, 3.0])
    states = np.zeros((4, 1, 17))
    states[:, 0, 0] = values
    states[:, 0, 16] = -values

    probes = analysis._fixed_rademacher_probes(17)

    assert not np.array_equal(probes[:, 0], probes[:, 16])
    np.testing.assert_allclose(participation_ratio(states), [1.0], atol=1e-12)


def test_wide_rademacher_probe_columns_do_not_repeat_every_sixteen() -> None:
    probes = analysis._fixed_rademacher_probes(257)

    assert np.unique(probes.T, axis=0).shape[0] == probes.shape[1]
    assert not np.any(np.all(probes[:, :-16] == probes[:, 16:], axis=0))


def test_trajectory_step_norm_handles_fixed_point_and_divergence() -> None:
    fixed = np.ones((3, 4, 2))
    divergent = np.array([[[0.0], [1.0], [3.0], [6.0]]] * 2)

    np.testing.assert_array_equal(trajectory_step_norm(fixed), np.zeros(4))
    np.testing.assert_allclose(trajectory_step_norm(divergent), [0.0, 1.0, 2.0, 3.0])


def test_answer_probe_fits_and_scores_on_disjoint_repeated_examples() -> None:
    features, answers, _ = _probe_fixture()

    accuracy = linear_probe_accuracy(
        features, answers, [0, 1, 2], [3, 4, 5], class_count=3
    )

    assert accuracy == 1.0


def test_matrix_free_probe_predictions_match_dense_centered_ridge_oracle() -> None:
    features = np.sin(np.arange(96, dtype=np.float64).reshape(16, 6) * 0.37)
    labels = np.stack((np.arange(16) % 3, (np.arange(16) + 1) % 3), axis=1)
    fit = np.arange(12)
    score = np.arange(12, 16)
    ridge = 0.03

    actual = analysis._linear_probe_predictions(features, labels, fit, score, 3, ridge)
    expected = _dense_probe_predictions(features, labels, fit, score, 3, ridge)

    np.testing.assert_array_equal(actual, expected)


def test_matrix_free_probe_converges_past_the_old_fixed_iteration_limit() -> None:
    generator = np.random.default_rng(0)
    features = generator.normal(size=(160, 80))
    labels = np.arange(160) % 10
    fit = np.arange(120)
    score = np.arange(120, 160)

    actual = analysis._linear_probe_predictions(features, labels, fit, score, 10, 1e-6)
    expected = _dense_probe_predictions(features, labels, fit, score, 10, 1e-6)

    np.testing.assert_array_equal(actual, expected)


def test_matrix_free_probe_names_a_genuinely_unresolved_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = np.random.default_rng(1)
    features = generator.normal(size=(40, 20))
    labels = np.arange(40) % 3
    monkeypatch.setattr(analysis, "_CG_MAX_ITERATIONS", 1)

    with pytest.raises(
        ValueError,
        match=r"features ridge probe failed to reach bounded relative residual .+ "
        r"within 1 matrix-free iterations; maximum residual",
    ):
        analysis._linear_probe_predictions(
            features, labels, np.arange(30), np.arange(30, 40), 3, 1e-6
        )


def test_analysis_does_not_use_dense_width_factorizations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("dense width-by-width factorization is forbidden")

    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "svd", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "pinv", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(np.linalg, "eigvalsh", forbidden)
    features = np.sin(np.arange(12 * 257, dtype=np.float64).reshape(12, 257) * 0.013)
    labels = np.arange(12) % 3
    states = np.stack((features, features * 0.9), axis=1)

    ratios = participation_ratio(states)
    predictions = analysis._linear_probe_predictions(
        features, labels, np.arange(8), np.arange(8, 12), 3, 0.1
    )

    assert ratios.shape == (2,)
    assert predictions.shape == (4,)


def test_rule_probe_scores_mean_per_source_symbol() -> None:
    features, _, rules = _probe_fixture()
    corrupted = rules.copy()
    corrupted[3, 0] = 2

    accuracy = linear_probe_accuracy(
        features, corrupted, [0, 1, 2], [3, 4, 5], class_count=3
    )

    assert accuracy == pytest.approx(8.0 / 9.0)


def test_probe_does_not_leak_score_labels_into_fit() -> None:
    features = np.array([[-1.0], [1.0], [-1.0], [1.0]])
    labels = np.array([0, 1, 1, 0])

    accuracy = linear_probe_accuracy(features, labels, [0, 1], [2, 3], class_count=2)

    assert accuracy == 0.0


def test_complete_analysis_separates_answer_and_rule_reports() -> None:
    features, answers, rules = _probe_fixture()
    states = np.stack((features, features * 2.0), axis=1)

    report = analyze_latent_workspace(
        states, features, answers, rules, [0, 1, 2], [3, 4, 5]
    )

    assert report["answer_decodability"] == {
        "workspace_per_iteration": [1.0, 1.0],
        "memory_read": 1.0,
        "final_workspace": 1.0,
    }
    assert report["rule_decodability"] == {
        "workspace_per_iteration": [1.0, 1.0],
        "memory_read": 1.0,
        "final_workspace": 1.0,
    }
    assert report["rule_per_symbol_decodability"] == {
        "workspace_per_iteration": [1.0, 1.0],
        "memory_read": 1.0,
        "final_workspace": 1.0,
    }
    assert report["participation_ratio_method"] == {
        "name": "deterministic_hutchinson",
        "probe_count": 16,
        "exact_through_width": 16,
        "limitation": (
            "Above the exact-width threshold this deterministic finite sketch "
            "has a nontrivial nullspace."
        ),
    }
    assert report["probe_split"] == {
        "fit_count": 3,
        "score_count": 3,
        "fit_indices": [0, 1, 2],
        "score_indices": [3, 4, 5],
    }
    assert "did not add decodable information" in report["comparison"]
    json.dumps(report, allow_nan=False)


def test_complete_analysis_reports_exact_rule_and_per_symbol_scores() -> None:
    identity = np.arange(10)
    features = np.vstack((np.eye(10), np.eye(10)))
    answers = np.tile(identity, 2)
    fit_rules = np.tile(identity, (10, 1))
    score_rules = fit_rules.copy()
    score_rules[:, [-2, -1]] = score_rules[:, [-1, -2]]
    rules = np.vstack((fit_rules, score_rules))
    states = features[:, None, :]

    report = analyze_latent_workspace(
        states,
        features,
        answers,
        rules,
        np.arange(10),
        np.arange(10, 20),
    )

    assert report["rule_decodability"]["final_workspace"] == 0.0
    assert report["rule_per_symbol_decodability"]["final_workspace"] == 0.8


def test_raw_memory_factor_probes_are_a_separate_optional_result() -> None:
    features, answers, rules = _probe_fixture()
    states = features[:, None, :]

    report = analyze_latent_workspace(
        states,
        features,
        answers,
        rules,
        [0, 1, 2],
        [3, 4, 5],
        memory_values=features[:, None, :],
        memory_keys=np.ones((6, 1, 3)),
    )

    assert report["raw_memory_factor_decodability"] == {
        "answer": 1.0,
        "full_rule_exact": 1.0,
        "rule_per_symbol": 1.0,
    }
    json.dumps(report, allow_nan=False)


def test_wide_raw_memory_factor_probe_remains_matrix_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_: object, **__: object) -> None:
        raise AssertionError("dense square factorization is forbidden")

    for name in ("solve", "svd", "inv", "pinv", "eigh", "eigvalsh"):
        monkeypatch.setattr(np.linalg, name, forbidden)
    generator = np.random.default_rng(17)
    episode_count = 160
    symbol_count = 10
    width = 32
    slots = 8
    states = generator.normal(size=(episode_count, 1, width))
    memory_read = generator.normal(size=(episode_count, width))
    answers = np.arange(episode_count) % symbol_count
    rules = np.stack(
        [generator.permutation(symbol_count) for _ in range(episode_count)]
    )
    memory_values = generator.normal(size=(episode_count, slots, width))
    memory_keys = generator.normal(size=(episode_count, slots, width))

    report = analyze_latent_workspace(
        states,
        memory_read,
        answers,
        rules,
        np.arange(120),
        np.arange(120, episode_count),
        memory_values=memory_values,
        memory_keys=memory_keys,
    )

    assert set(report["raw_memory_factor_decodability"]) == {
        "answer",
        "full_rule_exact",
        "rule_per_symbol",
    }
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    ("memory", "final", "expected"),
    [
        (0.8, 0.8, "matched or exceeded"),
        (0.9, 0.8, "did not add decodable information"),
        (0.4, 0.7, "exceeded memory-only"),
    ],
)
def test_memory_final_comparison_retains_both_outcomes(
    memory: float, final: float, expected: str
) -> None:
    assert expected in memory_final_comparison(memory, final)


@pytest.mark.parametrize(
    ("states", "message"),
    [
        (np.ones((2, 3)), "states shape"),
        (np.ones((0, 2, 3)), "states shape"),
        (np.array([[[np.inf]]]), "non-finite"),
    ],
)
def test_geometry_rejects_malformed_states(states: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        participation_ratio(states)


def test_trajectory_rejects_non_finite_derived_norm() -> None:
    limit = np.finfo(np.float64).max
    states = np.array([[[limit], [-limit]]])

    with pytest.raises(ValueError, match="non-finite trajectory_step_norm"):
        trajectory_step_norm(states)


def test_complete_analysis_names_mismatched_memory_shape() -> None:
    features, answers, rules = _probe_fixture()
    states = np.stack((features, features), axis=1)

    with pytest.raises(ValueError, match=r"states \(6, 2, 3\), memory_read \(5, 3\)"):
        analyze_latent_workspace(
            states, features[:-1], answers, rules, [0, 1, 2], [3, 4, 5]
        )

    with pytest.raises(ValueError, match=r"memory_read \(6, 2\)"):
        analyze_latent_workspace(
            states, features[:, :2], answers, rules, [0, 1, 2], [3, 4, 5]
        )


def test_complete_analysis_names_mismatched_answer_and_rule_shapes() -> None:
    features, answers, rules = _probe_fixture()
    states = np.stack((features, features), axis=1)

    with pytest.raises(ValueError, match=r"answers \(5,\), states \(6, 2, 3\)"):
        analyze_latent_workspace(
            states, features, answers[:-1], rules, [0, 1, 2], [3, 4, 5]
        )

    with pytest.raises(ValueError, match=r"rules \(5, 3\), states \(6, 2, 3\)"):
        analyze_latent_workspace(
            states, features, answers, rules[:-1], [0, 1, 2], [3, 4, 5]
        )


def test_complete_analysis_names_malformed_public_quantities() -> None:
    features, answers, rules = _probe_fixture()
    states = np.stack((features, features), axis=1)

    with pytest.raises(ValueError, match="memory_read shape"):
        analyze_latent_workspace(
            states,
            features[:, :, None],
            answers,
            rules,
            [0, 1, 2],
            [3, 4, 5],
        )
    with pytest.raises(ValueError, match="answers shape"):
        analyze_latent_workspace(
            states,
            features,
            answers.astype(np.float64),
            rules,
            [0, 1, 2],
            [3, 4, 5],
        )
    with pytest.raises(ValueError, match="rules shape"):
        analyze_latent_workspace(
            states,
            features,
            answers,
            rules.astype(np.float64),
            [0, 1, 2],
            [3, 4, 5],
        )


def test_raw_memory_factors_validate_pairing_shapes_and_finiteness() -> None:
    features, answers, rules = _probe_fixture()
    states = features[:, None, :]
    common = (states, features, answers, rules, [0, 1, 2], [3, 4, 5])

    with pytest.raises(ValueError, match="memory_keys is missing"):
        analyze_latent_workspace(*common, memory_values=features[:, None, :])
    with pytest.raises(ValueError, match=r"memory_values \(5, 1, 3\).+states"):
        analyze_latent_workspace(
            *common,
            memory_values=features[:-1, None, :],
            memory_keys=features[:, None, :],
        )
    with pytest.raises(ValueError, match="memory_keys shape .* non-finite"):
        invalid_keys = np.ones((6, 1, 3))
        invalid_keys[0, 0, 0] = np.inf
        analyze_latent_workspace(
            *common,
            memory_values=features[:, None, :],
            memory_keys=invalid_keys,
        )


def test_complete_analysis_rejects_non_permutation_rules() -> None:
    features, answers, rules = _probe_fixture()
    states = np.stack((features, features), axis=1)
    rules[0] = [0, 0, 2]

    with pytest.raises(ValueError, match="one permutation per episode"):
        analyze_latent_workspace(states, features, answers, rules, [0, 1, 2], [3, 4, 5])


@pytest.mark.parametrize(
    ("fit", "score", "message"),
    [
        ([], [1], "fit_indices must not be empty"),
        ([0], [], "score_indices must not be empty"),
        ([0, 1], [1, 2], "must be disjoint"),
        ([0, 0], [1], "fit_indices contains duplicate"),
        ([0], [9], "outside episode_count"),
        ([[0]], [1], "fit_indices shape"),
        ([0.0], [1], "integer episode indices"),
    ],
)
def test_probe_rejects_invalid_split(
    fit: list[object], score: list[object], message: str
) -> None:
    features, answers, _ = _probe_fixture()

    with pytest.raises(ValueError, match=message):
        linear_probe_accuracy(features, answers, fit, score, class_count=3)


@pytest.mark.parametrize(
    ("labels", "class_count", "message"),
    [
        (np.array([0, 1, 2, 0, 1]), 3, "labels shape"),
        (np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]), 3, "integer class"),
        (np.array([0, 1, 3, 0, 1, 2]), 3, "labels classes"),
        (np.array([0, 1, 2, 0, 1, 2]), 1, "at least 2"),
    ],
)
def test_probe_rejects_malformed_labels_or_class_count(
    labels: np.ndarray, class_count: int, message: str
) -> None:
    features, _, _ = _probe_fixture()

    with pytest.raises(ValueError, match=message):
        linear_probe_accuracy(
            features, labels, [0, 1, 2], [3, 4, 5], class_count=class_count
        )


@pytest.mark.parametrize("ridge", [0.0, -1.0, np.inf, True, "small"])
def test_probe_rejects_invalid_ridge(ridge: object) -> None:
    features, answers, _ = _probe_fixture()

    with pytest.raises(ValueError, match="ridge"):
        linear_probe_accuracy(
            features,
            answers,
            [0, 1, 2],
            [3, 4, 5],
            3,
            ridge,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("memory", "final", "quantity"),
    [(np.nan, 0.5, "memory_accuracy"), (0.5, 1.1, "final_accuracy")],
)
def test_comparison_rejects_invalid_accuracy(
    memory: float, final: float, quantity: str
) -> None:
    with pytest.raises(ValueError, match=quantity):
        memory_final_comparison(memory, final)
