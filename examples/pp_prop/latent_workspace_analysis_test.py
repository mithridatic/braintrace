"""Tests for Example 21 exact ARC scoring and trajectory analysis."""

from __future__ import annotations

import copy
import itertools
import json
import math

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


DecodedCandidate = analysis.DecodedCandidate
OutputLogits = analysis.OutputLogits
QueryScore = analysis.QueryScore
SelectedModelCandidate = analysis.SelectedModelCandidate
aggregate_arc_metrics = analysis.aggregate_arc_metrics
analyze_latent_trajectory = analysis.analyze_latent_trajectory
assess_model_only_completion = analysis.assess_model_only_completion
compare_control_trajectories = analysis.compare_control_trajectories
decode_candidates = analysis.decode_candidates
score_query_candidates = analysis.score_query_candidates
select_checkpoint_candidates = analysis.select_checkpoint_candidates


def _logits(
    grid: np.ndarray,
    *,
    second_cell: tuple[int, int, int] | None = None,
) -> OutputLogits:
    height = np.full(30, -20.0)
    width = np.full(30, -20.0)
    height[grid.shape[0] - 1] = 20.0
    width[grid.shape[1] - 1] = 20.0
    colors = np.full((30, 30, 10), -20.0)
    colors[:, :, 0] = 20.0
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            colors[row, column, :] = -20.0
            colors[row, column, int(grid[row, column])] = 20.0
    if second_cell is not None:
        row, column, color = second_cell
        colors[row, column, color] = 19.9
    return OutputLogits(height, width, colors)


def _stacked_logits(
    grids: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs = [_logits(grid) for grid in grids]
    return (
        np.stack([output.height for output in outputs]),
        np.stack([output.width for output in outputs]),
        np.stack([output.colors for output in outputs]),
    )


def _model_only_completion_fixture(
    solved_task_count: int,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    expected_queries = {
        f"task-{task_index:03d}": 2 if task_index == 0 else 1
        for task_index in range(400)
    }
    latest = _logits(np.array([[4]]), second_cell=(0, 0, 7))
    selected = select_checkpoint_candidates(
        {30: latest, 60: latest}, latest_checkpoint=60
    )
    candidates = [candidate.to_dict() for candidate in selected]
    records: list[dict[str, object]] = []
    for task_index, (task_id, query_count) in enumerate(expected_queries.items()):
        exact = task_index < solved_task_count
        for query_index in range(query_count):
            score = QueryScore(
                task_id=task_id,
                query_index=query_index,
                pass_at_1=False,
                pass_at_2=exact,
                shape_accuracy=False,
                valid_cell_pixel_accuracy=0.0,
                candidate_count=2,
            )
            records.append(
                {
                    "task_id": task_id,
                    "query_index": query_index,
                    "primary_candidate_mode": "model_only",
                    "candidates": copy.deepcopy(candidates),
                    "score": score.to_dict(),
                }
            )
    return records, expected_queries


def test_output_logits_validate_and_freeze_all_heads() -> None:
    logits = _logits(np.array([[3, 4], [5, 6]]))

    assert np.asarray(logits.height).shape == (30,)
    assert np.asarray(logits.width).shape == (30,)
    assert np.asarray(logits.colors).shape == (30, 30, 10)
    assert not np.asarray(logits.colors).flags.writeable


@pytest.mark.parametrize(
    ("head", "value", "message"),
    [
        ("height", np.zeros(29), "height logits shape"),
        ("width", np.zeros((1, 30)), "width logits shape"),
        ("colors", np.zeros((30, 30, 9)), "color logits shape"),
        ("height", np.full(30, np.nan), "non-finite"),
        ("colors", [["bad"]], "numeric array"),
    ],
)
def test_output_logits_reject_malformed_heads(
    head: str, value: object, message: str
) -> None:
    values: dict[str, object] = {
        "height": np.zeros(30),
        "width": np.zeros(30),
        "colors": np.zeros((30, 30, 10)),
    }
    values[head] = value

    with pytest.raises(ValueError, match=message):
        OutputLogits(**values)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([[1], [2, 3]], "rectangular integer grid"),
        (np.asarray([[True]]), "non-boolean integer"),
        (np.asarray([[-1]]), "colors"),
        (np.asarray([[10]]), "colors"),
    ],
)
def test_decoded_candidate_rejects_invalid_grids(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DecodedCandidate(value)


def test_decoder_emits_joint_argmax_and_lowest_margin_cell_alternative() -> None:
    first_grid = np.array([[1, 2], [3, 4]])
    logits = _logits(first_grid, second_cell=(1, 0, 9))

    candidates = decode_candidates(logits)

    np.testing.assert_array_equal(candidates[0].grid, first_grid)
    np.testing.assert_array_equal(candidates[1].grid, [[1, 2], [9, 4]])
    assert candidates[0].changed_decision is None
    assert candidates[1].changed_decision == "cell:1,0"
    assert candidates[0].log_probability > candidates[1].log_probability
    one_candidate = decode_candidates(logits, max_candidates=1)
    assert len(one_candidate) == 1
    np.testing.assert_array_equal(one_candidate[0].grid, candidates[0].grid)
    assert one_candidate[0].changed_decision is None
    json.dumps([candidate.to_dict() for candidate in candidates], allow_nan=False)


def test_factorized_log_probability_scores_every_included_cell() -> None:
    grid = np.array([[1, 2], [3, 4]])
    logits = _logits(grid)
    score = analysis._candidate_log_probability(logits, 1, 1, grid)
    expected = (
        analysis._log_softmax_choice(logits.height, 1)
        + analysis._log_softmax_choice(logits.width, 1)
        + sum(
            analysis._log_softmax_choice(logits.colors[row, column], int(grid[row, column]))
            for row in range(2)
            for column in range(2)
        )
    )
    assert score == pytest.approx(expected)


def test_checkpoint_selection_uses_only_latest_completed_sweep() -> (
    None
):
    checkpoint_30 = _logits(np.array([[3]]))
    checkpoint_60 = _logits(np.array([[6]]))
    checkpoint_90 = _logits(np.array([[9]]))

    selected = select_checkpoint_candidates(
        {
            0: _logits(np.array([[1]])),
            30: checkpoint_30,
            60: checkpoint_60,
            90: checkpoint_90,
        },
        latest_checkpoint=90,
    )

    np.testing.assert_array_equal(selected[0].candidate.grid, [[9]])
    np.testing.assert_array_equal(
        selected[1].candidate.grid, decode_candidates(checkpoint_90)[1].grid
    )
    assert selected[0].source_checkpoint == 90
    assert selected[0].selection_role == "latest_sweep_joint_argmax"
    assert selected[1].source_checkpoint == 90
    assert selected[1].selection_role == "latest_sweep_logit_runner_up"
    assert selected[1].candidate.changed_decision is not None
    assert selected[0].to_dict()["provenance"] == "model"
    json.dumps([candidate.to_dict() for candidate in selected], allow_nan=False)


def test_checkpoint_selection_never_promotes_an_earlier_distinct_grid() -> (
    None
):
    latest = _logits(np.array([[9]]))

    selected = select_checkpoint_candidates(
        {30: _logits(np.array([[3]])), 60: latest, 90: latest},
        latest_checkpoint=90,
    )

    np.testing.assert_array_equal(selected[1].candidate.grid, decode_candidates(latest)[1].grid)
    assert selected[1].source_checkpoint == 90
    assert selected[1].selection_role == "latest_sweep_logit_runner_up"


def test_checkpoint_selection_falls_back_to_latest_logit_runner_up() -> None:
    latest = _logits(np.array([[4]]), second_cell=(0, 0, 7))
    expected = decode_candidates(latest)[1]

    selected = select_checkpoint_candidates(
        {0: _logits(np.array([[2]])), 30: latest, 60: latest},
        latest_checkpoint=60,
    )

    np.testing.assert_array_equal(selected[0].candidate.grid, [[4]])
    np.testing.assert_array_equal(selected[1].candidate.grid, expected.grid)
    assert selected[1].source_checkpoint == 60
    assert selected[1].selection_role == "latest_sweep_logit_runner_up"
    assert not np.array_equal(selected[0].candidate.grid, selected[1].candidate.grid)


@pytest.mark.parametrize(
    ("checkpoints", "latest", "sweep_size", "message"),
    [
        ([(30, _logits(np.array([[1]])))], 30, 30, "mapping"),
        ({30: _logits(np.array([[1]]))}, True, 30, "latest_checkpoint"),
        ({30: _logits(np.array([[1]]))}, 0, 30, "completed sweep"),
        ({30: _logits(np.array([[1]]))}, 30, 0, "sweep_size"),
        ({30: _logits(np.array([[1]]))}, 30, True, "sweep_size"),
        ({30: _logits(np.array([[1]]))}, 60, 30, "latest checkpoint is absent"),
        (
            {
                30: _logits(np.array([[1]])),
                45: _logits(np.array([[2]])),
                60: _logits(np.array([[3]])),
            },
            60,
            30,
            "completed sweep",
        ),
        (
            {30: _logits(np.array([[1]])), 90: _logits(np.array([[2]]))},
            30,
            30,
            "after latest",
        ),
        ({30: np.zeros(4)}, 30, 30, "OutputLogits"),
    ],
)
def test_checkpoint_selection_rejects_malformed_history(
    checkpoints: object, latest: object, sweep_size: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        select_checkpoint_candidates(
            checkpoints,  # type: ignore[arg-type]
            latest_checkpoint=latest,  # type: ignore[arg-type]
            sweep_size=sweep_size,  # type: ignore[arg-type]
        )


def test_checkpoint_selection_fails_closed_without_a_distinct_latest_runner_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    latest = _logits(np.array([[4]]))
    first = decode_candidates(latest, max_candidates=1)[0]
    monkeypatch.setattr(
        analysis, "decode_candidates", lambda logits, max_candidates=2: (first,)
    )

    with pytest.raises(ValueError, match="distinct runner-up"):
        select_checkpoint_candidates({60: latest}, latest_checkpoint=60)


@pytest.mark.parametrize(
    ("candidate", "checkpoint", "role", "message"),
    [
        (object(), 30, "latest_sweep_joint_argmax", "DecodedCandidate"),
        (
            decode_candidates(_logits(np.array([[1]])), max_candidates=1)[0],
            0,
            "latest_sweep_joint_argmax",
            "positive",
        ),
        (
            decode_candidates(_logits(np.array([[1]])), max_candidates=1)[0],
            30,
            "rule",
            "selection_role",
        ),
        (
            decode_candidates(_logits(np.array([[1]])), max_candidates=1)[0],
            30,
            "latest_sweep_logit_runner_up",
            "changed decision",
        ),
        (
            decode_candidates(_logits(np.array([[1]]), second_cell=(0, 0, 2)))[1],
            30,
            "earlier_sweep_joint_argmax",
            "cannot name a changed decision",
        ),
    ],
)
def test_selected_model_candidate_rejects_invalid_metadata(
    candidate: object, checkpoint: int, role: str, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        SelectedModelCandidate(
            candidate,  # type: ignore[arg-type]
            checkpoint,
            role,  # type: ignore[arg-type]
        )


def test_decoder_tie_order_prefers_shape_then_row_major_cells() -> None:
    grid = np.array([[7]])
    logits = _logits(grid)
    height = np.asarray(logits.height).copy()
    width = np.asarray(logits.width).copy()
    height[1] = height[0] - 0.5
    width[1] = width[0] - 0.5
    colors = np.asarray(logits.colors).copy()
    colors[0, 0, 8] = colors[0, 0, 7] - 0.5

    candidates = decode_candidates(OutputLogits(height, width, colors))

    assert candidates[1].changed_decision == "shape"
    assert np.asarray(candidates[1].grid).shape == (1, 2)


def test_decoder_can_select_width_runner_up() -> None:
    logits = _logits(np.array([[4]]))
    width = np.asarray(logits.width).copy()
    width[1] = width[0] - 0.01

    candidates = decode_candidates(OutputLogits(logits.height, width, logits.colors))

    assert candidates[1].changed_decision == "shape"
    assert np.asarray(candidates[1].grid).shape == (1, 2)


@pytest.mark.parametrize("value", [0, 3, True, 1.5])
def test_decoder_rejects_invalid_candidate_count(value: object) -> None:
    with pytest.raises(ValueError, match="one or two"):
        decode_candidates(_logits(np.array([[1]])), value)  # type: ignore[arg-type]


def test_decoder_rejects_unvalidated_container() -> None:
    with pytest.raises(TypeError, match="OutputLogits"):
        decode_candidates(object())  # type: ignore[arg-type]


def test_decode_candidates_ranks_shapes_by_complete_factorized_probability() -> None:
    height = np.full((30,), -100.0)
    width = np.full((30,), -100.0)
    height[:2] = (0.0, 2.0)
    width[:2] = (0.0, 2.0)
    colors = np.full((30, 30, 10), -100.0)
    colors[0, 0, 0] = 100.0
    colors[0, 1] = 0.0
    colors[1, 0] = 0.0
    colors[1, 1] = 0.0

    first, second = decode_candidates(OutputLogits(height, width, colors))

    assert first.grid.shape == (1, 1)
    assert second.grid.shape == (1, 2)
    assert first.log_probability > second.log_probability
    assert second.changed_decision == "shape"


def test_decode_candidates_global_top_two_prefers_shape_on_exact_tie() -> None:
    height = np.full((30,), -100.0)
    width = np.full((30,), -100.0)
    height[0] = 0.0
    width[:2] = (0.0, 0.0)
    colors = np.full((30, 30, 10), -100.0)
    colors[0, 0, 0] = 0.0
    colors[0, 0, 1] = -math.log(10.0)
    colors[0, 1, 0] = 0.0

    first, second = decode_candidates(OutputLogits(height, width, colors))

    assert first.grid.shape == (1, 1)
    assert second.grid.shape == (1, 2)
    assert second.changed_decision == "shape"


def test_decode_candidates_matches_exhaustive_tiny_factorized_distributions() -> None:
    """Global top two must equal explicit enumeration, not a local heuristic."""
    for seed in range(20):
        rng = np.random.default_rng(seed)
        height = np.full((30,), -1.0e9)
        width = np.full((30,), -1.0e9)
        height[:2] = rng.normal(size=2)
        width[:2] = rng.normal(size=2)
        colors = np.full((30, 30, 10), -1.0e9)
        colors[:2, :2, :2] = rng.normal(size=(2, 2, 2))

        decoded = decode_candidates(OutputLogits(height, width, colors))

        def log_softmax(values: np.ndarray) -> np.ndarray:
            maximum = float(np.max(values))
            return values - (maximum + math.log(float(np.exp(values - maximum).sum())))

        height_logp = log_softmax(height)
        width_logp = log_softmax(width)
        color_logp = np.apply_along_axis(log_softmax, -1, colors)
        enumerated: list[tuple[float, int, int, tuple[int, ...]]] = []
        for height_value in (1, 2):
            for width_value in (1, 2):
                cell_count = height_value * width_value
                for flat_colors in itertools.product((0, 1), repeat=cell_count):
                    grid = np.asarray(flat_colors).reshape(height_value, width_value)
                    score = float(
                        height_logp[height_value - 1]
                        + width_logp[width_value - 1]
                        + sum(
                            color_logp[row, column, int(grid[row, column])]
                            for row in range(height_value)
                            for column in range(width_value)
                        )
                    )
                    enumerated.append(
                        (score, height_value, width_value, tuple(flat_colors))
                    )
        expected = sorted(enumerated, key=lambda item: item[0], reverse=True)[:2]
        for candidate, (_, height_value, width_value, flat_colors) in zip(
            decoded, expected, strict=True
        ):
            assert candidate.grid.shape == (height_value, width_value)
            np.testing.assert_array_equal(
                candidate.grid, np.asarray(flat_colors).reshape(height_value, width_value)
            )


def test_one_wrong_cell_fails_exact_but_preserves_pixel_diagnostic() -> None:
    target = np.arange(9, dtype=np.int8).reshape(3, 3)
    prediction = target.copy()
    prediction[2, 2] = 0

    score = score_query_candidates(
        [prediction], target, task_id="near-miss", query_index=0
    )

    assert not score.pass_at_1
    assert not score.pass_at_2
    assert score.shape_accuracy
    assert score.valid_cell_pixel_accuracy == pytest.approx(8.0 / 9.0)
    assert score.to_dict()["valid_cell_pixel_accuracy_diagnostic"] == pytest.approx(
        8.0 / 9.0
    )


def test_wrong_shape_fails_exact_and_missing_cells_earn_no_credit() -> None:
    target = np.array([[1, 2], [3, 4]])
    prediction = np.array([[1, 2]])

    score = score_query_candidates([prediction], target, task_id="shape", query_index=1)

    assert not score.shape_accuracy
    assert not score.pass_at_1
    assert score.valid_cell_pixel_accuracy == 0.5


def test_real_second_candidate_can_be_the_only_exact_success() -> None:
    first = np.array([[1, 2], [3, 4]])
    target = np.array([[1, 2], [9, 4]])
    candidates = decode_candidates(_logits(first, second_cell=(1, 0, 9)))

    score = score_query_candidates(candidates, target, task_id="second", query_index=0)

    assert not score.pass_at_1
    assert score.pass_at_2
    assert score.candidate_count == 2


def test_duplicate_second_candidate_is_removed() -> None:
    grid = np.array([[1, 2]])

    score = score_query_candidates(
        [grid, grid.copy()], grid, task_id="dedupe", query_index=0
    )

    assert score.pass_at_1 and score.pass_at_2
    assert score.candidate_count == 1


@pytest.mark.parametrize(
    ("candidates", "target", "message"),
    [
        ([], [[1]], "one or two"),
        (([[1]], [[2]], [[3]]), [[1]], "one or two"),
        ([[[True]]], [[1]], "non-boolean"),
        ([[[10]]], [[1]], "colors"),
        ([[[1]]], [], "target grid shape"),
        ("grid", [[1]], "sequence"),
    ],
)
def test_query_scorer_rejects_malformed_grids(
    candidates: object, target: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        score_query_candidates(
            candidates,  # type: ignore[arg-type]
            target,
            task_id="bad",
            query_index=0,
        )


def test_strict_task_metrics_are_conjunctive_over_multiple_queries() -> None:
    scores = [
        QueryScore("a", 0, True, True, True, 1.0, 2),
        QueryScore("a", 1, False, True, True, 0.75, 2),
        QueryScore("b", 0, True, True, True, 1.0, 1),
    ]

    report = aggregate_arc_metrics(scores)

    assert report["query_pass_at_1"] == pytest.approx(2.0 / 3.0)
    assert report["strict_task_pass_at_1"] == 0.5
    assert report["strict_task_pass_at_2"] == 1.0
    assert report["tasks"] == {
        "a": {"query_count": 2, "pass_at_1": False, "pass_at_2": True},
        "b": {"query_count": 1, "pass_at_1": True, "pass_at_2": True},
    }
    assert report["valid_cell_pixel_accuracy_diagnostic"] == pytest.approx(2.75 / 3.0)
    json.dumps(report, allow_nan=False)


def test_model_only_completion_gate_passes_at_exactly_160_of_400_tasks() -> None:
    records, expected_queries = _model_only_completion_fixture(160)

    report = assess_model_only_completion(records, expected_queries)

    assert report["primary_candidate_mode"] == "model_only"
    assert report["required_task_count"] == 400
    assert report["evaluated_task_count"] == 400
    assert report["evaluated_query_count"] == 401
    assert report["required_exact_task_count"] == 160
    assert report["exact_task_count"] == 160
    assert report["strict_task_pass_at_2"] == 0.4
    assert report["passed"] is True
    assert report["tasks"]["task-000"] == {"query_count": 2, "pass_at_2": True}
    json.dumps(report, allow_nan=False)


def test_model_only_completion_gate_fails_at_159_of_400_tasks() -> None:
    records, expected_queries = _model_only_completion_fixture(159)

    report = assess_model_only_completion(records, expected_queries)

    assert report["exact_task_count"] == 159
    assert report["strict_task_pass_at_2"] == 159 / 400
    assert report["passed"] is False


def test_model_only_completion_requires_every_official_query_to_pass_at_two() -> None:
    records, expected_queries = _model_only_completion_fixture(160)
    second_query = next(
        record
        for record in records
        if record["task_id"] == "task-000" and record["query_index"] == 1
    )
    second_query["score"]["pass_at_2"] = False

    report = assess_model_only_completion(records, expected_queries)

    assert report["tasks"]["task-000"]["pass_at_2"] is False
    assert report["exact_task_count"] == 159
    assert report["passed"] is False


def test_model_only_completion_requires_exactly_400_expected_tasks() -> None:
    records, expected_queries = _model_only_completion_fixture(160)
    expected_queries.pop("task-399")

    with pytest.raises(ValueError, match="exactly 400"):
        assess_model_only_completion(records, expected_queries)


@pytest.mark.parametrize(
    ("expected", "records", "message"),
    [
        (None, [], "mapping"),
        ({"": 1}, [], "nonempty"),
        ({"task": 0}, [], "official query"),
        ({"task": 1}, "not records", "sequence"),
    ],
)
def test_model_only_completion_rejects_malformed_manifest(
    expected: object, records: object, message: str
) -> None:
    if expected is not None and len(expected) < 400:  # type: ignore[arg-type]
        expected = {
            **expected,  # type: ignore[misc]
            **{f"task-{index:03d}": 1 for index in range(400 - len(expected))},  # type: ignore[arg-type]
        }
    with pytest.raises(ValueError, match=message):
        assess_model_only_completion(records, expected)  # type: ignore[arg-type]


def test_model_only_completion_rejects_malformed_candidate_provenance() -> None:
    records, expected_queries = _model_only_completion_fixture(160)
    candidate = records[0]["candidates"][0]
    candidate.pop("height")
    with pytest.raises(ValueError, match="missing provenance"):
        assess_model_only_completion(records, expected_queries)

    records, expected_queries = _model_only_completion_fixture(160)
    candidate = records[0]["candidates"][0]
    candidate["width"] = 2
    with pytest.raises(ValueError, match="declared shape"):
        assess_model_only_completion(records, expected_queries)

    records, expected_queries = _model_only_completion_fixture(160)
    records[0]["score"] = None
    with pytest.raises(ValueError, match="score must be a mapping"):
        assess_model_only_completion(records, expected_queries)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing_query", "missing official query"),
        ("duplicate_query", "duplicate task/query"),
        ("non_model", "model provenance"),
        ("one_candidate", "exactly two candidates"),
        ("duplicate_candidates", "distinct"),
        ("invalid_role", "selection_role"),
        ("invalid_source", "earlier candidate checkpoint"),
        ("wrong_mode", "primary_candidate_mode"),
        ("score_identity", "score identity"),
        ("score_candidate_count", "candidate_count"),
    ],
)
def test_model_only_completion_rejects_incomplete_or_invalid_records(
    case: str, message: str
) -> None:
    records, expected_queries = _model_only_completion_fixture(160)
    first = records[0]
    if case == "missing_query":
        records.pop()
    elif case == "duplicate_query":
        records.append(copy.deepcopy(first))
    elif case == "non_model":
        first["candidates"][0]["provenance"] = "verified_rule"
    elif case == "one_candidate":
        first["candidates"] = first["candidates"][:1]
    elif case == "duplicate_candidates":
        first["candidates"][1]["grid"] = copy.deepcopy(first["candidates"][0]["grid"])
    elif case == "invalid_role":
        first["candidates"][0]["selection_role"] = "earlier_sweep_joint_argmax"
    elif case == "invalid_source":
        first["candidates"][1]["selection_role"] = "earlier_sweep_joint_argmax"
        first["candidates"][1]["changed_decision"] = None
        first["candidates"][1]["source_checkpoint"] = 60
    elif case == "wrong_mode":
        first["primary_candidate_mode"] = "verified_rules"
    elif case == "score_identity":
        first["score"]["task_id"] = "another-task"
    elif case == "score_candidate_count":
        first["score"]["candidate_count"] = 1

    with pytest.raises(ValueError, match=message):
        assess_model_only_completion(records, expected_queries)


def test_aggregate_rejects_empty_duplicate_and_wrong_type() -> None:
    score = QueryScore("a", 0, False, False, False, 0.0, 1)
    with pytest.raises(ValueError, match="at least one"):
        aggregate_arc_metrics([])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_arc_metrics([score, score])
    with pytest.raises(ValueError, match="QueryScore"):
        aggregate_arc_metrics([object()])  # type: ignore[list-item]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"task_id": "", "query_index": 0}, "task_id"),
        ({"task_id": "a", "query_index": -1}, "query_index"),
        ({"task_id": "a", "query_index": True}, "query_index"),
    ],
)
def test_query_scorer_validates_identity(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        score_query_candidates([[[1]]], [[1]], **kwargs)  # type: ignore[arg-type]


def test_query_score_rejects_inconsistent_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="pass_at_2"):
        QueryScore("a", 0, True, False, True, 1.0, 1)
    with pytest.raises(ValueError, match="boolean"):
        QueryScore("a", 0, 1, True, True, 1.0, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pixel"):
        QueryScore("a", 0, False, False, False, np.nan, 1)
    with pytest.raises(ValueError, match="one or two"):
        QueryScore("a", 0, False, False, False, 0.0, 3)


def test_candidate_validates_grid_metadata_and_serializes() -> None:
    candidate = DecodedCandidate([[0, 9]], "cell:0,1", -1.25)
    assert candidate.to_dict() == {
        "height": 1,
        "width": 2,
        "grid": [[0, 9]],
        "changed_decision": "cell:0,1",
        "log_probability": -1.25,
    }
    with pytest.raises(ValueError, match="changed_decision"):
        DecodedCandidate([[1]], 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="log_probability"):
        DecodedCandidate([[1]], log_probability=np.inf)


def test_fixed_trajectory_reports_zero_displacement_and_convergence() -> None:
    grid = np.array([[3, 4]])
    height, width, colors = _stacked_logits([grid, grid])
    spikes = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])
    voltages = np.array([[0.1, -0.2, 0.3, -0.4]] * 2)

    report = analyze_latent_trajectory(
        height,
        width,
        colors,
        spikes,
        voltages,
        target=grid,
        task_id="fixed",
        step_indices=[0, 8],
    )

    assert report["converged_steps"] == [8]
    second = report["steps"][1]
    assert second["changed_cell_count"] == 0
    assert second["spike_hamming_displacement"] == 0
    assert second["voltage_l2_displacement"] == 0.0
    assert second["converged"]
    assert second["score"]["pass_at_1"]
    assert second["candidates"][0]["grid"] == [[3, 4]]
    json.dumps(report, allow_nan=False)


def test_trajectory_reports_separate_synaptic_currents_and_uses_them_for_convergence() -> (
    None
):
    grid = np.array([[3]])
    height, width, colors = _stacked_logits([grid, grid])
    spikes = np.zeros((2, 2), dtype=np.int8)
    voltages = np.zeros((2, 2), dtype=np.float32)
    feedforward = np.array([[1.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    recurrent = np.array([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32)

    report = analyze_latent_trajectory(
        height,
        width,
        colors,
        spikes,
        voltages,
        feedforward_current=feedforward,
        recurrent_current=recurrent,
    )

    second = report["steps"][1]
    assert second["feedforward_current_l2"] == 0.5
    assert second["feedforward_current_l2_displacement"] == 0.5
    assert second["recurrent_current_l2"] == 2.0
    assert second["recurrent_current_l2_displacement"] == 0.0
    assert not second["converged"]
    assert report["steps"][0]["state_sha256"] != second["state_sha256"]


def test_trajectory_exposes_changed_cells_saturation_and_silence() -> None:
    first = np.array([[1, 2]])
    second = np.array([[1, 7]])
    height, width, colors = _stacked_logits([first, second, second])
    spikes = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0]])
    voltages = np.array([[0.0] * 4, [1.0] * 4, [0.0] * 4])

    report = analyze_latent_trajectory(
        height, width, colors, spikes, voltages, step_indices=[0, 1, 2]
    )

    assert report["near_silence_steps"] == [0, 2]
    assert report["near_saturation_steps"] == [1]
    assert report["steps"][1]["changed_cell_count"] == 1
    assert report["steps"][1]["changed_cell_fraction"] == 0.5
    assert report["steps"][1]["spike_hamming_fraction"] == 1.0
    assert report["steps"][1]["voltage_l2_displacement"] == 2.0
    assert report["steps"][1]["raster_active_indices"] == [0, 1, 2, 3]
    assert report["firing_rate_distribution"] == {
        "minimum": 0.0,
        "mean": pytest.approx(1.0 / 3.0),
        "maximum": 1.0,
    }


def test_shape_change_counts_added_and_removed_cells() -> None:
    height, width, colors = _stacked_logits(
        [np.array([[1]]), np.array([[1, 0], [0, 0]])]
    )
    report = analyze_latent_trajectory(
        height, width, colors, [[0], [0]], [[0.0], [0.0]]
    )

    assert report["steps"][1]["changed_cell_count"] == 3
    assert report["steps"][1]["changed_cell_fraction"] == 0.75


def test_voltage_tolerance_can_mark_numerically_stable_transition() -> None:
    height, width, colors = _stacked_logits([np.array([[1]]), np.array([[1]])])
    report = analyze_latent_trajectory(
        height,
        width,
        colors,
        [[0, 0], [0, 0]],
        [[0.0, 0.0], [1e-7, -1e-7]],
        convergence_atol=1e-6,
    )

    assert report["steps"][1]["converged"]
    assert report["steps"][1]["voltage_mean_absolute"] == 1e-7
    assert report["steps"][1]["state_sha256"] != report["steps"][0]["state_sha256"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("height_shape", "height logit sequence shape"),
        ("width_steps", "width logit sequence shape"),
        ("color_shape", "color logit sequence shape"),
        ("empty", "at least one step"),
        ("nonfinite", "non-finite"),
        ("spike_steps", "spikes have"),
        ("spike_nonbinary", "binary"),
        ("voltage_shape", "voltages shape"),
    ],
)
def test_trajectory_rejects_malformed_logits_and_states(
    mutation: str, message: str
) -> None:
    height, width, colors = _stacked_logits([np.array([[1]])])
    spikes: object = [[0, 1]]
    voltages: object = [[0.0, 1.0]]
    if mutation == "height_shape":
        height = np.zeros((1, 29))
    elif mutation == "width_steps":
        width = np.zeros((2, 30))
    elif mutation == "color_shape":
        colors = np.zeros((1, 30, 30, 9))
    elif mutation == "empty":
        height = np.zeros((0, 30))
        width = np.zeros((0, 30))
        colors = np.zeros((0, 30, 30, 10))
    elif mutation == "nonfinite":
        height[0, 0] = np.inf
    elif mutation == "spike_steps":
        spikes = [[0, 1], [0, 1]]
    elif mutation == "spike_nonbinary":
        spikes = [[0, 2]]
    elif mutation == "voltage_shape":
        voltages = [[0.0]]

    with pytest.raises(ValueError, match=message):
        analyze_latent_trajectory(height, width, colors, spikes, voltages)


def test_trajectory_rejects_partial_currents_and_mismatched_current_shapes() -> None:
    height, width, colors = _stacked_logits([np.array([[1]])])
    spikes = np.zeros((1, 2), dtype=np.int8)
    voltages = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="provided together"):
        analyze_latent_trajectory(
            height,
            width,
            colors,
            spikes,
            voltages,
            feedforward_current=np.zeros((1, 2)),
        )
    with pytest.raises(ValueError, match="feedforward synaptic current shape"):
        analyze_latent_trajectory(
            height,
            width,
            colors,
            spikes,
            voltages,
            feedforward_current=np.zeros((1, 3)),
            recurrent_current=np.zeros((1, 2)),
        )


def test_control_comparison_rejects_unmatched_or_empty_current_mappings() -> None:
    states = np.zeros((1, 2), dtype=np.float32)
    spikes = np.zeros((1, 2), dtype=np.int8)
    with pytest.raises(ValueError, match="provided together"):
        compare_control_trajectories(
            spikes,
            states,
            spikes,
            states,
            control_name="partial",
            intact_synaptic_currents={"feedforward": states},
        )
    with pytest.raises(ValueError, match="identical nonempty"):
        compare_control_trajectories(
            spikes,
            states,
            spikes,
            states,
            control_name="empty",
            intact_synaptic_currents={},
            control_synaptic_currents={},
        )
    with pytest.raises(ValueError, match="identical nonempty"):
        compare_control_trajectories(
            spikes,
            states,
            spikes,
            states,
            control_name="different-names",
            intact_synaptic_currents={"feedforward": states},
            control_synaptic_currents={"recurrent": states},
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"step_indices": [1, 1]}, "strictly increasing"),
        ({"step_indices": [0]}, "2 integers"),
        ({"convergence_atol": -1.0}, "nonnegative"),
        ({"silence_rate": 0.96, "saturation_rate": 0.95}, "smaller"),
        ({"silence_rate": -0.1}, r"in \[0, 1\]"),
        ({"saturation_rate": np.inf}, r"in \[0, 1\]"),
        ({"raster_neurons": 0}, "positive"),
    ],
)
def test_trajectory_rejects_invalid_options(
    kwargs: dict[str, object], message: str
) -> None:
    height, width, colors = _stacked_logits([np.array([[1]]), np.array([[1]])])
    with pytest.raises(ValueError, match=message):
        analyze_latent_trajectory(
            height,
            width,
            colors,
            [[0], [0]],
            [[0.0], [0.0]],
            **kwargs,
        )


def test_byte_identical_control_is_stated_as_causally_null() -> None:
    spikes = np.array([[0, 1], [1, 0]], dtype=np.int8)
    voltages = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    intact = {"query_pass_at_1": 0.5, "nested": {"pixel": 0.75}}
    control = {"query_pass_at_1": 0.25, "nested": {"pixel": 0.5}}

    report = compare_control_trajectories(
        spikes,
        voltages,
        spikes.copy(),
        voltages.copy(),
        control_name="no-context",
        intact_scores=intact,
        control_scores=control,
    )

    assert report["causally_null_at_measured_precision"]
    assert report["state_byte_identical_by_step"] == [True, True]
    assert report["spike_hamming_by_step"] == [0, 0]
    assert report["voltage_l2_by_step"] == [0.0, 0.0]
    assert report["score_deltas_control_minus_intact"] == {
        "nested.pixel": -0.25,
        "query_pass_at_1": -0.25,
    }
    assert "causally null at measured precision" in report["interpretation"]
    json.dumps(report, allow_nan=False)


def test_control_change_reports_state_distances_and_non_null_interpretation() -> None:
    report = compare_control_trajectories(
        [[0, 0], [0, 1]],
        [[0.0, 0.0], [1.0, 1.0]],
        [[0, 0], [1, 1]],
        [[0.0, 0.0], [1.0, 3.0]],
        control_name="slot-ablation",
    )

    assert not report["causally_null_at_measured_precision"]
    assert report["state_byte_identical_by_step"] == [True, False]
    assert report["spike_hamming_by_step"] == [0, 1]
    assert report["spike_hamming_fraction_by_step"] == [0.0, 0.5]
    assert report["voltage_l2_by_step"] == [0.0, 2.0]
    assert "changed measured latent states" in report["interpretation"]


def test_synaptic_current_change_prevents_false_causal_null() -> None:
    zeros = np.zeros((2, 3), dtype=np.float32)
    changed_recurrent = zeros.copy()
    changed_recurrent[1, 0] = 2.0

    report = compare_control_trajectories(
        np.zeros((2, 3), dtype=np.int8),
        zeros,
        np.zeros((2, 3), dtype=np.int8),
        zeros.copy(),
        control_name="current-only-change",
        intact_synaptic_currents={
            "feedforward": zeros,
            "recurrent": zeros,
        },
        control_synaptic_currents={
            "feedforward": zeros.copy(),
            "recurrent": changed_recurrent,
        },
    )

    assert not report["causally_null_at_measured_precision"]
    assert report["state_byte_identical_by_step"] == [True, False]
    assert report["synaptic_current_l2_by_step"] == {
        "feedforward": [0.0, 0.0],
        "recurrent": [0.0, 2.0],
    }


def test_numeric_equality_with_different_dtype_is_not_byte_identity() -> None:
    report = compare_control_trajectories(
        np.array([[0, 1]], dtype=np.int8),
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([[0, 1]], dtype=np.int64),
        np.array([[0.0, 1.0]], dtype=np.float64),
        control_name="dtype-control",
    )

    assert not report["causally_null_at_measured_precision"]
    assert report["spike_hamming_by_step"] == [0]
    assert report["voltage_l2_by_step"] == [0.0]


def test_control_comparison_rejects_shape_name_and_unpaired_scores() -> None:
    with pytest.raises(ValueError, match="control_name"):
        compare_control_trajectories([[0]], [[0.0]], [[0]], [[0.0]], control_name="")
    with pytest.raises(ValueError, match="shape must match"):
        compare_control_trajectories(
            [[0]], [[0.0]], [[0, 0]], [[0.0, 0.0]], control_name="bad-shape"
        )
    with pytest.raises(ValueError, match="provided together"):
        compare_control_trajectories(
            [[0]],
            [[0.0]],
            [[0]],
            [[0.0]],
            control_name="scores",
            intact_scores={"pass": 1.0},
        )


def test_adam_parameter_travel_budget_flags_a_starved_operating_point() -> None:
    """The shipped operating point cannot move a head weight one sigma.

    Adam's per-coordinate displacement per step is bounded by the learning rate,
    because ``|m_hat / sqrt(v_hat)| <= 1``, so over ``U`` updates no weight can
    travel further than ``learning_rate * U``. At ``1e-4`` and 260 updates that
    is 0.026 against an answer-head initialisation sigma of ``1/sqrt(1414)``, so
    the trained head is a perturbed random initialisation and the decoder can
    only emit the dataset prior. Seven ARC-AGI-1 runs were executed at this
    operating point before anything surfaced the arithmetic.
    """
    budget = analysis.adam_parameter_travel_budget(1e-4, 260, 1414)

    assert budget["head_width"] == 1414
    assert budget["displacement_bound"] == pytest.approx(0.026)
    assert budget["initialization_sigma"] == pytest.approx(1.0 / math.sqrt(1414))
    assert budget["sigmas_of_travel"] == pytest.approx(0.026 * math.sqrt(1414))
    assert budget["sigmas_of_travel"] < 1.0
    assert budget["starved"] is True

    raised = analysis.adam_parameter_travel_budget(3e-3, 260, 1414)

    assert raised["sigmas_of_travel"] > 25.0
    assert raised["starved"] is False


def test_adam_parameter_travel_budget_rejects_nonpositive_inputs() -> None:
    """A travel budget is undefined without a rate, updates, and a width."""
    with pytest.raises(ValueError):
        analysis.adam_parameter_travel_budget(0.0, 260, 1414)
    with pytest.raises(ValueError):
        analysis.adam_parameter_travel_budget(1e-4, 0, 1414)
    with pytest.raises(ValueError):
        analysis.adam_parameter_travel_budget(1e-4, 260, 0)
