"""Tests for the Example 21 event budget derivation and dry run."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from examples.pp_prop import example21_event_budget as budget
from examples.pp_prop.example21_evolve import PipelineConfig, build_update_schedule

CORPUS_TASKS = 400


def _demonstrations(index: int) -> int:
    """Return a deterministic demonstration count in 1..4 for one task index."""

    return 1 + index % 4


def _queries(index: int) -> int:
    """Give every sixteenth task two queries, as the real corpus has some."""

    return 2 if index % 16 == 0 else 1


def _write_corpus(root: Path) -> Path:
    directory = root / "data" / "training"
    directory.mkdir(parents=True)
    for index in range(CORPUS_TASKS):
        payload = {
            "train": [{"input": [[0, 1]], "output": [[1]]}] * _demonstrations(index),
            "test": [{"input": [[2, 2], [2, 2]], "output": [[3]]}] * _queries(index),
        }
        (directory / f"{index:08x}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    return root


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory):
    root = _write_corpus(tmp_path_factory.mktemp("arc"))
    manifest, queries = budget.corpus_query_events(root)
    return root, manifest, queries


def test_advancing_events_follow_the_encoder_layout() -> None:
    assert budget.advancing_events(0) == 65
    assert budget.advancing_events(2) == 193
    assert budget.advancing_events(10) == 705
    assert budget.QUERY_FIXED_EVENTS == 1 + 32 + 2 + 30
    with pytest.raises(ValueError, match="nonnegative"):
        budget.advancing_events(-1)


def test_corpus_query_events_count_the_real_encoder_output(corpus) -> None:
    _, manifest, queries = corpus
    assert len(manifest.task_ids) == CORPUS_TASKS
    assert [(q.task_id, q.query_index) for q in queries] == list(manifest.query_order)
    assert len(queries) == sum(_queries(index) for index in range(CORPUS_TASKS))
    for query in queries:
        index = int(query.task_id, 16)
        assert query.demonstrations == _demonstrations(index)
        assert query.advancing == 65 + 64 * _demonstrations(index)


def test_scoring_and_block_events_sum_the_scheduled_queries(corpus) -> None:
    _, manifest, queries = corpus
    scope = manifest.task_ids[:8]
    count, events = budget.scoring_events(queries, scope)
    expected = [q for q in queries if q.task_id in scope]
    assert count == len(expected) == 9  # task 0 carries two queries
    assert events == sum(q.advancing for q in expected)

    schedule = build_update_schedule(manifest, 128, 128)
    lookup = {(q.task_id, q.query_index): q.advancing for q in queries}
    assert budget.block_events(manifest, queries, 128) == sum(
        lookup[(entry.task_id, entry.query_index)] for entry in schedule.entries
    )


def test_first_round_stages_follow_the_lifecycle_and_the_scope(corpus) -> None:
    _, manifest, queries = corpus
    unscreened = budget.first_round_stages(
        manifest, queries, PipelineConfig(screen_tasks=0)
    )
    assert [stage.stage for stage in unscreened] == [
        "initialize",
        "train",
        "edge-arm0",
        "edge-arm1",
        "neuron-arm0",
        "neuron-arm1",
    ]
    corpus_events = sum(q.advancing for q in queries)
    assert all(stage.scoring_events == corpus_events for stage in unscreened)
    assert unscreened[0].updates == 0 and unscreened[0].update_events == 0
    assert unscreened[1].update_events == budget.block_events(manifest, queries, 0)
    assert unscreened[2].update_events == budget.block_events(manifest, queries, 128)
    assert unscreened[4].update_events == budget.block_events(manifest, queries, 256)
    assert unscreened[1].events == unscreened[1].update_events + corpus_events

    scoped = budget.first_round_stages(
        manifest, queries, PipelineConfig(score_tasks=8, screen_tasks=2)
    )
    assert [stage.stage for stage in scoped] == [
        "initialize",
        "train",
        "round-screen",
        "edge-arm0",
        "edge-arm1",
        "neuron-arm0",
        "neuron-arm1",
        "round-score",
    ]
    _, scope_events = budget.scoring_events(queries, manifest.task_ids[:8])
    _, screen_events = budget.scoring_events(queries, manifest.task_ids[:2])
    assert scoped[0].scoring_events == scoped[-1].scoring_events == scope_events
    assert scoped[2].scoring_events == scoped[3].scoring_events == screen_events
    # The scope shrinks the scoring passes and leaves every block unchanged.
    assert scoped[1].update_events == unscreened[1].update_events


def test_event_budget_prices_stages_and_reports_what_a_cap_admits(corpus) -> None:
    _, manifest, queries = corpus
    costs = {
        "forward_seconds_per_event": 1.0,
        "learner_seconds_per_event": 10.0,
        "fixed_seconds": 100.0,
    }
    config = PipelineConfig(score_tasks=8, screen_tasks=2)
    result = budget.event_budget(
        manifest, queries, config, costs=costs, cap_seconds=3600.0
    )

    assert result["config"]["score_tasks"] == 8
    assert result["training_manifest_sha256"] == manifest.digest
    assert result["scope"] == {"score_tasks": 8, "screen_tasks": 2}
    assert result["corpus"]["min_advancing_per_query"] == 65 + 64
    assert result["block"]["floor_advancing_events"] == 128 * 129
    assert result["block"]["floor_seconds"] == 100.0 + 128 * 129 * 10.0
    rows = {row["stage"]: row for row in result["stages"]}
    initialize = rows["initialize"]
    assert initialize["seconds"] == 100.0 + initialize["scoring_events"]
    assert initialize["fits_cap"] and not rows["train"]["fits_cap"]
    assert rows["train"]["seconds"] == (
        100.0 + 10.0 * rows["train"]["update_events"] + rows["train"]["scoring_events"]
    )
    assert result["first_round_seconds"] == sum(
        row["seconds"] for row in result["stages"]
    )
    assert result["updates_under_cap"] == int((3600.0 - 100.0) // (129 * 10.0))
    largest = result["largest_score_tasks_under_cap"]
    _, fitting = budget.scoring_events(queries, manifest.task_ids[:largest])
    _, overflowing = budget.scoring_events(queries, manifest.task_ids[: largest + 1])
    assert 100.0 + fitting <= 3600.0 < 100.0 + overflowing

    unaffordable = budget.event_budget(
        manifest,
        queries,
        config,
        costs={**costs, "fixed_seconds": 4000.0},
        cap_seconds=3600.0,
    )
    assert unaffordable["updates_under_cap"] == 0
    assert unaffordable["largest_score_tasks_under_cap"] == 0
    assert (
        "score_tasks"
        not in budget.event_budget(manifest, queries, PipelineConfig(), costs=costs)[
            "config"
        ]
    )


def test_render_budget_lists_every_stage_and_the_cap_verdicts(corpus) -> None:
    _, manifest, queries = corpus
    result = budget.event_budget(
        manifest,
        queries,
        PipelineConfig(score_tasks=4, screen_tasks=2),
        costs=budget.PROFILE_PINNED,
        cap_seconds=3600.0,
    )
    text = budget.render_budget(result)
    for row in result["stages"]:
        assert row["stage"] in text
    assert "score_tasks=4 screen_tasks=2" in text
    assert "largest score_tasks whose initialize fits the cap" in text
    assert "a block needs 128" in text


def test_cli_prints_the_budget_and_requires_launch_arguments(
    corpus, capsys, monkeypatch
) -> None:
    root, manifest, queries = corpus
    assert (
        budget.main(
            [
                "--arc-root",
                str(root),
                "--score-tasks",
                "8",
                "--screen-tasks",
                "2",
                "--profile",
                "dt0005",
                "--fixed-seconds",
                "10",
                "--json",
            ]
        )
        == 0
    )
    printed = json.loads(capsys.readouterr().out)
    expected = budget.event_budget(
        manifest,
        queries,
        PipelineConfig(score_tasks=8, screen_tasks=2),
        costs={**budget.PROFILE_DT0005, "fixed_seconds": 10.0},
    )
    assert printed == json.loads(json.dumps(expected))

    assert budget.main(["budget", "--arc-root", str(root), "--screen-tasks", "0"]) == 0
    assert "stage" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exit_info:
        budget.main(["evolve", "--arc-root", str(root), "--score-tasks", "8"])
    assert exit_info.value.code == 2
    assert "--output-dir" in capsys.readouterr().err

    launched = {}

    def fake_run(adapter, output_dir, *, config, progress_reporter):
        launched["config"] = config
        launched["adapter"] = adapter
        raise KeyboardInterrupt("launched")

    import brainstate

    from examples.pp_prop import example21_evolve, h01_arc_adapter

    monkeypatch.setattr(example21_evolve, "run_evolution", fake_run)
    monkeypatch.setattr(
        h01_arc_adapter,
        "H01ArcAdapter",
        lambda arc_root, manifest: (arc_root, manifest),
    )
    monkeypatch.setattr(brainstate.environ, "context", lambda **kwargs: _Null())
    with pytest.raises(KeyboardInterrupt, match="launched"):
        budget.main(
            [
                "evolve",
                "--arc-root",
                str(root),
                "--score-tasks",
                "8",
                "--screen-tasks",
                "2",
                "--output-dir",
                str(root / "run"),
                "--h01-manifest",
                str(root / "manifest.json"),
            ]
        )
    assert launched["config"] == PipelineConfig(score_tasks=8, screen_tasks=2)
    assert launched["adapter"] == (root, root / "manifest.json")


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _real_arc_root() -> Path | None:
    candidates = [os.environ.get("EXAMPLE21_ARC_ROOT")]
    candidates.append(str(Path(__file__).resolve().parents[2] / "var" / "arc-agi-1"))
    for candidate in candidates:
        if candidate and (Path(candidate) / "data" / "training").is_dir():
            return Path(candidate)
    return None


@pytest.mark.skipif(
    _real_arc_root() is None, reason="ARC-AGI-1 corpus is not checked out"
)
def test_real_corpus_reproduces_the_profiled_event_counts() -> None:
    manifest, queries = budget.corpus_query_events(_real_arc_root())
    result = budget.event_budget(
        manifest, queries, PipelineConfig(), costs=budget.PROFILE_PINNED
    )
    assert result["corpus"]["tasks"] == 400 and result["corpus"]["queries"] == 416
    assert result["corpus"]["advancing_events"] == 116_128
    assert result["corpus"]["min_advancing_per_query"] == 193
    assert result["corpus"]["max_advancing_per_query"] == 705
    assert result["block"]["advancing_events_from_cursor_0"] == 35_648
    assert budget.scoring_events(queries, manifest.task_ids[:2]) == (2, 770)
    assert manifest.task_ids[:2] == ("007bbfb7", "00d62c1b")
    assert result["stages"][0]["seconds"] == pytest.approx(158_052.08, abs=0.01)
    assert result["stages"][1]["update_events"] == 35_648
    assert result["stages"][1]["seconds"] == pytest.approx(627_906.37, abs=0.01)
