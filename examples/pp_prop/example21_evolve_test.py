"""Tests for the adapter-driven Example 21 evolution lifecycle."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from examples.pp_prop import example21_evolve as evolve
from examples.pp_prop.example21_evolve import (
    DEFAULT_MAX_CHECKPOINT_BYTES,
    DEFAULT_MAX_NEURONS,
    DEFAULT_MAX_RECURRENT_EDGES,
    DEFAULT_OPTIMIZER,
    DEFAULT_PATIENCE,
    DEFAULT_ROUNDS,
    DEFAULT_UPDATES,
    PROOF_UPDATES,
    CandidateAttempt,
    CandidateSnapshot,
    CorpusManifest,
    PipelineConfig,
    PipelineError,
    PipelineStore,
    ProgressConflictError,
    ResourceUsage,
    ResumeMismatchError,
    RunState,
    ScoreSnapshot,
    StageContext,
    build_update_schedule,
    plot_score_history,
    run_evolution,
    select_candidate,
)


def _manifest(role: str = "training") -> CorpusManifest:
    task_ids = tuple(f"{index:08x}" for index in range(400))
    return CorpusManifest(
        role=role,
        task_ids=task_ids,
        source_digests=tuple(f"{index:064x}" for index in range(400)),
        query_order=tuple((task_id, 0) for task_id in task_ids),
    )


def _score(
    exact_count: int, loss: float = 10.0, *, finite: bool = True
) -> ScoreSnapshot:
    task_ids = _manifest().task_ids
    return ScoreSnapshot(
        task_ids=task_ids,
        task_exact=tuple(index < exact_count for index in range(400)),
        task_loss=tuple(0.0 if index < exact_count else loss for index in range(400)),
        finite=finite,
    )


def _terminal_result(exact_count: int = 0) -> dict[str, object]:
    score = _score(exact_count)
    return {
        "task_count": len(score.task_ids),
        "strict_task_pass_at_1_count": score.exact_count,
        "task_ids": list(score.task_ids),
        "task_exact": list(score.task_exact),
        "task_loss": list(score.task_loss),
        "mean_unresolved_task_loss": score.unresolved_loss,
        "finite": score.finite,
    }


def _identity(kind: str, name: str) -> str:
    return hashlib.sha256(f"{kind}:{name}".encode()).hexdigest()


def _candidate(
    name: str,
    *,
    exact_count: int = 0,
    loss: float = 10.0,
    persistent_bytes: int = 1_000,
    neurons: int = 100,
    edges: int = 200,
    topology_changed: bool = True,
    finite: bool = True,
) -> CandidateSnapshot:
    return CandidateSnapshot(
        candidate_id=name,
        checkpoint_path=f"temporary/{name}.npz",
        checkpoint_sha256=_identity("checkpoint", name),
        topology_sha256=_identity("topology", name),
        parameters_sha256=_identity("parameters", name),
        optimizer_sha256=_identity("muon", name),
        score=_score(exact_count, loss, finite=finite),
        resources=ResourceUsage(
            persistent_bytes=persistent_bytes,
            checkpoint_bytes=persistent_bytes // 2,
            neurons=neurons,
            recurrent_edges=edges,
        ),
        topology_changed=topology_changed,
    )


def test_defaults_use_muon_and_128_updates_without_changing_proof() -> None:
    config = PipelineConfig()
    assert DEFAULT_OPTIMIZER == config.optimizer == "muon"
    assert DEFAULT_UPDATES == config.updates == 128
    assert DEFAULT_ROUNDS == config.rounds == 8
    assert DEFAULT_PATIENCE == config.patience == 2
    assert DEFAULT_MAX_NEURONS == config.max_neurons == 4_096
    assert DEFAULT_MAX_RECURRENT_EDGES == config.max_recurrent_edges == 65_536
    assert PROOF_UPDATES == 8
    with pytest.raises(ValueError, match="exactly 128"):
        PipelineConfig(updates=64)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("max_neurons", DEFAULT_MAX_NEURONS + 1),
        ("max_recurrent_edges", DEFAULT_MAX_RECURRENT_EDGES + 1),
        ("max_checkpoint_bytes", DEFAULT_MAX_CHECKPOINT_BYTES + 1),
    ),
)
def test_config_cannot_loosen_shipped_resource_maxima(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="shipped maximum"):
        PipelineConfig(**{field: value})


def test_direct_config_and_attempt_update_types_fail_closed() -> None:
    with pytest.raises(TypeError, match="JSON integers"):
        PipelineConfig(rounds=True)
    with pytest.raises(TypeError, match="JSON integers"):
        PipelineConfig(max_neurons=100.0)
    candidate = _candidate("updates")
    with pytest.raises(ValueError, match="exactly 128"):
        CandidateAttempt.completed("add", candidate, executed_updates=127)
    with pytest.raises(ValueError, match="zero"):
        CandidateAttempt.blocked("add", "cap", executed_updates=1)
    with pytest.raises(ValueError, match="0 to 128"):
        CandidateAttempt.failed("add", "partial", executed_updates=True)
    assert (
        CandidateAttempt.failed("add", "partial", executed_updates=64).executed_updates
        == 64
    )


def test_persisted_json_rejects_scalar_and_container_coercion() -> None:
    config = PipelineConfig().to_dict()
    config["updates"] = 128.9
    with pytest.raises(TypeError, match="updates"):
        PipelineConfig.from_dict(config)

    config = PipelineConfig().to_dict()
    config["optimizer"] = 1
    with pytest.raises(TypeError, match="optimizer"):
        PipelineConfig.from_dict(config)

    score = _score(0).to_dict()
    score["task_exact"] = ["false"] * 400
    with pytest.raises(TypeError, match="task_exact"):
        ScoreSnapshot.from_dict(score)

    score = _score(0).to_dict()
    score["task_loss"] = [1] * 400
    with pytest.raises(TypeError, match="task_loss"):
        ScoreSnapshot.from_dict(score)

    manifest = _manifest().to_dict()
    manifest["query_order"] = tuple(manifest["query_order"])
    with pytest.raises(TypeError, match="query_order"):
        CorpusManifest.from_dict(manifest)

    state = RunState.initial(
        PipelineConfig(rounds=1), _manifest(), _candidate("strict-state")
    ).to_dict()
    state["closed"] = 0
    with pytest.raises(TypeError, match="closed"):
        RunState.from_dict(state)


def test_public_apis_use_numpy_docstring_contracts() -> None:
    source_path = Path(__file__).with_name("example21_evolve.py")
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in module.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        docstring = ast.get_docstring(node) or ""
        if isinstance(node, ast.ClassDef):
            has_class_contract = any(
                section in docstring
                for section in (
                    "\nAttributes\n----------",
                    "\nParameters\n----------",
                )
            )
            assert has_class_contract, node.name
            continue
        parameters = [
            argument.arg
            for argument in (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
        ]
        if parameters:
            assert "\nParameters\n----------" in docstring, node.name
            for parameter in parameters:
                assert f"\n{parameter} :" in docstring, (node.name, parameter)
        returns_value = any(
            isinstance(child, ast.Return) and child.value is not None
            for child in ast.walk(node)
        )
        if returns_value:
            assert "\nReturns\n-------" in docstring, node.name


def test_manifest_and_schedule_are_sorted_complete_and_cursor_stable() -> None:
    manifest = _manifest()
    manifest.validate(expected_tasks=400)
    schedule = build_update_schedule(manifest, cursor=398, updates=128)
    assert len(schedule.entries) == 128
    assert schedule.cursor_start == 398
    assert schedule.cursor_end == 526
    assert schedule.entries[0].task_id == manifest.task_ids[398]
    assert schedule.entries[2].task_id == manifest.task_ids[0]
    assert schedule.entries[0].ordinal == 398
    assert schedule.entries[-1].ordinal == 525
    assert schedule == build_update_schedule(manifest, cursor=398, updates=128)

    with pytest.raises(ValueError, match="sorted"):
        replace(manifest, task_ids=tuple(reversed(manifest.task_ids))).validate()
    with pytest.raises(ValueError, match="400"):
        replace(manifest, task_ids=manifest.task_ids[:-1]).validate()


def test_config_manifest_and_value_objects_fail_closed_and_round_trip() -> None:
    config = PipelineConfig()
    assert PipelineConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError, match="Muon"):
        PipelineConfig(optimizer="adam")
    with pytest.raises(ValueError, match="positive"):
        PipelineConfig(rounds=0)

    manifest = _manifest()
    assert CorpusManifest.from_dict(manifest.to_dict()) == manifest
    with pytest.raises(ValueError, match="role"):
        replace(manifest, role="other").validate()
    duplicate_ids = (manifest.task_ids[0], manifest.task_ids[0], *manifest.task_ids[2:])
    with pytest.raises(ValueError, match="unique"):
        replace(manifest, task_ids=duplicate_ids).validate()
    with pytest.raises(ValueError, match="digests"):
        replace(
            manifest, source_digests=("bad", *manifest.source_digests[1:])
        ).validate()
    with pytest.raises(ValueError, match="empty"):
        replace(manifest, query_order=()).validate()
    with pytest.raises(ValueError, match="invalid task"):
        replace(
            manifest,
            query_order=(("unknown", 0), *manifest.query_order[1:]),
        ).validate()
    with pytest.raises(ValueError, match="sorted and unique"):
        replace(
            manifest,
            query_order=(manifest.query_order[0], *manifest.query_order),
        ).validate()
    with pytest.raises(ValueError, match="every task"):
        replace(manifest, query_order=manifest.query_order[:-1]).validate()
    malformed = manifest.to_dict()
    malformed["query_order"] = 3
    with pytest.raises(TypeError, match="JSON list"):
        CorpusManifest.from_dict(malformed)

    with pytest.raises(ValueError, match="nonempty"):
        ScoreSnapshot((), (), ())
    with pytest.raises(ValueError, match="align"):
        ScoreSnapshot(("a",), (), (1.0,))
    with pytest.raises(ValueError, match="nonnegative"):
        ScoreSnapshot(("a",), (False,), (-1.0,))
    nonfinite_score = ScoreSnapshot(("a",), (False,), (float("nan"),))
    assert not nonfinite_score.finite
    assert math.isinf(nonfinite_score.unresolved_loss) or math.isnan(
        nonfinite_score.unresolved_loss
    )
    score = _score(2)
    assert ScoreSnapshot.from_dict(score.to_dict()) == score

    with pytest.raises(ValueError, match="nonnegative"):
        ResourceUsage(-1, 0, 1, 0)
    with pytest.raises(ValueError, match="one neuron"):
        ResourceUsage(0, 0, 0, 0)
    candidate = _candidate("round-trip")
    assert CandidateSnapshot.from_dict(candidate.to_dict()) == candidate
    with pytest.raises(ValueError, match="SHA-256"):
        replace(candidate, optimizer_sha256="")
    with pytest.raises(ValueError, match="status"):
        CandidateAttempt("x", "other")
    with pytest.raises(ValueError, match="require"):
        CandidateAttempt("x", "completed")
    assert CandidateAttempt.blocked("x", "cap").status == "blocked"
    assert CandidateAttempt.failed("y", "compile").status == "failed"


def test_schedule_rejects_evaluation_and_invalid_positions() -> None:
    with pytest.raises(ValueError, match="training manifest"):
        build_update_schedule(_manifest("evaluation"), 0)
    with pytest.raises(ValueError, match="nonnegative"):
        build_update_schedule(_manifest(), -1)
    with pytest.raises(ValueError, match="positive"):
        build_update_schedule(_manifest(), 0, updates=0)


def test_selection_protects_solved_tasks_and_uses_loss_then_resources() -> None:
    parent = _candidate("parent", exact_count=2, loss=10.0)
    regressed = _candidate("regressed", exact_count=3, loss=1.0)
    regressed_exact = list(regressed.score.task_exact)
    regressed_exact[0] = False
    regressed = replace(
        regressed,
        score=replace(regressed.score, task_exact=tuple(regressed_exact)),
    )
    too_small = _candidate("too-small", exact_count=2, loss=9.9995)
    protected = _candidate("protected", exact_count=2, loss=9.9989)
    smaller = _candidate("smaller", exact_count=2, loss=10.0, persistent_bytes=900)

    result = select_candidate(
        parent,
        (
            CandidateAttempt.completed("regressed", regressed),
            CandidateAttempt.completed("too-small", too_small),
            CandidateAttempt.completed("protected", protected),
            CandidateAttempt.completed("smaller", smaller),
        ),
    )
    assert result.selected.candidate_id == "protected"
    assert result.dispositions["regressed"] == "rejected-regression"
    assert result.dispositions["too-small"] == "rejected-no-improvement"
    assert result.dispositions["protected"] == "accepted"
    assert result.dispositions["smaller"] == "rejected"


def test_selection_rejects_nonfinite_caps_and_compresses_only_at_mastery() -> None:
    config = PipelineConfig(max_neurons=100, max_recurrent_edges=200)
    parent = _candidate("parent", exact_count=400, persistent_bytes=1_000)
    nonfinite = _candidate("nonfinite", exact_count=400, finite=False)
    oversized = _candidate("oversized", exact_count=400, neurons=101)
    inexact = _candidate("inexact", exact_count=399, persistent_bytes=500)
    compressed = _candidate(
        "compressed", exact_count=400, persistent_bytes=900, neurons=99
    )
    result = select_candidate(
        parent,
        tuple(
            CandidateAttempt.completed(value.candidate_id, value)
            for value in (nonfinite, oversized, inexact, compressed)
        ),
        config=config,
        compression=True,
    )
    assert result.selected.candidate_id == "compressed"
    assert result.dispositions["nonfinite"] == "rejected-nonfinite"
    assert result.dispositions["oversized"] == "rejected-limit"
    assert result.dispositions["inexact"] == "rejected-regression"


def test_selection_records_blocked_failed_mismatch_ties_and_all_caps() -> None:
    parent = _candidate("parent", exact_count=1)
    mismatched_score = ScoreSnapshot(
        task_ids=tuple(reversed(parent.score.task_ids)),
        task_exact=tuple(reversed(parent.score.task_exact)),
        task_loss=tuple(reversed(parent.score.task_loss)),
    )
    mismatch = replace(_candidate("mismatch", exact_count=2), score=mismatched_score)
    edge_cap = _candidate("edge-cap", edges=201)
    checkpoint_cap = replace(
        _candidate("checkpoint-cap"),
        resources=replace(_candidate("checkpoint-cap").resources, checkpoint_bytes=501),
    )
    config = PipelineConfig(
        max_neurons=100, max_recurrent_edges=200, max_checkpoint_bytes=500
    )
    result = select_candidate(
        parent,
        (
            CandidateAttempt.blocked("blocked", "no donor"),
            CandidateAttempt.failed("failed", "compile"),
            CandidateAttempt.completed("mismatch", mismatch),
            CandidateAttempt.completed("edge-cap", edge_cap),
            CandidateAttempt.completed("checkpoint-cap", checkpoint_cap),
        ),
        config=config,
    )
    assert result.parent_retained
    assert result.dispositions == {
        "blocked": "blocked",
        "failed": "failed",
        "mismatch": "rejected-score-mismatch",
        "edge-cap": "rejected-limit",
        "checkpoint-cap": "rejected-limit",
    }
    with pytest.raises(ValueError, match="unique"):
        select_candidate(
            parent,
            (
                CandidateAttempt.blocked("same", "a"),
                CandidateAttempt.failed("same", "b"),
            ),
        )

    not_mastered = select_candidate(
        parent,
        (
            CandidateAttempt.completed(
                "smaller", _candidate("smaller", exact_count=1, persistent_bytes=1)
            ),
        ),
        compression=True,
    )
    assert not_mastered.parent_retained
    mastered = _candidate("mastered", exact_count=400)
    no_compression = select_candidate(
        mastered,
        (CandidateAttempt.completed("same", _candidate("same", exact_count=400)),),
        compression=True,
    )
    assert no_compression.parent_retained


def test_selection_rejects_candidate_aliases() -> None:
    parent = _candidate("parent")
    first = _candidate("first", exact_count=1)
    aliased_path = replace(
        _candidate("second", exact_count=2),
        checkpoint_path=first.checkpoint_path,
    )
    with pytest.raises(ValueError, match="alias"):
        select_candidate(
            parent,
            (
                CandidateAttempt.completed("add", first),
                CandidateAttempt.completed("prune", aliased_path),
            ),
        )
    aliased_digest = replace(
        _candidate("third", exact_count=1),
        checkpoint_sha256=first.checkpoint_sha256,
    )
    with pytest.raises(ValueError, match="alias"):
        select_candidate(
            parent,
            (
                CandidateAttempt.completed("add", first),
                CandidateAttempt.completed("prune", aliased_digest),
            ),
        )


class _Adapter:
    def __init__(self, *, interrupt_at: str | None = None) -> None:
        self.calls: list[tuple] = []
        self.evaluation_calls = 0
        self.interrupt_at = interrupt_at

    def training_manifest(self) -> CorpusManifest:
        self.calls.append(("training-manifest",))
        return _manifest("training")

    def evaluation_manifest(self) -> CorpusManifest:
        self.calls.append(("evaluation-manifest",))
        return _manifest("evaluation")

    def initialize(self, config: PipelineConfig, output_dir: Path) -> CandidateSnapshot:
        self.calls.append(("initialize", config.optimizer, config.updates))
        return _candidate("initial", topology_changed=False)

    def restore(self, candidate: CandidateSnapshot) -> CandidateSnapshot:
        self.calls.append(("restore", candidate.candidate_id))
        return candidate

    def _stage_candidate(self, candidate, context):
        path = context.output_dir / ".candidates" / f"{candidate.candidate_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(candidate.candidate_id.encode())
        return replace(
            candidate,
            checkpoint_path=str(path),
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            resources=replace(
                candidate.resources, checkpoint_bytes=path.stat().st_size
            ),
        )

    def train_parent(
        self,
        parent: CandidateSnapshot,
        schedule,
        context: StageContext,
    ) -> CandidateSnapshot:
        self.calls.append(
            (
                "train-parent",
                parent.candidate_id,
                parent.topology_sha256,
                parent.parameters_sha256,
                parent.optimizer_sha256,
                id(schedule),
                context.stage,
            )
        )
        return self._stage_candidate(
            _candidate(
                f"trained-{context.round_index}",
                exact_count=parent.score.exact_count,
                loss=max(0.0, parent.score.unresolved_loss - 0.01),
                topology_changed=False,
            ),
            context,
        )

    def run_candidate(
        self,
        parent: CandidateSnapshot,
        arm: str,
        schedule,
        context: StageContext,
    ) -> CandidateAttempt:
        self.calls.append(
            (
                "candidate",
                context.stage,
                arm,
                parent.candidate_id,
                parent.topology_sha256,
                parent.parameters_sha256,
                parent.optimizer_sha256,
                id(schedule),
            )
        )
        if self.interrupt_at == context.stage:
            self.interrupt_at = None
            raise KeyboardInterrupt(context.stage)
        winners = {
            "edge-add": (1, 9.0),
            "neuron-add": (2, 8.0),
            "edge-revisit-add": (3, 7.0),
            "dale-excitatory": (4, 6.0),
        }
        key = f"{context.stage}-{arm}"
        if key in winners:
            exact, loss = winners[key]
            return CandidateAttempt.completed(
                arm,
                self._stage_candidate(
                    _candidate(
                        key,
                        exact_count=exact,
                        loss=loss,
                        persistent_bytes=1_000 + exact,
                    ),
                    context,
                ),
            )
        return CandidateAttempt.completed(
            arm,
            self._stage_candidate(
                _candidate(
                    key,
                    exact_count=parent.score.exact_count,
                    loss=parent.score.unresolved_loss + 1.0,
                    topology_changed=False,
                ),
                context,
            ),
        )

    def persist(
        self,
        candidate: CandidateSnapshot,
        destination: Path,
        *,
        parent_checkpoint_sha256: str | None,
        stage_id: str,
    ) -> CandidateSnapshot:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(candidate.candidate_id.encode())
        persisted = replace(
            candidate,
            checkpoint_path=str(destination),
            checkpoint_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        )
        source = Path(candidate.checkpoint_path)
        if source.parent.name == ".candidates":
            source.unlink(missing_ok=True)
        self.calls.append(
            (
                "persist",
                candidate.candidate_id,
                parent_checkpoint_sha256,
                stage_id,
                persisted.checkpoint_sha256,
            )
        )
        return persisted

    def discard(self, attempt: CandidateAttempt) -> None:
        self.calls.append(("discard", attempt.name, attempt.status))

    def render_topology(self, candidate: CandidateSnapshot, output_path: Path) -> None:
        output_path.write_bytes(candidate.topology_sha256.encode())

    def evaluate_terminal(
        self, candidate: CandidateSnapshot, manifest: CorpusManifest
    ) -> dict:
        self.evaluation_calls += 1
        self.calls.append(("evaluate", candidate.candidate_id, manifest.role))
        return _terminal_result(candidate.score.exact_count)


def _history_plotter(_records, output_path: Path) -> None:
    output_path.write_bytes(b"history")


def _pending_fixture() -> tuple[RunState, RunState, dict[str, object]]:
    before = RunState.initial(
        PipelineConfig(rounds=1), _manifest(), _candidate("pending-parent")
    )
    selected = _candidate("pending-child", exact_count=1)
    attempt = CandidateAttempt.completed("training", selected)
    after = replace(
        before,
        sequence=1,
        cursor=128,
        next_stage="edge",
        accepted=selected,
    )
    document = evolve._pending_transition_document(
        before,
        after,
        stage_id="r000-train",
        stage="train",
        parent=before.accepted,
        selected=selected,
        attempts=(attempt,),
        dispositions={"training": "accepted"},
        elapsed_seconds=1.0,
    )
    return before, after, document


def test_pending_transition_rejects_ambiguous_or_corrupt_evidence() -> None:
    before, after, document = _pending_fixture()
    with pytest.raises(ValueError, match="at most one"):
        evolve._pending_transition_document(
            before,
            after,
            stage_id="r000-train",
            stage="train",
            parent=before.accepted,
            selected=after.accepted,
            attempts=(),
            dispositions={"one": "accepted", "two": "accepted"},
            elapsed_seconds=1.0,
        )

    def rejected(mutator) -> None:
        corrupted = json.loads(json.dumps(document))
        mutator(corrupted)
        with pytest.raises(ProgressConflictError, match="inconsistent"):
            evolve._pending_transition_parts(before, corrupted)

    rejected(lambda value: value.update(schema_version=0))
    rejected(lambda value: value.update(config_sha256="0" * 64))
    rejected(lambda value: value.update(sequence_before="0"))
    rejected(lambda value: value.update(stage=""))
    rejected(lambda value: value.update(stage="dale"))
    rejected(lambda value: value.update(stage_id="not-canonical"))
    rejected(lambda value: value.update(elapsed_seconds=-1.0))
    rejected(lambda value: value["attempts"].append(value["attempts"][0]))
    rejected(lambda value: value.update(dispositions={}))
    rejected(
        lambda value: value["attempts"][0].update(
            candidate=_candidate("other-child", exact_count=1).to_dict()
        )
    )
    rejected(
        lambda value: (
            value.update(selected_attempt="missing"),
            value.update(dispositions={"missing": "accepted"}),
        )
    )
    rejected(lambda value: value.update(selected_attempt=None))
    rejected(
        lambda value: (
            value.update(selected_attempt=None),
            value.update(dispositions={"training": "rejected"}),
        )
    )
    rejected(
        lambda value: value["parent"]["score"].update(
            task_ids=[f"other-{task_id}" for task_id in _manifest().task_ids]
        )
    )
    rejected(
        lambda value: value["selected"]["score"].update(
            task_ids=[f"other-{task_id}" for task_id in _manifest().task_ids]
        )
    )
    rejected(lambda value: value.update(sequence_before=2))
    rejected(lambda value: value.update(state_before_sha256="0" * 64))
    rejected(lambda value: value.update(parent=_candidate("other-parent").to_dict()))
    rejected(lambda value: value["state_after"].update(sequence=2))
    rejected(lambda value: value["state_after"].update(next_stage="dale"))
    rejected(
        lambda value: value["state_after"].update(accepted=before.accepted.to_dict())
    )


def test_pending_recovery_rejects_missing_identity_sequence_and_progress(
    tmp_path: Path,
) -> None:
    before, after, document = _pending_fixture()
    identity_store = PipelineStore(tmp_path / "identity")
    identity_store.write_pending({"state_after": {}})
    with pytest.raises(ProgressConflictError, match="identity"):
        evolve._recover_pending_transition(_Adapter(), identity_store, before)

    sequence_store = PipelineStore(tmp_path / "sequence")
    sequence_store.write_pending(document)
    too_far = replace(after, sequence=2, cursor=256)
    with pytest.raises(ProgressConflictError, match="sequence differs"):
        evolve._recover_pending_transition(_Adapter(), sequence_store, too_far)

    progress_store = PipelineStore(tmp_path / "progress")
    progress_store.write_pending(document)
    with pytest.raises(ProgressConflictError, match="lacks matching"):
        evolve._recover_pending_transition(_Adapter(), progress_store, after)

    with pytest.raises(ProgressConflictError, match="escapes"):
        evolve._discard_recovered_attempts(
            _Adapter(),
            PipelineStore(tmp_path / "escaped"),
            (CandidateAttempt.completed("bad", _candidate("outside")),),
        )


def test_committed_pending_recovery_is_bound_to_progress_cleanup_evidence(
    tmp_path: Path,
) -> None:
    before, after, document = _pending_fixture()
    attempt = CandidateAttempt.completed("training", after.accepted)
    record = PipelineStore(tmp_path).progress_record(
        before,
        after,
        stage_id="r000-train",
        stage="train",
        parent=before.accepted,
        selected=after.accepted,
        attempts=(attempt,),
        dispositions={"training": "accepted"},
        elapsed_seconds=1.0,
    )
    forged = replace(
        after.accepted,
        checkpoint_path=str(tmp_path / ".candidates" / "forged-cleanup.npz"),
        checkpoint_sha256=_identity("checkpoint", "forged-cleanup"),
    ).to_dict()
    source_drift = json.loads(json.dumps(document))
    source_drift["selected"] = forged
    source_drift["attempts"][0]["candidate"] = forged
    source_drift["state_after"]["accepted"] = forged
    elapsed_drift = json.loads(json.dumps(document))
    elapsed_drift["elapsed_seconds"] = 2.0

    for corrupted in (source_drift, elapsed_drift):
        with pytest.raises(ProgressConflictError, match="inconsistent"):
            evolve._pending_transition_parts(
                after,
                corrupted,
                committed_record=record,
            )


def test_two_no_progress_rounds_stop_stable_without_edge_revisit(
    tmp_path: Path,
) -> None:
    class StableAdapter(_Adapter):
        def train_parent(self, parent, schedule, context):
            return _candidate(
                f"unchanged-{context.round_index}",
                exact_count=parent.score.exact_count,
                loss=parent.score.unresolved_loss,
                topology_changed=False,
            )

        def run_candidate(self, parent, arm, schedule, context):
            self.calls.append(("candidate", context.stage, arm, parent.candidate_id))
            if arm in {"add", "excitatory"}:
                return CandidateAttempt.blocked(arm, "bounded")
            return CandidateAttempt.failed(arm, "no improving child")

    adapter = StableAdapter()
    state = run_evolution(
        adapter,
        tmp_path,
        config=PipelineConfig(rounds=3, patience=2),
        history_plotter=_history_plotter,
    )
    assert state.closed and state.terminal_reason == "stable"
    assert state.stable_rounds == 2
    assert state.cursor == 2 * 4 * DEFAULT_UPDATES
    candidate_stages = [call[1] for call in adapter.calls if call[0] == "candidate"]
    assert "edge-revisit" not in candidate_stages
    assert candidate_stages.count("edge") == 4
    assert adapter.evaluation_calls == 1


def test_training_mastery_runs_compression_first_then_closes(
    tmp_path: Path,
) -> None:
    class MasteryAdapter(_Adapter):
        def initialize(self, config, output_dir):
            return _candidate("almost", exact_count=399, topology_changed=False)

        def train_parent(self, parent, schedule, context):
            return _candidate("mastered", exact_count=400, topology_changed=False)

        def run_candidate(self, parent, arm, schedule, context):
            self.calls.append(("candidate", context.stage, arm, parent.candidate_id))
            if context.stage == "compression-edge":
                return CandidateAttempt.completed(
                    arm,
                    _candidate(
                        "compressed-edge",
                        exact_count=400,
                        persistent_bytes=900,
                        edges=180,
                    ),
                )
            if context.stage == "compression-neuron":
                return CandidateAttempt.completed(
                    arm,
                    _candidate(
                        "compressed-neuron",
                        exact_count=400,
                        persistent_bytes=800,
                        neurons=90,
                        edges=180,
                    ),
                )
            raise AssertionError(context.stage)

    adapter = MasteryAdapter()
    state = run_evolution(
        adapter,
        tmp_path,
        config=PipelineConfig(rounds=1),
        history_plotter=_history_plotter,
    )
    assert state.closed and state.terminal_reason == "mastery"
    assert state.accepted.candidate_id == "compressed-neuron"
    assert state.cursor == 3 * DEFAULT_UPDATES
    assert [call[1] for call in adapter.calls if call[0] == "candidate"] == [
        "compression-edge",
        "compression-neuron",
    ]


def test_structural_mastery_switches_immediately_to_compression(
    tmp_path: Path,
) -> None:
    class StructuralMasteryAdapter(_Adapter):
        def initialize(self, config, output_dir):
            return _candidate("almost", exact_count=399, topology_changed=False)

        def train_parent(self, parent, schedule, context):
            return _candidate(
                "trained-almost",
                exact_count=399,
                loss=9.0,
                topology_changed=False,
            )

        def run_candidate(self, parent, arm, schedule, context):
            self.calls.append(("candidate", context.stage, arm, parent.candidate_id))
            if context.stage == "edge":
                if arm == "add":
                    return CandidateAttempt.completed(
                        arm, _candidate("edge-mastered", exact_count=400)
                    )
                return CandidateAttempt.failed(arm, "no improving child")
            if context.stage == "compression-edge":
                return CandidateAttempt.completed(
                    arm,
                    _candidate(
                        "compressed-edge",
                        exact_count=400,
                        persistent_bytes=900,
                        edges=180,
                    ),
                )
            if context.stage == "compression-neuron":
                return CandidateAttempt.completed(
                    arm,
                    _candidate(
                        "compressed-neuron",
                        exact_count=400,
                        persistent_bytes=800,
                        neurons=90,
                        edges=180,
                    ),
                )
            raise AssertionError(f"mastery must skip {context.stage}")

    adapter = StructuralMasteryAdapter()
    state = run_evolution(
        adapter,
        tmp_path,
        config=PipelineConfig(rounds=1),
        history_plotter=_history_plotter,
    )
    assert state.closed and state.terminal_reason == "mastery"
    assert state.accepted.candidate_id == "compressed-neuron"
    assert [call[1] for call in adapter.calls if call[0] == "candidate"] == [
        "edge",
        "edge",
        "compression-edge",
        "compression-neuron",
    ]


def test_mastery_compression_repeats_until_resource_stability(
    tmp_path: Path,
) -> None:
    class IterativeCompressionAdapter(_Adapter):
        def initialize(self, config, output_dir):
            return _candidate("almost", exact_count=399, topology_changed=False)

        def train_parent(self, parent, schedule, context):
            return _candidate(
                "mastered",
                exact_count=400,
                persistent_bytes=1_000,
                topology_changed=False,
            )

        def run_candidate(self, parent, arm, schedule, context):
            self.calls.append(("candidate", context.stage, arm, context.round_index))
            target_bytes = {
                (0, "compression-edge"): 900,
                (0, "compression-neuron"): 800,
                (1, "compression-edge"): 700,
                (1, "compression-neuron"): 600,
            }.get((context.round_index, context.stage))
            if target_bytes is None:
                return CandidateAttempt.blocked(arm, "no smaller exact topology")
            return CandidateAttempt.completed(
                arm,
                _candidate(
                    f"compressed-{context.round_index}-{context.stage}",
                    exact_count=400,
                    persistent_bytes=target_bytes,
                    neurons=max(1, parent.resources.neurons - 1),
                    edges=max(0, parent.resources.recurrent_edges - 1),
                ),
            )

    adapter = IterativeCompressionAdapter()
    state = run_evolution(
        adapter,
        tmp_path,
        config=PipelineConfig(rounds=4, patience=1),
        history_plotter=_history_plotter,
    )

    assert state.closed and state.terminal_reason == "mastery"
    assert state.round_index == 2
    assert state.accepted.resources.persistent_bytes == 600
    assert state.cursor == 7 * DEFAULT_UPDATES
    assert [(call[1], call[3]) for call in adapter.calls if call[0] == "candidate"] == [
        ("compression-edge", 0),
        ("compression-neuron", 0),
        ("compression-edge", 1),
        ("compression-neuron", 1),
        ("compression-edge", 2),
        ("compression-neuron", 2),
    ]


def test_selected_edge_child_is_next_neuron_parent_with_topology_parameters_and_muon(
    tmp_path: Path,
) -> None:
    adapter = _Adapter()
    state = run_evolution(
        adapter,
        tmp_path,
        config=PipelineConfig(rounds=1),
        history_plotter=_history_plotter,
    )
    assert state.closed
    assert state.terminal_reason == "round-budget"
    assert adapter.evaluation_calls == 1

    neuron_call = next(
        call for call in adapter.calls if call[:3] == ("candidate", "neuron", "add")
    )
    assert neuron_call[3:7] == (
        "edge-add",
        _identity("topology", "edge-add"),
        _identity("parameters", "edge-add"),
        _identity("muon", "edge-add"),
    )
    revisit_call = next(
        call
        for call in adapter.calls
        if call[:3] == ("candidate", "edge-revisit", "add")
    )
    assert revisit_call[3] == "neuron-add"
    dale_call = next(
        call
        for call in adapter.calls
        if call[:3] == ("candidate", "dale", "excitatory")
    )
    assert dale_call[3] == "edge-revisit-add"

    sibling_calls = [
        call for call in adapter.calls if call[0] == "candidate" and call[1] == "edge"
    ]
    assert [call[2] for call in sibling_calls] == ["add", "prune"]
    assert sibling_calls[0][-1] == sibling_calls[1][-1]
    assert (tmp_path / "run-state.json").is_file()
    assert (tmp_path / "progress.jsonl").is_file()
    assert (tmp_path / "topology.png").read_bytes() == _identity(
        "topology", "dale-excitatory"
    ).encode()
    assert (tmp_path / "score-history.png").read_bytes() == b"history"
    assert (tmp_path / "evaluation.json").is_file()
    progress = [
        json.loads(line)
        for line in (tmp_path / "progress.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    edge_record = next(record for record in progress if record["stage"] == "edge")
    assert {
        "parent_checkpoint_sha256",
        "child_checkpoint_sha256",
        "siblings",
        "exact_task_count",
        "solved_task_ids",
        "unresolved_task_loss",
        "updates",
        "neurons",
        "recurrent_edges",
        "persistent_bytes",
        "checkpoint_bytes",
        "elapsed_seconds",
        "peak_host_ram_bytes",
        "device_memory_bytes",
        "state_after",
    } <= edge_record.keys()


def test_progress_reports_executed_updates_per_arm_and_cursor_separately(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(rounds=1)
    manifest = _manifest()
    parent = _candidate("edge-parent", topology_changed=False)
    before = replace(
        RunState.initial(config, manifest, parent),
        sequence=1,
        cursor=DEFAULT_UPDATES,
        next_stage="edge",
    )
    selected = _candidate("edge-child", exact_count=1)
    after = replace(
        before,
        sequence=2,
        cursor=2 * DEFAULT_UPDATES,
        next_stage="neuron",
        accepted=selected,
    )
    attempts = (
        CandidateAttempt.completed("add", selected),
        CandidateAttempt.blocked("prune", "at minimum topology"),
    )
    record = PipelineStore(tmp_path).progress_record(
        before,
        after,
        stage_id="r000-edge",
        stage="edge",
        parent=parent,
        selected=selected,
        attempts=attempts,
        dispositions={"add": "accepted", "prune": "blocked"},
        elapsed_seconds=1.0,
    )
    assert record["updates"] == DEFAULT_UPDATES
    assert record["total_executed_updates"] == DEFAULT_UPDATES
    assert record["cursor_advance"] == DEFAULT_UPDATES
    siblings = record["siblings"]
    assert siblings[0]["status"] == "completed"
    assert siblings[0]["disposition"] == "accepted"
    assert siblings[0]["executed_updates"] == DEFAULT_UPDATES
    assert siblings[1]["status"] == "blocked"
    assert siblings[1]["disposition"] == "blocked"
    assert siblings[1]["executed_updates"] == 0


def test_progress_replay_rejects_corrupt_sibling_and_update_evidence(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(rounds=1)
    manifest = _manifest()
    parent = _candidate("edge-parent", topology_changed=False)
    before = replace(
        RunState.initial(config, manifest, parent),
        sequence=1,
        cursor=DEFAULT_UPDATES,
        next_stage="edge",
    )
    selected = _candidate("edge-child", exact_count=1)
    rejected = _candidate("edge-rejected", persistent_bytes=1_100)
    after = replace(
        before,
        sequence=2,
        cursor=2 * DEFAULT_UPDATES,
        next_stage="neuron",
        accepted=selected,
    )
    attempts = (
        CandidateAttempt.completed("add", selected),
        CandidateAttempt.completed("prune", rejected),
    )
    record = PipelineStore(tmp_path).progress_record(
        before,
        after,
        stage_id="r000-edge",
        stage="edge",
        parent=parent,
        selected=selected,
        attempts=attempts,
        dispositions={"add": "accepted", "prune": "rejected-no-improvement"},
        elapsed_seconds=1.0,
    )
    assert record["updates"] == 2 * DEFAULT_UPDATES
    assert record["total_executed_updates"] == 2 * DEFAULT_UPDATES

    def rejected_record(mutator) -> None:
        corrupted = json.loads(json.dumps(record))
        mutator(corrupted)
        with pytest.raises(ProgressConflictError):
            evolve._validate_progress_transition(before, after, corrupted)

    rejected_record(lambda value: value["siblings"].append(dict(value["siblings"][0])))
    rejected_record(lambda value: value["siblings"].reverse())
    rejected_record(lambda value: value["siblings"][0].update(status="other"))
    rejected_record(lambda value: value["siblings"][0].update(reason="fabricated"))
    rejected_record(lambda value: value["siblings"][0].update(checkpoint_sha256="bad"))
    rejected_record(lambda value: value["siblings"][0].update(executed_updates=0))
    rejected_record(lambda value: value["siblings"][1].update(disposition="accepted"))
    rejected_record(lambda value: value["siblings"][1].update(persistent_bytes=-1))
    rejected_record(
        lambda value: value["siblings"][1]["candidate"]["resources"].update(
            persistent_bytes=1_099
        )
    )
    rejected_record(lambda value: value.update(peak_host_ram_bytes=0))
    rejected_record(lambda value: value.update(total_executed_updates=0))
    rejected_record(lambda value: value.update(stage_id="r999-edge"))
    rejected_record(lambda value: value.update(disposition="retained-parent"))
    rejected_record(lambda value: value.update(sequence_before=before.sequence + 1))
    rejected_record(lambda value: value.update(sequence_after=after.sequence + 1))
    rejected_record(lambda value: value.update(round=before.round_index + 1))
    rejected_record(
        lambda value: value["state_before"]["accepted"].update(
            candidate_id="forged-parent"
        )
    )
    rejected_record(lambda value: value["state_before"].update(next_stage="neuron"))
    rejected_record(lambda value: value.update(parent_checkpoint_sha256="bad"))
    rejected_record(lambda value: value.update(child_checkpoint_sha256="bad"))
    rejected_record(lambda value: value.update(selected_candidate_id="other"))
    rejected_record(lambda value: value.update(cursor_advance=0))
    rejected_record(lambda value: value.update(solved_task_ids=[]))
    rejected_record(lambda value: value.update(exact_task_count=0))
    rejected_record(lambda value: value.update(unresolved_task_loss=2.0))
    rejected_record(lambda value: value.update(device_memory_bytes=0))
    rejected_record(lambda value: value.update(updates=0))
    rejected_record(lambda value: value.update(elapsed_seconds=-1.0))


def test_progress_replay_rejects_noncanonical_boundary_transitions(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(rounds=1)
    manifest = _manifest()
    accepted = _candidate("boundary", topology_changed=False)
    initial = RunState.initial(config, manifest, accepted)
    round_before = replace(initial, sequence=1, next_stage="round-end")
    round_after = evolve._expected_round_successor(round_before)
    store = PipelineStore(tmp_path)
    round_record = store.progress_record(
        round_before,
        round_after,
        stage_id="r000-round-end",
        stage="round-end",
        parent=round_before.round_entry,
        selected=accepted,
        attempts=(),
        elapsed_seconds=1.0,
    )
    evolve._validate_progress_transition(round_before, round_after, round_record)

    bad_round_identity = json.loads(json.dumps(round_record))
    bad_round_identity["stage_id"] = "r999-round-end"
    with pytest.raises(ProgressConflictError, match="Round-end"):
        evolve._validate_progress_transition(
            round_before, round_after, bad_round_identity
        )
    with pytest.raises(ProgressConflictError, match="successor"):
        evolve._validate_progress_transition(
            round_before,
            replace(round_after, sequence=round_after.sequence + 1),
            round_record,
        )

    terminal_after = replace(
        round_after,
        sequence=round_after.sequence + 1,
        next_stage="closed",
        closed=True,
        evaluation_completed=True,
        evaluation_digest=_identity("evaluation", "boundary"),
    )
    terminal_record = store.progress_record(
        round_after,
        terminal_after,
        stage_id="terminal-evaluation",
        stage="terminal",
        parent=accepted,
        selected=accepted,
        attempts=(),
        elapsed_seconds=1.0,
    )
    evolve._validate_progress_transition(round_after, terminal_after, terminal_record)

    bad_terminal_identity = json.loads(json.dumps(terminal_record))
    bad_terminal_identity["stage_id"] = "terminal-other"
    with pytest.raises(ProgressConflictError, match="Terminal"):
        evolve._validate_progress_transition(
            round_after, terminal_after, bad_terminal_identity
        )
    with pytest.raises(ProgressConflictError, match="Closed state"):
        evolve._validate_progress_transition(
            terminal_after, terminal_after, terminal_record
        )


def test_interrupted_run_resumes_without_evaluation_or_repeating_durable_stage(
    tmp_path: Path,
) -> None:
    interrupted = _Adapter(interrupt_at="edge")
    with pytest.raises(KeyboardInterrupt):
        run_evolution(
            interrupted,
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    assert interrupted.evaluation_calls == 0
    saved = RunState.from_dict(
        json.loads((tmp_path / "run-state.json").read_text(encoding="utf-8"))
    )
    assert saved.next_stage == "edge"
    assert saved.accepted.candidate_id == "trained-0"

    resumed = _Adapter()
    state = run_evolution(
        resumed,
        tmp_path,
        config=PipelineConfig(rounds=1),
        history_plotter=_history_plotter,
    )
    assert state.closed
    assert ("restore", "trained-0") in resumed.calls
    assert not any(call[0] == "train-parent" for call in resumed.calls)
    assert resumed.evaluation_calls == 1


def test_crash_after_selected_persist_resumes_pending_without_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_append = PipelineStore.append_progress
    interrupted_once = False

    def interrupt_after_persist(self, record):
        nonlocal interrupted_once
        if record["stage"] == "train" and not interrupted_once:
            interrupted_once = True
            raise KeyboardInterrupt("after selected checkpoint persistence")
        return original_append(self, record)

    monkeypatch.setattr(PipelineStore, "append_progress", interrupt_after_persist)
    interrupted = _Adapter()
    with pytest.raises(KeyboardInterrupt, match="after selected"):
        run_evolution(
            interrupted,
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    trained_checkpoint = tmp_path / "checkpoints" / "r000-train.npz"
    checkpoint_bytes = trained_checkpoint.read_bytes()

    monkeypatch.setattr(PipelineStore, "append_progress", original_append)
    resumed = _Adapter(interrupt_at="edge")
    with pytest.raises(KeyboardInterrupt, match="edge"):
        run_evolution(
            resumed,
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    assert not any(call[0] == "train-parent" for call in resumed.calls)
    assert trained_checkpoint.read_bytes() == checkpoint_bytes
    assert not (tmp_path / "pending-transition.json").exists()


def test_fresh_resume_attests_pending_ancestry_before_first_persist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AncestryAdapter(_Adapter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.parents = {}

        def initialize(self, config, output_dir):
            candidate = super().initialize(config, output_dir)
            self.parents[candidate.checkpoint_sha256] = None
            return candidate

        def train_parent(self, parent, schedule, context):
            candidate = super().train_parent(parent, schedule, context)
            self.parents[candidate.checkpoint_sha256] = parent.checkpoint_sha256
            return candidate

        def attest_pending(self, candidate, *, parent_checkpoint_sha256, stage_id):
            path = Path(candidate.checkpoint_path)
            assert path.parent.name == ".candidates"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == (
                candidate.checkpoint_sha256
            )
            self.parents[candidate.checkpoint_sha256] = parent_checkpoint_sha256
            self.calls.append(("attest", candidate.candidate_id, stage_id))
            return candidate

        def persist(self, candidate, destination, **kwargs):
            expected = self.parents.get(candidate.checkpoint_sha256, object())
            if expected != kwargs.get("parent_checkpoint_sha256"):
                raise ValueError("candidate ancestry is not registered")
            persisted = super().persist(candidate, destination, **kwargs)
            self.parents.pop(candidate.checkpoint_sha256, None)
            return persisted

    original_write_pending = PipelineStore.write_pending
    interrupted_once = False

    def interrupt_before_persist(self, document):
        nonlocal interrupted_once
        original_write_pending(self, document)
        if document["stage"] == "train" and not interrupted_once:
            interrupted_once = True
            raise KeyboardInterrupt("after pending journal")

    monkeypatch.setattr(PipelineStore, "write_pending", interrupt_before_persist)
    with pytest.raises(KeyboardInterrupt, match="pending journal"):
        run_evolution(
            AncestryAdapter(),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    assert not (tmp_path / "checkpoints" / "r000-train.npz").exists()

    monkeypatch.setattr(PipelineStore, "write_pending", original_write_pending)
    resumed = AncestryAdapter(interrupt_at="edge")
    with pytest.raises(KeyboardInterrupt, match="edge"):
        run_evolution(
            resumed,
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    assert not any(call[0] == "train-parent" for call in resumed.calls)
    assert any(call[0] == "attest" for call in resumed.calls)
    assert (tmp_path / "checkpoints" / "r000-train.npz").is_file()


def test_crash_after_progress_append_reconciles_pending_without_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_write = PipelineStore.write_state

    def interrupt_after_progress(self, state):
        if state.sequence == 1:
            raise KeyboardInterrupt("after progress append")
        return original_write(self, state)

    monkeypatch.setattr(PipelineStore, "write_state", interrupt_after_progress)
    with pytest.raises(KeyboardInterrupt, match="after progress"):
        run_evolution(
            _Adapter(),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    assert (tmp_path / "pending-transition.json").is_file()
    assert len((tmp_path / "progress.jsonl").read_text().splitlines()) == 1

    monkeypatch.setattr(PipelineStore, "write_state", original_write)
    resumed = _Adapter(interrupt_at="edge")
    with pytest.raises(KeyboardInterrupt, match="edge"):
        run_evolution(
            resumed,
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    assert not any(call[0] == "train-parent" for call in resumed.calls)
    assert not (tmp_path / "pending-transition.json").exists()
    assert ("discard", "training", "completed") not in resumed.calls


def test_destination_recovery_cleans_selected_source_after_durable_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LeavesSelectedSourceAdapter(_Adapter):
        def persist(self, candidate, destination, **kwargs):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(candidate.candidate_id.encode())
            persisted = replace(
                candidate,
                checkpoint_path=str(destination),
                checkpoint_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                resources=replace(
                    candidate.resources, checkpoint_bytes=destination.stat().st_size
                ),
            )
            self.calls.append(
                (
                    "persist",
                    candidate.candidate_id,
                    kwargs.get("parent_checkpoint_sha256"),
                    kwargs.get("stage_id"),
                    persisted.checkpoint_sha256,
                )
            )
            return persisted

    original_append = PipelineStore.append_progress

    def interrupt_train_commit(self, record):
        if record["stage"] == "train":
            raise KeyboardInterrupt("after train destination")
        return original_append(self, record)

    monkeypatch.setattr(PipelineStore, "append_progress", interrupt_train_commit)
    with pytest.raises(KeyboardInterrupt, match="train destination"):
        run_evolution(
            LeavesSelectedSourceAdapter(),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    selected_source = tmp_path / ".candidates" / "trained-0.npz"
    assert selected_source.is_file()
    assert (tmp_path / "checkpoints" / "r000-train.npz").is_file()
    assert (tmp_path / "checkpoints" / "r000-train.lineage.json").is_file()

    monkeypatch.setattr(PipelineStore, "append_progress", original_append)
    with pytest.raises(KeyboardInterrupt, match="edge"):
        run_evolution(
            LeavesSelectedSourceAdapter(interrupt_at="edge"),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    assert not selected_source.exists()


def test_destination_recovery_rejects_tampered_immediate_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_append = PipelineStore.append_progress

    def interrupt_train_commit(self, record):
        if record["stage"] == "train":
            raise KeyboardInterrupt("after train destination")
        return original_append(self, record)

    monkeypatch.setattr(PipelineStore, "append_progress", interrupt_train_commit)
    with pytest.raises(KeyboardInterrupt, match="train destination"):
        run_evolution(
            _Adapter(),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    lineage_path = tmp_path / "checkpoints" / "r000-train.lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    lineage["parent_checkpoint_sha256"] = "0" * 64
    lineage_path.write_text(json.dumps(lineage), encoding="utf-8")

    monkeypatch.setattr(PipelineStore, "append_progress", original_append)
    with pytest.raises(ProgressConflictError, match="lineage"):
        run_evolution(
            _Adapter(),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )


def test_progress_replay_rejects_selected_source_lineage_drift(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(rounds=1)
    with pytest.raises(KeyboardInterrupt, match="edge"):
        run_evolution(
            _Adapter(interrupt_at="edge"),
            tmp_path,
            config=config,
            history_plotter=_history_plotter,
        )
    progress_path = tmp_path / "progress.jsonl"
    records = [json.loads(line) for line in progress_path.read_text().splitlines()]
    train_record = records[0]
    selected_sibling = train_record["siblings"][0]
    forged_sha256 = _identity("checkpoint", "forged-replay-source")
    selected_sibling["candidate"]["checkpoint_path"] = str(
        tmp_path / ".candidates" / "forged-replay-source.npz"
    )
    selected_sibling["candidate"]["checkpoint_sha256"] = forged_sha256
    selected_sibling["checkpoint_sha256"] = forged_sha256
    progress_path.write_text(
        "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(ProgressConflictError, match="lineage"):
        PipelineStore(tmp_path).load_state(config, _manifest())


def test_fresh_resume_removes_recovered_rejected_candidate_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_append = PipelineStore.append_progress

    def interrupt_edge_commit(self, record):
        if record["stage"] == "edge":
            raise KeyboardInterrupt("after edge persistence")
        return original_append(self, record)

    monkeypatch.setattr(PipelineStore, "append_progress", interrupt_edge_commit)
    with pytest.raises(KeyboardInterrupt, match="edge persistence"):
        run_evolution(
            _Adapter(),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    candidate_root = tmp_path / ".candidates"
    edge_paths = [
        candidate_root / "edge-add.npz",
        candidate_root / "edge-prune.npz",
    ]
    assert not edge_paths[0].exists()
    assert edge_paths[1].is_file()

    monkeypatch.setattr(PipelineStore, "append_progress", original_append)
    with pytest.raises(KeyboardInterrupt, match="neuron"):
        run_evolution(
            _Adapter(interrupt_at="neuron"),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    assert all(not path.exists() for path in edge_paths)


def test_selected_source_removed_by_persist_is_not_discarded(
    tmp_path: Path,
) -> None:
    class PersistOwnsSelectedSourceAdapter(_Adapter):
        def persist(self, candidate, destination, **kwargs):
            persisted = super().persist(candidate, destination, **kwargs)
            source = Path(candidate.checkpoint_path)
            if source.parent.name == ".candidates":
                source.unlink(missing_ok=True)
            return persisted

        def discard(self, attempt):
            if (
                attempt.candidate is not None
                and not Path(attempt.candidate.checkpoint_path).exists()
            ):
                raise AssertionError("selected source must not be discarded")
            return super().discard(attempt)

    adapter = PersistOwnsSelectedSourceAdapter(interrupt_at="edge")
    with pytest.raises(KeyboardInterrupt, match="edge"):
        run_evolution(
            adapter,
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    assert not any(call[:2] == ("discard", "training") for call in adapter.calls)


def test_persist_rejects_adapter_checkpoint_digest_not_matching_bytes(
    tmp_path: Path,
) -> None:
    class LyingDigestAdapter(_Adapter):
        def persist(self, candidate, destination, **kwargs):
            persisted = super().persist(candidate, destination, **kwargs)
            return replace(persisted, checkpoint_sha256="0" * 64)

    with pytest.raises(PipelineError, match="digest"):
        run_evolution(
            LyingDigestAdapter(),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )


def test_checkpoint_handoff_rejects_component_path_file_and_restore_drift(
    tmp_path: Path,
) -> None:
    candidate = _candidate("handoff")
    config = PipelineConfig(rounds=1)

    class ComponentDriftAdapter(_Adapter):
        def persist(self, candidate, destination, **kwargs):
            persisted = super().persist(candidate, destination, **kwargs)
            return replace(
                persisted,
                topology_sha256=_identity("topology", "changed"),
            )

    with pytest.raises(PipelineError, match="topology"):
        evolve._persist_selected(
            ComponentDriftAdapter(),
            PipelineStore(tmp_path / "component"),
            candidate,
            stage_id="initial",
            parent_checkpoint_sha256=None,
            config=config,
        )

    class WrongPathAdapter(_Adapter):
        def persist(self, candidate, destination, **kwargs):
            persisted = super().persist(candidate, destination, **kwargs)
            return replace(
                persisted, checkpoint_path=str(destination.with_suffix(".bad"))
            )

    with pytest.raises(PipelineError, match="path differs"):
        evolve._persist_selected(
            WrongPathAdapter(),
            PipelineStore(tmp_path / "path"),
            candidate,
            stage_id="initial",
            parent_checkpoint_sha256=None,
            config=config,
        )

    class MissingFileAdapter(_Adapter):
        def persist(self, candidate, destination, **kwargs):
            return replace(candidate, checkpoint_path=str(destination))

    with pytest.raises(PipelineError, match="missing"):
        evolve._persist_selected(
            MissingFileAdapter(),
            PipelineStore(tmp_path / "missing"),
            candidate,
            stage_id="initial",
            parent_checkpoint_sha256=None,
            config=config,
        )

    directory_store = PipelineStore(tmp_path / "directory")
    destination = directory_store.checkpoint_dir / "r000-edge.npz"
    destination.mkdir(parents=True)
    with pytest.raises(PipelineError, match="not a file"):
        evolve._persist_or_restore_selected(
            _Adapter(),
            directory_store,
            candidate,
            stage_id="r000-edge",
            parent_checkpoint_sha256=_identity("checkpoint", "parent"),
            config=config,
        )

    expected = replace(
        candidate,
        checkpoint_path=str(tmp_path / "expected.npz"),
    )
    restored = replace(
        expected,
        parameters_sha256=_identity("parameters", "changed"),
    )
    with pytest.raises(ResumeMismatchError, match="continuation evidence"):
        evolve._verify_restored(expected, restored)


def test_candidate_and_plot_artifact_guards_fail_closed(tmp_path: Path) -> None:
    config = PipelineConfig(rounds=1, max_neurons=100)
    with pytest.raises(PipelineError, match="non-finite"):
        evolve._require_candidate(_candidate("nan", finite=False), config)
    with pytest.raises(PipelineError, match="neuron-cap"):
        evolve._require_candidate(_candidate("large", neurons=101), config)
    with pytest.raises(PipelineError, match="training manifest"):
        evolve._require_candidate(
            _candidate("mismatch"),
            config,
            tuple(f"other-{task_id}" for task_id in _manifest().task_ids),
        )

    class MissingTopologyAdapter(_Adapter):
        def render_topology(self, candidate, output_path):
            return None

    store = PipelineStore(tmp_path / "topology")
    with pytest.raises(PipelineError, match="Topology renderer"):
        evolve._refresh_artifacts(
            MissingTopologyAdapter(), store, _candidate("plot"), _history_plotter
        )
    store = PipelineStore(tmp_path / "history")
    with pytest.raises(PipelineError, match="History renderer"):
        evolve._refresh_artifacts(
            _Adapter(), store, _candidate("plot"), lambda records, path: None
        )


@pytest.mark.parametrize("failure", ("regression", "nonfinite"))
def test_parent_training_regression_retains_durable_parent_then_discards_temp(
    tmp_path: Path,
    failure: str,
) -> None:
    class RegressingAdapter(_Adapter):
        def initialize(self, config, output_dir):
            return _candidate("protected-parent", exact_count=1, topology_changed=False)

        def train_parent(self, parent, schedule, context):
            if failure == "nonfinite":
                return _candidate(
                    "nonfinite-training",
                    exact_count=1,
                    finite=False,
                    topology_changed=False,
                )
            trained = _candidate(
                "regressing-training",
                exact_count=0,
                loss=1.0,
                topology_changed=False,
            )
            return trained

    adapter = RegressingAdapter(interrupt_at="edge")
    with pytest.raises(KeyboardInterrupt):
        run_evolution(
            adapter,
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    state = RunState.from_dict(
        json.loads((tmp_path / "run-state.json").read_text(encoding="utf-8"))
    )
    assert state.accepted.candidate_id == "protected-parent"
    assert state.next_stage == "edge"
    assert ("discard", "training", "completed") in adapter.calls


def test_resume_rejects_config_drift_and_closed_run(tmp_path: Path) -> None:
    adapter = _Adapter(interrupt_at="edge")
    with pytest.raises(KeyboardInterrupt):
        run_evolution(
            adapter,
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    with pytest.raises(ResumeMismatchError, match="configuration"):
        run_evolution(
            _Adapter(),
            tmp_path,
            config=PipelineConfig(rounds=2),
            history_plotter=_history_plotter,
        )
    state = run_evolution(
        _Adapter(),
        tmp_path,
        config=PipelineConfig(rounds=1),
        history_plotter=_history_plotter,
    )
    assert state.closed
    closed_retry = _Adapter()
    assert (
        run_evolution(
            closed_retry,
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
        == state
    )
    assert closed_retry.evaluation_calls == 0


def test_closed_resume_rehashes_terminal_evaluation_artifact(tmp_path: Path) -> None:
    run_evolution(
        _Adapter(),
        tmp_path,
        config=PipelineConfig(rounds=1),
        history_plotter=_history_plotter,
    )
    (tmp_path / "evaluation.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ProgressConflictError, match="evaluation digest"):
        run_evolution(
            _Adapter(),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )


def test_terminal_crash_after_evaluation_uses_durable_intent_on_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_atomic = evolve._atomic_json
    interrupted = False

    def interrupt_before_result(path, value):
        nonlocal interrupted
        if Path(path).name == "evaluation.json" and not interrupted:
            interrupted = True
            assert (tmp_path / "evaluation-intent.json").is_file()
            raise KeyboardInterrupt("after evaluation call")
        return original_atomic(path, value)

    monkeypatch.setattr(evolve, "_atomic_json", interrupt_before_result)
    first = _Adapter()
    with pytest.raises(KeyboardInterrupt, match="evaluation call"):
        run_evolution(
            first,
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    assert first.evaluation_calls == 1
    assert not (tmp_path / "evaluation.json").exists()

    monkeypatch.setattr(evolve, "_atomic_json", original_atomic)
    resumed = _Adapter()
    state = run_evolution(
        resumed,
        tmp_path,
        config=PipelineConfig(rounds=1),
        history_plotter=_history_plotter,
    )
    assert state.closed
    assert resumed.evaluation_calls == 1
    assert not (tmp_path / "evaluation-intent.json").exists()


def test_terminal_durable_result_finalizes_without_rescoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_progress_record = PipelineStore.progress_record
    interrupted = False

    def interrupt_before_terminal_progress(self, *args, **kwargs):
        nonlocal interrupted
        if kwargs.get("stage") == "terminal" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("after durable evaluation result")
        return original_progress_record(self, *args, **kwargs)

    monkeypatch.setattr(
        PipelineStore, "progress_record", interrupt_before_terminal_progress
    )
    first = _Adapter()
    with pytest.raises(KeyboardInterrupt, match="durable evaluation result"):
        run_evolution(
            first,
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    assert first.evaluation_calls == 1
    assert (tmp_path / "evaluation.json").is_file()
    assert (tmp_path / "evaluation-intent.json").is_file()

    monkeypatch.setattr(PipelineStore, "progress_record", original_progress_record)
    resumed = _Adapter()
    state = run_evolution(
        resumed,
        tmp_path,
        config=PipelineConfig(rounds=1),
        history_plotter=_history_plotter,
    )
    assert state.closed
    assert resumed.evaluation_calls == 0
    assert not (tmp_path / "evaluation-intent.json").exists()


def test_closed_retry_repairs_missing_plots_without_rescoring(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_refresh = evolve._refresh_artifacts
    interrupted = False

    def interrupt_after_terminal_commit(adapter, store, candidate, plotter):
        nonlocal interrupted
        if (
            not interrupted
            and store.state_path.is_file()
            and json.loads(store.state_path.read_text(encoding="utf-8"))["closed"]
        ):
            interrupted = True
            (store.output_dir / "topology.png").unlink(missing_ok=True)
            (store.output_dir / "score-history.png").unlink(missing_ok=True)
            raise KeyboardInterrupt("after terminal commit")
        return original_refresh(adapter, store, candidate, plotter)

    monkeypatch.setattr(evolve, "_refresh_artifacts", interrupt_after_terminal_commit)
    first = _Adapter()
    with pytest.raises(KeyboardInterrupt, match="terminal commit"):
        run_evolution(
            first,
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    assert first.evaluation_calls == 1
    assert not (tmp_path / "topology.png").exists()
    assert not (tmp_path / "score-history.png").exists()

    monkeypatch.setattr(evolve, "_refresh_artifacts", original_refresh)
    resumed = _Adapter()
    state = run_evolution(
        resumed,
        tmp_path,
        config=PipelineConfig(rounds=1),
        history_plotter=_history_plotter,
    )
    assert state.closed
    assert resumed.evaluation_calls == 0
    assert (tmp_path / "topology.png").is_file()
    assert (tmp_path / "score-history.png").is_file()


def test_resume_reuses_matching_terminal_artifact_without_second_evaluation(
    tmp_path: Path,
) -> None:
    class StopBeforeEvaluation(_Adapter):
        def evaluation_manifest(self):
            raise KeyboardInterrupt("before evaluation access")

    with pytest.raises(KeyboardInterrupt):
        run_evolution(
            StopBeforeEvaluation(),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    state = RunState.from_dict(
        json.loads((tmp_path / "run-state.json").read_text(encoding="utf-8"))
    )
    assert state.next_stage == "terminal-evaluation"
    evaluation_manifest = _manifest("evaluation")
    (tmp_path / "evaluation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_sha256": state.accepted.checkpoint_sha256,
                "evaluation_manifest_sha256": evaluation_manifest.digest,
                "result": _terminal_result(),
            }
        ),
        encoding="utf-8",
    )

    resumed = _Adapter()
    closed = run_evolution(
        resumed,
        tmp_path,
        config=PipelineConfig(rounds=1),
        history_plotter=_history_plotter,
    )
    assert closed.closed and closed.evaluation_completed
    assert resumed.evaluation_calls == 0


def test_resume_rejects_parseable_inconsistent_run_state(tmp_path: Path) -> None:
    adapter = _Adapter(interrupt_at="edge")
    with pytest.raises(KeyboardInterrupt):
        run_evolution(
            adapter,
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    state_path = tmp_path / "run-state.json"
    document = json.loads(state_path.read_text(encoding="utf-8"))
    document["closed"] = True
    state_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ResumeMismatchError, match="inconsistent"):
        run_evolution(
            _Adapter(),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )


def test_resume_rejects_incomplete_terminal_evaluation_artifact(
    tmp_path: Path,
) -> None:
    class StopBeforeEvaluation(_Adapter):
        def evaluation_manifest(self):
            raise KeyboardInterrupt("before evaluation access")

    with pytest.raises(KeyboardInterrupt):
        run_evolution(
            StopBeforeEvaluation(),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )
    state = RunState.from_dict(
        json.loads((tmp_path / "run-state.json").read_text(encoding="utf-8"))
    )
    evaluation_manifest = _manifest("evaluation")
    (tmp_path / "evaluation.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_sha256": state.accepted.checkpoint_sha256,
                "evaluation_manifest_sha256": evaluation_manifest.digest,
                "result": {"task_count": 400},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProgressConflictError, match="result"):
        run_evolution(
            _Adapter(),
            tmp_path,
            config=PipelineConfig(rounds=1),
            history_plotter=_history_plotter,
        )


def test_terminal_evaluation_document_rejects_schema_lineage_and_score_drift() -> None:
    initial = RunState.initial(
        PipelineConfig(rounds=1), _manifest(), _candidate("terminal")
    )
    state = replace(
        initial,
        sequence=1,
        next_stage="terminal-evaluation",
        terminal_reason="round-budget",
    )
    manifest = _manifest("evaluation")
    document = {
        "schema_version": 1,
        "checkpoint_sha256": state.accepted.checkpoint_sha256,
        "evaluation_manifest_sha256": manifest.digest,
        "result": _terminal_result(),
    }

    def rejected(mutator) -> None:
        corrupted = json.loads(json.dumps(document))
        mutator(corrupted)
        with pytest.raises(ProgressConflictError, match="evaluation result"):
            evolve._validate_terminal_evaluation(corrupted, state, manifest)

    rejected(lambda value: value.update(schema_version=0))
    rejected(lambda value: value.update(checkpoint_sha256="0" * 64))
    rejected(lambda value: value.update(result=[]))
    rejected(lambda value: value["result"].update(strict_task_pass_at_1_count=1))
    rejected(
        lambda value: value.update(
            result={
                "task_count": "400",
                "strict_task_pass_at_1_count": "400",
                "task_ids": list(manifest.task_ids),
                "task_exact": ["false"] * 400,
                "task_loss": ["0.0"] * 400,
                "mean_unresolved_task_loss": "0.0",
                "finite": "false",
            }
        )
    )


def test_store_reconciles_durable_progress_before_stale_state(tmp_path: Path) -> None:
    store = PipelineStore(tmp_path)
    config = PipelineConfig(rounds=1)
    manifest = _manifest()
    accepted = _candidate("initial", topology_changed=False)
    state = RunState.initial(config, manifest, accepted)
    store.write_state(state)
    trained_source = _candidate("trained", exact_count=1, topology_changed=False)
    trained = evolve._persist_selected(
        _Adapter(),
        store,
        trained_source,
        stage_id="r000-train",
        parent_checkpoint_sha256=state.accepted.checkpoint_sha256,
        config=config,
    )
    advanced = replace(
        state,
        sequence=1,
        cursor=128,
        next_stage="edge",
        accepted=trained,
    )
    record = store.progress_record(
        state,
        advanced,
        stage_id="r000-train",
        stage="train",
        parent=state.accepted,
        selected=advanced.accepted,
        attempts=(CandidateAttempt.completed("training", trained_source),),
        dispositions={"training": "accepted"},
        elapsed_seconds=1.0,
    )
    store.append_progress(record)

    reconciled = store.load_state(config, manifest)
    assert reconciled.sequence == 1
    assert reconciled.accepted.candidate_id == "trained"
    assert json.loads(store.state_path.read_text())["sequence"] == 1
    store.append_progress(record)
    assert len(store.read_progress()) == 1


def test_store_rejects_state_that_conflicts_with_its_progress_record(
    tmp_path: Path,
) -> None:
    store = PipelineStore(tmp_path)
    config = PipelineConfig(rounds=1)
    manifest = _manifest()
    initial = RunState.initial(config, manifest, _candidate("initial"))
    trained_source = _candidate("trained", exact_count=1)
    trained = evolve._persist_selected(
        _Adapter(),
        store,
        trained_source,
        stage_id="r000-train",
        parent_checkpoint_sha256=initial.accepted.checkpoint_sha256,
        config=config,
    )
    advanced = replace(
        initial,
        sequence=1,
        cursor=128,
        next_stage="edge",
        accepted=trained,
    )
    record = store.progress_record(
        initial,
        advanced,
        stage_id="r000-train",
        stage="train",
        parent=initial.accepted,
        selected=advanced.accepted,
        attempts=(CandidateAttempt.completed("training", trained_source),),
        dispositions={"training": "accepted"},
        elapsed_seconds=1.0,
    )
    store.append_progress(record)
    store.write_state(replace(advanced, accepted=_candidate("tampered")))
    with pytest.raises(ProgressConflictError, match="run state"):
        store.load_state(config, manifest)


def test_resume_rejects_initial_and_progress_lifecycle_jumps(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(rounds=1)
    manifest = _manifest()
    initial = RunState.initial(config, manifest, _candidate("initial"))

    initial_jump = PipelineStore(tmp_path / "initial-jump")
    document = initial.to_dict()
    document["next_stage"] = "edge"
    initial_jump.state_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ResumeMismatchError, match="Initial"):
        initial_jump.load_state(config, manifest)

    progress_jump = PipelineStore(tmp_path / "progress-jump")
    progress_jump.write_state(initial)
    advanced = replace(
        initial,
        sequence=1,
        cursor=128,
        next_stage="edge",
        accepted=_candidate("trained", exact_count=1),
    )
    record = progress_jump.progress_record(
        initial,
        advanced,
        stage_id="r000-train",
        stage="train",
        parent=initial.accepted,
        selected=advanced.accepted,
        attempts=(CandidateAttempt.completed("training", advanced.accepted),),
        dispositions={"training": "accepted"},
        elapsed_seconds=1.0,
    )
    record["stage"] = "dale"
    record["stage_id"] = "r000-dale"
    progress_jump.append_progress(record)
    with pytest.raises(ProgressConflictError, match="stage"):
        progress_jump.load_state(config, manifest)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    (
        ("schema_version", 0, "schema"),
        ("disposition", "terminal", "disposition"),
        ("round", 1, "round evidence"),
    ),
)
def test_resume_rejects_progress_scalar_evidence_drift(
    tmp_path: Path,
    field: str,
    bad_value: object,
    message: str,
) -> None:
    config = PipelineConfig(rounds=1)
    manifest = _manifest()
    initial = RunState.initial(config, manifest, _candidate("initial"))
    advanced = replace(
        initial,
        sequence=1,
        cursor=128,
        next_stage="edge",
        accepted=_candidate("trained", exact_count=1),
    )
    store = PipelineStore(tmp_path)
    store.write_state(initial)
    record = store.progress_record(
        initial,
        advanced,
        stage_id="r000-train",
        stage="train",
        parent=initial.accepted,
        selected=advanced.accepted,
        attempts=(CandidateAttempt.completed("training", advanced.accepted),),
        dispositions={"training": "accepted"},
        elapsed_seconds=1.0,
    )
    record[field] = bad_value
    store.append_progress(record)

    with pytest.raises(ProgressConflictError, match=message):
        store.load_state(config, manifest)


def test_store_rejects_malformed_or_conflicting_journals(tmp_path: Path) -> None:
    pending_store = PipelineStore(tmp_path / "pending")
    pending = {"stage_id": "r000-train", "state_after": {}}
    pending_store.write_pending(pending)
    pending_store.write_pending(pending)
    with pytest.raises(ProgressConflictError, match="different pending"):
        pending_store.write_pending({**pending, "stage_id": "r000-edge"})
    pending_store.pending_path.write_text("{", encoding="utf-8")
    with pytest.raises(ProgressConflictError, match="unreadable"):
        pending_store.read_pending()
    pending_store.pending_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ProgressConflictError, match="JSON object"):
        pending_store.read_pending()
    pending_store.clear_pending()
    pending_store.clear_pending()

    incomplete_store = PipelineStore(tmp_path / "incomplete")
    incomplete_store.progress_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ProgressConflictError, match="incomplete"):
        incomplete_store.read_progress()
    malformed_store = PipelineStore(tmp_path / "malformed")
    malformed_store.progress_path.write_text("{\n", encoding="utf-8")
    with pytest.raises(ProgressConflictError, match="malformed"):
        malformed_store.read_progress()

    conflict_store = PipelineStore(tmp_path / "conflict")
    record = {
        "stage_id": "one",
        "sequence_after": 1,
        "state_before": {},
        "state_after": {},
    }
    with pytest.raises(ValueError, match="stage_id"):
        conflict_store.append_progress({})
    conflict_store.append_progress(record)
    with pytest.raises(ProgressConflictError, match="stage one"):
        conflict_store.append_progress({**record, "extra": True})
    with pytest.raises(ProgressConflictError, match="sequence"):
        conflict_store.append_progress(
            {
                "stage_id": "two",
                "sequence_after": 1,
                "state_before": {},
                "state_after": {},
            }
        )


def test_store_rejects_unreadable_drifted_or_gapped_resume_evidence(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(rounds=1)
    manifest = _manifest()
    state = RunState.initial(config, manifest, _candidate("initial"))

    unreadable = PipelineStore(tmp_path / "unreadable")
    unreadable.state_path.write_text("{", encoding="utf-8")
    with pytest.raises(ResumeMismatchError, match="cannot be read"):
        unreadable.load_state(config, manifest)

    drifted = PipelineStore(tmp_path / "drifted")
    drifted.write_state(state)
    changed_digests = list(manifest.source_digests)
    changed_digests[0] = "f" * 64
    with pytest.raises(ResumeMismatchError, match="manifest differs"):
        drifted.load_state(
            config, replace(manifest, source_digests=tuple(changed_digests))
        )

    gap = PipelineStore(tmp_path / "gap")
    gap.write_state(state)
    gap.append_progress(
        {
            "stage_id": "gap",
            "sequence_before": 1,
            "sequence_after": 2,
            "state_before": state.position_dict(),
            "state_after": state.position_dict(),
        }
    )
    with pytest.raises(ProgressConflictError, match="inconsistent"):
        gap.load_state(config, manifest)

    malformed = PipelineStore(tmp_path / "bad-state-after")
    malformed.write_state(state)
    malformed.append_progress(
        {
            "stage_id": "bad",
            "sequence_before": 0,
            "sequence_after": 1,
            "state_before": state.position_dict(),
            "state_after": {},
        }
    )
    with pytest.raises(
        ProgressConflictError, match="lifecycle evidence is inconsistent"
    ):
        malformed.load_state(config, manifest)

    wrong_sequence = PipelineStore(tmp_path / "wrong-sequence")
    wrong_sequence.write_state(state)
    state_after = state.position_dict()
    state_after["sequence"] = 2
    wrong_sequence.append_progress(
        {
            "stage_id": "wrong",
            "sequence_before": 0,
            "sequence_after": 1,
            "state_before": state.position_dict(),
            "state_after": state_after,
        }
    )
    with pytest.raises(
        ProgressConflictError, match="lifecycle evidence is inconsistent"
    ):
        wrong_sequence.load_state(config, manifest)


def test_run_state_rejects_invalid_provenance_counters_and_candidates() -> None:
    state = RunState.initial(
        PipelineConfig(rounds=1), _manifest(), _candidate("accepted")
    )
    with pytest.raises(ValueError, match="training manifest"):
        replace(state, training_manifest=_manifest("evaluation"))
    with pytest.raises(ValueError, match="counters"):
        replace(state, cursor=-1)
    mismatched_score = replace(
        state.accepted.score,
        task_ids=tuple(f"other-{task_id}" for task_id in state.accepted.score.task_ids),
    )
    with pytest.raises(ValueError, match="candidate evidence"):
        replace(state, accepted=replace(state.accepted, score=mismatched_score))

    document = state.to_dict()
    document["schema_version"] = 0
    with pytest.raises(ResumeMismatchError, match="schema"):
        RunState.from_dict(document)
    document = state.to_dict()
    document["config_sha256"] = "0" * 64
    with pytest.raises(ResumeMismatchError, match="configuration digest"):
        RunState.from_dict(document)
    document = state.to_dict()
    document["training_manifest_sha256"] = "0" * 64
    with pytest.raises(ResumeMismatchError, match="manifest digest"):
        RunState.from_dict(document)


def test_history_plot_contains_exact_loss_topology_and_bytes(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    store = PipelineStore(tmp_path)
    config = PipelineConfig(rounds=1)
    manifest = _manifest()
    state = replace(
        RunState.initial(config, manifest, _candidate("initial")),
        sequence=1,
        cursor=128,
        next_stage="edge",
    )
    next_state = replace(
        state,
        sequence=2,
        cursor=256,
        next_stage="neuron",
        accepted=_candidate("next", exact_count=3, loss=4.0),
    )
    record = store.progress_record(
        state,
        next_state,
        stage_id="r000-edge",
        stage="edge",
        parent=state.accepted,
        selected=next_state.accepted,
        attempts=(
            CandidateAttempt.completed("add", next_state.accepted),
            CandidateAttempt.blocked("prune", "no candidate"),
        ),
        dispositions={"add": "accepted", "prune": "blocked"},
        elapsed_seconds=2.0,
    )
    output = tmp_path / "history.png"
    evidence = plot_score_history((record,), output)
    assert output.is_file() and output.stat().st_size > 0
    assert evidence == {
        "accepted_stage_count": 2,
        "output": str(output),
    }


def test_history_includes_baseline_and_skips_nonchanging_transitions(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    baseline = _candidate("baseline", exact_count=1, persistent_bytes=1_000)
    accepted = _candidate("accepted", exact_count=2, persistent_bytes=900)

    def point(
        stage_id: str,
        disposition: str,
        candidate: CandidateSnapshot,
    ) -> dict[str, object]:
        return {
            "stage_id": stage_id,
            "disposition": disposition,
            "selected_candidate_id": candidate.candidate_id,
            "child_checkpoint_sha256": candidate.checkpoint_sha256,
            "exact_task_count": candidate.score.exact_count,
            "unresolved_task_loss": candidate.score.unresolved_loss,
            "neurons": candidate.resources.neurons,
            "recurrent_edges": candidate.resources.recurrent_edges,
            "persistent_bytes": candidate.resources.persistent_bytes,
            "state_before": {"accepted": baseline.to_dict()},
            "state_after": {"accepted": candidate.to_dict()},
        }

    records = (
        point("r000-train", "accepted", accepted),
        point("r000-edge", "retained-parent", accepted),
        point("r000-round-end", "retained-parent", accepted),
        point("terminal-evaluation", "terminal", accepted),
    )
    output = tmp_path / "deduplicated-history.png"

    evidence = plot_score_history(records, output)

    assert output.is_file()
    assert evidence["accepted_stage_count"] == 2
