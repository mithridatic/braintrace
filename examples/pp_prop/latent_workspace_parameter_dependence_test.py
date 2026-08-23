"""Tests for checkpoint-conditioned logits and parameter-dependence evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import pathlib

import numpy as np
import pytest

from examples.pp_prop.latent_workspace_analysis import DecodedCandidate, OutputLogits

_EXACT_COUNT_NAMES = (
    "query_pass_at_1_count",
    "query_pass_at_2_count",
    "strict_task_pass_at_1_count",
    "strict_task_pass_at_2_count",
)


def _subject():
    return importlib.import_module(
        "examples.pp_prop.latent_workspace_parameter_dependence"
    )


def _digest(symbol: str) -> str:
    return symbol * 64


def _configuration_sha256(configuration: dict[str, object]) -> str:
    payload = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _leaf_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _refresh_run_digests(run: dict[str, object]) -> None:
    run["candidate_output_sha256"] = _canonical_sha256(run["candidate_sha256"])
    run["exact_membership_sha256"] = _canonical_sha256(run["exact_membership"])


def _baseline_run() -> dict[str, object]:
    leaves = [
        {
            "name": "decoder.weight",
            "shape": [1],
            "dtype": "<f4",
            "sha256": _digest("3"),
        },
        {
            "name": "reservoir.weight",
            "shape": [1],
            "dtype": "<f4",
            "sha256": _digest("4"),
        },
    ]
    run = {
        "checkpoint_sha256": _digest("0"),
        "parameter_sha256": _digest("1"),
        "source_revision": "a" * 40,
        "source_dirty": True,
        "manifest_sha256": _digest("b"),
        "topology_sha256": _digest("c"),
        "parameter_schema_sha256": _canonical_sha256(
            [{key: leaf[key] for key in ("name", "shape", "dtype")} for leaf in leaves]
        ),
        "ordered_parameter_leaves": leaves,
        "ordered_parameter_leaves_sha256": _canonical_sha256(leaves),
        "parameter_dependencies": ["decoder.weight", "reservoir.weight"],
        "dale_dependency": {"claimed": False, "status": "not_claimed"},
        "query_manifest_sha256": _digest("2"),
        "exact_counts": dict.fromkeys(_EXACT_COUNT_NAMES, 4),
        "cumulative_score": 16,
        "candidate_sha256": [
            [_digest("3"), _digest("4")],
            [_digest("5"), _digest("6")],
            [_digest("7"), _digest("8")],
            [_digest("9"), _digest("a")],
        ],
        "exact_membership": [
            [True, True],
            [True, True],
            [True, True],
            [True, True],
        ],
    }
    _refresh_run_digests(run)
    return run


def _perturbation(
    name: str,
    *,
    checkpoint_symbol: str,
    parameter_symbol: str,
    candidate_symbol: str,
    changed_count: str,
    query_index: int,
) -> dict[str, object]:
    run = copy.deepcopy(_baseline_run())
    run["checkpoint_sha256"] = _digest(checkpoint_symbol)
    run["parameter_sha256"] = _digest(parameter_symbol)
    counts = run["exact_counts"]
    assert isinstance(counts, dict)
    counts[changed_count] = 3
    run["cumulative_score"] = 15
    candidates = run["candidate_sha256"]
    assert isinstance(candidates, list)
    candidates[query_index][0] = _digest(candidate_symbol)
    membership = run["exact_membership"]
    assert isinstance(membership, list)
    membership[query_index][0] = False
    leaves = run["ordered_parameter_leaves"]
    assert isinstance(leaves, list)
    leaves[0]["sha256"] = _digest(parameter_symbol)
    run["ordered_parameter_leaves_sha256"] = _canonical_sha256(leaves)
    _refresh_run_digests(run)
    run["mutation"] = {
        "kind": name,
        **(
            {"factor": 0.5}
            if name == "scale"
            else {"run_id": "independent-seed-73"}
            if name == "swap"
            else {"seed": 73}
        ),
    }
    return run


def _valid_evidence() -> dict[str, object]:
    baseline = _baseline_run()
    return {
        "baseline": baseline,
        "repeat_intact": copy.deepcopy(baseline),
        "perturbations": {
            "scale": _perturbation(
                "scale",
                checkpoint_symbol="b",
                parameter_symbol="c",
                candidate_symbol="d",
                changed_count="query_pass_at_1_count",
                query_index=0,
            ),
            "swap": _perturbation(
                "swap",
                checkpoint_symbol="c",
                parameter_symbol="d",
                candidate_symbol="e",
                changed_count="strict_task_pass_at_1_count",
                query_index=1,
            ),
            "reseed": _perturbation(
                "reseed",
                checkpoint_symbol="d",
                parameter_symbol="e",
                candidate_symbol="f",
                changed_count="query_pass_at_2_count",
                query_index=2,
            ),
        },
    }


def _assert_failed(evidence: object) -> dict[str, object]:
    report = _subject().assess_parameter_dependence(evidence)
    assert report["passed"] is False
    assert isinstance(report["checks"], dict)
    assert not all(report["checks"].values())
    assert report["reasons"]
    return report


def _model_logits(preferred_color: int) -> OutputLogits:
    colors = np.zeros((30, 30, 10), dtype=np.float32)
    colors[0, 0, preferred_color] = 8.0
    return OutputLogits(
        np.zeros((30,), dtype=np.float32),
        np.zeros((30,), dtype=np.float32),
        colors,
    )


def _demonstration_candidates() -> tuple[DecodedCandidate, DecodedCandidate]:
    return (
        DecodedCandidate([[0]], log_probability=-0.1),
        DecodedCandidate([[1]], log_probability=-0.2),
    )


def test_checkpoint_conditioned_candidates_change_rank_with_model_logits() -> None:
    first = _subject().checkpoint_conditioned_candidates(
        _demonstration_candidates(), _model_logits(preferred_color=0), 1.0
    )
    second = _subject().checkpoint_conditioned_candidates(
        _demonstration_candidates(), _model_logits(preferred_color=1), 1.0
    )

    assert len(first) == 2
    assert len(second) == 2
    assert {np.asarray(item.grid).tobytes() for item in first} == {
        np.asarray(item.grid).tobytes() for item in second
    }
    assert not np.array_equal(first[0].grid, second[0].grid)
    assert all(np.isfinite(item.log_probability) for item in (*first, *second))


@pytest.mark.parametrize("coupling", [0.0, -1.0, np.nan, np.inf, True])
def test_checkpoint_conditioned_candidates_reject_nonpositive_or_nonfinite_coupling(
    coupling: float,
) -> None:
    with pytest.raises(ValueError, match="coupling"):
        _subject().checkpoint_conditioned_candidates(
            _demonstration_candidates(), _model_logits(preferred_color=0), coupling
        )


@pytest.mark.parametrize("candidate_mode", ["one", "duplicate"])
def test_checkpoint_conditioned_candidates_require_two_distinct_proposals(
    candidate_mode: str,
) -> None:
    first, _second = _demonstration_candidates()
    candidates = (first,) if candidate_mode == "one" else (first, first)

    with pytest.raises(ValueError):
        _subject().checkpoint_conditioned_candidates(
            candidates, _model_logits(preferred_color=0), 1.0
        )


def test_checkpoint_conditioned_candidates_reject_wrong_container_and_logits() -> None:
    with pytest.raises(TypeError):
        _subject().checkpoint_conditioned_candidates(
            _demonstration_candidates(), object(), 1.0
        )
    with pytest.raises(TypeError):
        _subject().checkpoint_conditioned_candidates(
            "not-candidates", _model_logits(preferred_color=0), 1.0
        )


@pytest.mark.parametrize(
    ("location", "nonfinite"),
    [
        ("proposal", np.nan),
        ("proposal", np.inf),
        ("height", np.nan),
        ("width", np.inf),
        ("colors", -np.inf),
    ],
)
def test_checkpoint_conditioned_candidates_reject_nonfinite_scores_and_logits(
    location: str, nonfinite: float
) -> None:
    candidates = list(_demonstration_candidates())
    logits = _model_logits(preferred_color=0)
    if location == "proposal":
        object.__setattr__(candidates[0], "log_probability", nonfinite)
    else:
        values = np.asarray(getattr(logits, location)).copy()
        values.flat[0] = nonfinite
        object.__setattr__(logits, location, values)

    with pytest.raises(ValueError, match="finite"):
        _subject().checkpoint_conditioned_candidates(candidates, logits, 1.0)


def test_complete_moving_perturbation_matrix_passes_at_cumulative_16() -> None:
    report = _subject().assess_parameter_dependence(_valid_evidence())

    assert report["passed"] is True
    assert report["minimum_cumulative_score"] == 16
    assert report["baseline_cumulative_score"] == 16
    assert isinstance(report["checks"], dict)
    assert report["checks"]
    assert all(report["checks"].values())
    assert report["reasons"] == []


def test_cumulative_15_fails_even_when_every_dependence_control_moves() -> None:
    evidence = _valid_evidence()
    for name in ("baseline", "repeat_intact"):
        run = evidence[name]
        assert isinstance(run, dict)
        counts = run["exact_counts"]
        assert isinstance(counts, dict)
        counts["strict_task_pass_at_2_count"] = 3
        run["cumulative_score"] = 15
    perturbations = evidence["perturbations"]
    assert isinstance(perturbations, dict)
    for run in perturbations.values():
        assert isinstance(run, dict)
        counts = run["exact_counts"]
        assert isinstance(counts, dict)
        counts["strict_task_pass_at_2_count"] = 3
        run["cumulative_score"] = sum(counts.values())

    report = _assert_failed(evidence)

    assert any("16" in str(reason) for reason in report["reasons"])


@pytest.mark.parametrize(
    "field",
    [
        "checkpoint_sha256",
        "parameter_sha256",
        "query_manifest_sha256",
        "exact_counts",
        "cumulative_score",
        "candidate_sha256",
        "exact_membership",
    ],
)
def test_repeat_intact_must_match_every_causal_output_field(field: str) -> None:
    evidence = _valid_evidence()
    repeat = evidence["repeat_intact"]
    assert isinstance(repeat, dict)
    if field.endswith("sha256"):
        repeat[field] = _digest("f")
    elif field == "exact_counts":
        repeat[field] = dict.fromkeys(_EXACT_COUNT_NAMES, 3)
    elif field == "cumulative_score":
        repeat[field] = 15
    elif field == "candidate_sha256":
        repeat[field][0][0] = _digest("f")
    else:
        repeat[field][0][0] = False

    _assert_failed(evidence)


@pytest.mark.parametrize("missing", ["scale", "swap", "reseed"])
def test_all_three_named_perturbations_are_required(missing: str) -> None:
    evidence = _valid_evidence()
    perturbations = evidence["perturbations"]
    assert isinstance(perturbations, dict)
    perturbations.pop(missing)

    _assert_failed(evidence)


@pytest.mark.parametrize("flat_part", ["score", "outputs"])
def test_each_perturbation_must_move_score_and_decoded_outputs(
    flat_part: str,
) -> None:
    evidence = _valid_evidence()
    baseline = evidence["baseline"]
    perturbations = evidence["perturbations"]
    assert isinstance(baseline, dict)
    assert isinstance(perturbations, dict)
    scale = perturbations["scale"]
    assert isinstance(scale, dict)
    if flat_part == "score":
        scale["exact_counts"] = copy.deepcopy(baseline["exact_counts"])
        scale["cumulative_score"] = baseline["cumulative_score"]
    else:
        scale["candidate_sha256"] = copy.deepcopy(baseline["candidate_sha256"])
        scale["exact_membership"] = copy.deepcopy(baseline["exact_membership"])

    _assert_failed(evidence)


def test_digest_only_checkpoint_changes_do_not_prove_parameter_dependence() -> None:
    evidence = _valid_evidence()
    baseline = evidence["baseline"]
    perturbations = evidence["perturbations"]
    assert isinstance(baseline, dict)
    assert isinstance(perturbations, dict)
    for run in perturbations.values():
        assert isinstance(run, dict)
        for field in (
            "exact_counts",
            "cumulative_score",
            "candidate_sha256",
            "exact_membership",
        ):
            run[field] = copy.deepcopy(baseline[field])

    _assert_failed(evidence)


@pytest.mark.parametrize(
    "corrupt",
    [
        "missing_baseline",
        "bad_digest",
        "wrong_manifest",
        "three_exact_counts",
        "inconsistent_cumulative",
        "malformed_candidate_pair",
        "malformed_membership_pair",
        "missing_mutation_metadata",
        "wrong_mutation_kind",
    ],
)
def test_malformed_perturbation_evidence_fails_closed(corrupt: str) -> None:
    evidence = _valid_evidence()
    perturbations = evidence["perturbations"]
    assert isinstance(perturbations, dict)
    scale = perturbations["scale"]
    assert isinstance(scale, dict)
    if corrupt == "missing_baseline":
        evidence.pop("baseline")
    elif corrupt == "bad_digest":
        scale["parameter_sha256"] = "not-a-sha256"
    elif corrupt == "wrong_manifest":
        scale["query_manifest_sha256"] = _digest("f")
    elif corrupt == "three_exact_counts":
        counts = scale["exact_counts"]
        assert isinstance(counts, dict)
        counts.pop("strict_task_pass_at_2_count")
        scale["cumulative_score"] = sum(counts.values())
    elif corrupt == "inconsistent_cumulative":
        scale["cumulative_score"] = 999
    elif corrupt == "malformed_candidate_pair":
        scale["candidate_sha256"][0] = [_digest("f")]
    elif corrupt == "malformed_membership_pair":
        scale["exact_membership"][0] = [True]
    elif corrupt == "missing_mutation_metadata":
        scale.pop("mutation")
    else:
        scale["mutation"] = {"kind": "reseed", "seed": 73}

    _assert_failed(evidence)


def test_malformed_evidence_does_not_raise_instead_of_failing_closed() -> None:
    report = _subject().assess_parameter_dependence({"baseline": object()})

    assert report["passed"] is False
    assert report["reasons"]


def _write_small_checkpoint(path) -> dict[str, np.ndarray]:
    architecture = np.frombuffer(
        b'{"schema_version":1,"memory_coding":"frozen"}', dtype=np.uint8
    )
    values = {
        "arr_0": np.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32),
        "arr_1": np.asarray([0.25, -0.5, 1.5], dtype=np.float64),
        "__architecture__": architecture,
    }
    np.savez(path, **values)
    return values


def _loaded_checkpoint(path) -> dict[str, np.ndarray]:
    with np.load(path) as stored:
        return {name: np.asarray(stored[name]) for name in stored.files}


def test_scale_checkpoint_preserves_schema_and_changes_ordered_parameters(
    tmp_path,
) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "scaled.npz"
    original = _write_small_checkpoint(source)

    report = _subject().write_checkpoint_perturbation(
        source, destination, kind="scale", factor=0.5
    )

    scaled = _loaded_checkpoint(destination)
    assert tuple(scaled) == tuple(original)
    for name in original:
        assert scaled[name].shape == original[name].shape
        assert scaled[name].dtype == original[name].dtype
    assert np.array_equal(scaled["__architecture__"], original["__architecture__"])
    assert np.array_equal(scaled["arr_0"], original["arr_0"] * np.float32(0.5))
    assert np.array_equal(scaled["arr_1"], original["arr_1"] * np.float64(0.5))
    assert report["kind"] == "scale"
    assert report["factor"] == 0.5
    assert report["schema_identical"] is True
    assert report["parameter_names"] == ["arr_0", "arr_1"]
    assert report["source_checkpoint_sha256"] != report["checkpoint_sha256"]
    assert report["source_parameter_sha256"] != report["parameter_sha256"]


def test_checkpoint_writer_preserves_architecture_first_archive_order(tmp_path) -> None:
    """Production checkpoints store architecture metadata before parameters."""

    source = tmp_path / "architecture-first.npz"
    destination = tmp_path / "scaled.npz"
    architecture = np.frombuffer(b'{"schema_version":1}', dtype=np.uint8)
    np.savez(
        source,
        __architecture__=architecture,
        arr_0=np.asarray([1.0, -2.0], dtype=np.float32),
    )

    report = _subject().write_checkpoint_perturbation(
        source, destination, kind="scale", factor=0.5
    )

    assert report["schema_identical"] is True
    with np.load(destination, allow_pickle=False) as stored:
        assert stored.files == ["__architecture__", "arr_0"]


def test_perturb_checkpoint_cli_creates_scale_and_reports_success(
    tmp_path, capsys
) -> None:
    source = tmp_path / "source.npz"
    destination = tmp_path / "scale.npz"
    _write_small_checkpoint(source)

    code = _subject().main(
        [
            "perturb-checkpoint",
            "--source",
            str(source),
            "--destination",
            str(destination),
            "--kind",
            "scale",
            "--factor",
            "0.5",
        ]
    )

    assert code == 0
    assert destination.is_file()
    assert '"schema_identical": true' in capsys.readouterr().out


def test_reseed_checkpoint_is_deterministic_and_uses_brainstate_random(
    tmp_path, monkeypatch
) -> None:
    subject = _subject()
    source = tmp_path / "source.npz"
    first_path = tmp_path / "reseed-73-a.npz"
    second_path = tmp_path / "reseed-73-b.npz"
    other_path = tmp_path / "reseed-74.npz"
    original = _write_small_checkpoint(source)
    observed_seeds: list[int] = []
    real_random_state = subject.brainstate.random.RandomState

    def observed_random_state(seed):
        observed_seeds.append(int(seed))
        return real_random_state(seed)

    monkeypatch.setattr(subject.brainstate.random, "RandomState", observed_random_state)
    first_report = subject.write_checkpoint_perturbation(
        source, first_path, kind="reseed", seed=73
    )
    second_report = subject.write_checkpoint_perturbation(
        source, second_path, kind="reseed", seed=73
    )
    other_report = subject.write_checkpoint_perturbation(
        source, other_path, kind="reseed", seed=74
    )

    first = _loaded_checkpoint(first_path)
    second = _loaded_checkpoint(second_path)
    other = _loaded_checkpoint(other_path)
    assert observed_seeds == [73, 73, 74]
    assert first_report["parameter_sha256"] == second_report["parameter_sha256"]
    assert first_report["parameter_sha256"] != other_report["parameter_sha256"]
    assert first_report["source_parameter_sha256"] != first_report["parameter_sha256"]
    for name in ("arr_0", "arr_1"):
        assert first[name].shape == original[name].shape
        assert first[name].dtype == original[name].dtype
        assert first[name].tobytes() == second[name].tobytes()
    assert any(
        first[name].tobytes() != other[name].tobytes() for name in ("arr_0", "arr_1")
    )
    assert np.array_equal(first["__architecture__"], original["__architecture__"])


@pytest.mark.parametrize("factor", [None, 0.0, -0.5, 1.0, np.nan, np.inf, True])
def test_checkpoint_scale_rejects_invalid_factor(tmp_path, factor: object) -> None:
    source = tmp_path / "source.npz"
    _write_small_checkpoint(source)

    with pytest.raises(ValueError, match="factor"):
        _subject().write_checkpoint_perturbation(
            source, tmp_path / "invalid.npz", kind="scale", factor=factor
        )


@pytest.mark.parametrize(
    ("kind", "arguments"),
    [
        ("swap", {}),
        ("unknown", {}),
        ("reseed", {}),
        ("reseed", {"seed": -1}),
        ("reseed", {"seed": True}),
    ],
)
def test_checkpoint_perturbation_rejects_invalid_kind_or_seed(
    tmp_path, kind: str, arguments: dict[str, object]
) -> None:
    source = tmp_path / "source.npz"
    _write_small_checkpoint(source)

    with pytest.raises(ValueError, match="kind|seed"):
        _subject().write_checkpoint_perturbation(
            source, tmp_path / "invalid.npz", kind=kind, **arguments
        )


@pytest.mark.parametrize("corrupt", ["missing_architecture", "noncanonical_names"])
def test_checkpoint_perturbation_rejects_invalid_schema(tmp_path, corrupt: str) -> None:
    source = tmp_path / "source.npz"
    architecture = np.frombuffer(b'{"schema_version":1}', dtype=np.uint8)
    if corrupt == "missing_architecture":
        np.savez(source, arr_0=np.ones((2,), dtype=np.float32))
    else:
        np.savez(
            source,
            arr_0=np.ones((2,), dtype=np.float32),
            arr_2=np.ones((3,), dtype=np.float32),
            __architecture__=architecture,
        )

    with pytest.raises(ValueError, match="schema|architecture|parameter"):
        _subject().write_checkpoint_perturbation(
            source, tmp_path / "invalid.npz", kind="scale", factor=0.5
        )


@pytest.mark.parametrize(
    "corrupt",
    ["nonfloating", "architecture_dtype", "architecture_json", "architecture_scalar"],
)
def test_checkpoint_audit_rejects_unsafe_arrays_and_architecture(
    tmp_path, corrupt: str
) -> None:
    source = tmp_path / "invalid.npz"
    parameter = np.ones((2,), dtype=np.float32)
    architecture = np.frombuffer(b'{"schema_version":1}', dtype=np.uint8)
    if corrupt == "nonfloating":
        parameter = np.ones((2,), dtype=np.int32)
    elif corrupt == "architecture_dtype":
        architecture = architecture.astype(np.int16)
    elif corrupt == "architecture_json":
        architecture = np.frombuffer(b"not-json", dtype=np.uint8)
    elif corrupt == "architecture_scalar":
        architecture = np.frombuffer(b"[]", dtype=np.uint8)
    np.savez(source, arr_0=parameter, __architecture__=architecture)

    with pytest.raises(ValueError):
        _subject().write_checkpoint_perturbation(
            source, tmp_path / "output.npz", kind="scale", factor=0.5
        )


def _checkpoint_candidate(grid: list[list[int]], rank: int) -> dict[str, object]:
    array = np.asarray(grid, dtype=np.int8)
    return {
        "rank": rank,
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "grid": array.tolist(),
        "changed_decision": None if rank == 1 else "color:0:0",
        "log_probability": -float(rank),
        "provenance": "model",
        "dependency_class": "model_checkpoint",
        "proposal_source": "demonstration_fitted_forest",
        "ranking_source": "trained_network_factorized_candidate_log_probability",
        "answer_head_version": "checkpoint_conditioned_v1",
        "source_checkpoint": 60,
        "selection_role": "checkpoint_conditioned_rank",
        "forest_rank": rank,
        "forest_log_probability": -0.25 * rank,
        "network_log_probability": -0.75 * rank,
        "combined_log_probability": -float(rank),
        "parameter_dependencies": ["decoder.weight", "reservoir.weight"],
    }


def _checkpoint_query(
    task_id: str,
    query_index: int,
    first: int,
    second: int,
    *,
    pass_at_1: bool,
    pass_at_2: bool,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "query_index": query_index,
        "primary_candidate_mode": "model_only",
        "submission_role": "primary_submission",
        "candidates": [
            _checkpoint_candidate([[first]], 1),
            _checkpoint_candidate([[second]], 2),
        ],
        "score": {
            "task_id": task_id,
            "query_index": query_index,
            "pass_at_1": pass_at_1,
            "pass_at_2": pass_at_2,
            "shape_accuracy_diagnostic": True,
            "valid_cell_pixel_accuracy_diagnostic": float(pass_at_1),
            "candidate_count": 2,
        },
    }


def _checkpoint_conditioned_result(
    *,
    checkpoint_sha256: str | None = None,
    parameter_sha256: str | None = None,
    seed: int = 31337,
) -> dict[str, object]:
    queries = [
        _checkpoint_query("task-a", 0, 1, 2, pass_at_1=True, pass_at_2=True),
        _checkpoint_query("task-a", 1, 3, 4, pass_at_1=False, pass_at_2=True),
        _checkpoint_query("task-b", 0, 5, 6, pass_at_1=True, pass_at_2=True),
    ]
    checkpoint_digest = checkpoint_sha256 or _digest("0")
    parameter_digest = parameter_sha256 or _digest("1")
    configuration = {
        "answer_head": "checkpoint_conditioned",
        "checkpoint_conditioning_coupling": 1.0,
        "submission_checkpoint": 60,
        "primary_candidate_mode": "model_only",
        "seed": seed,
        "neuron_count": 4096,
        "recurrent_edges": 4_194_304,
        "latent_steps": 60,
        "checkpoints": [0, 30, 60],
        "device": "gpu",
        "smoke": False,
        "structural_only": False,
        "evaluation_task_limit": None,
    }
    topology_sha256 = _digest("2")
    ordered_leaves = [
        {
            "name": "decoder.weight",
            "shape": [1],
            "dtype": "<f4",
            "sha256": _digest("3"),
        },
        {
            "name": "reservoir.weight",
            "shape": [1],
            "dtype": "<f4",
            "sha256": _digest("4"),
        },
    ]
    return {
        "schema_version": 2,
        "protocol_version": 2,
        "configuration": configuration,
        "configuration_sha256": _configuration_sha256(configuration),
        "implementation": {
            "source_revision": "a" * 40,
            "source_dirty": True,
        },
        "data_summary": {"manifest_sha256": _digest("5")},
        "model": {
            "neuron_count": 4096,
            "recurrent_edge_count": 4_194_304,
            "topology_sha256": topology_sha256,
            "neuron_typing": {
                "mode": "none",
                "recurrent_sign_violation_count": None,
            },
        },
        "device": {"platform": "gpu"},
        "training": {
            "performed": False,
            "parameter_checkpoint_sha256": checkpoint_digest,
            "parameter_sha256_before": parameter_digest,
            "parameter_sha256_after": parameter_digest,
        },
        "evaluation": {
            "parameter_sha256_before": parameter_digest,
            "parameter_sha256_after": parameter_digest,
            "parameter_binding": {
                "checkpoint_sha256": checkpoint_digest,
                "parameter_sha256": parameter_digest,
                "topology_sha256": topology_sha256,
                "ordered_leaves": ordered_leaves,
            },
            "metrics_by_effort": {
                "60": {
                    "query_count": 3,
                    "task_count": 2,
                    "query_pass_at_1": 2 / 3,
                    "query_pass_at_2": 1.0,
                    "strict_task_pass_at_1": 0.5,
                    "strict_task_pass_at_2": 1.0,
                    "tasks": {
                        "task-a": {
                            "query_count": 2,
                            "pass_at_1": False,
                            "pass_at_2": True,
                        },
                        "task-b": {
                            "query_count": 1,
                            "pass_at_1": True,
                            "pass_at_2": True,
                        },
                    },
                }
            },
            "submitted_completion": {
                "evaluated_task_count": 2,
                "evaluated_query_count": 3,
                "exact_query_count_at_1": 2,
                "exact_query_count_at_2": 3,
                "exact_task_count_at_1": 1,
                "exact_task_count_at_2": 2,
            },
            "checkpoint_queries": {"60": queries},
        },
    }


def test_parameter_dependence_run_extracts_canonical_exact_evidence() -> None:
    run = _subject().parameter_dependence_run_from_result(
        _checkpoint_conditioned_result()
    )

    assert run["checkpoint_sha256"] == _digest("0")
    assert run["parameter_sha256"] == _digest("1")
    assert (
        run["configuration_sha256"]
        == _checkpoint_conditioned_result()["configuration_sha256"]
    )
    assert run["source_revision"] == "a" * 40
    assert run["source_dirty"] is True
    assert run["manifest_sha256"] == _digest("5")
    assert run["topology_sha256"] == _digest("2")
    assert run["parameter_dependencies"] == [
        "decoder.weight",
        "reservoir.weight",
    ]
    assert run["dale_dependency"] == {
        "claimed": False,
        "status": "not_claimed",
    }
    assert run["query_manifest"] == [
        ["task-a", 0],
        ["task-a", 1],
        ["task-b", 0],
    ]
    assert len(run["query_manifest_sha256"]) == 64
    assert len(run["candidate_sha256"]) == 3
    assert all(len(pair) == 2 for pair in run["candidate_sha256"])
    assert run["exact_membership"] == [
        [True, True],
        [False, True],
        [True, True],
    ]
    assert run["strict_task_ids"] == ["task-a", "task-b"]
    assert run["strict_task_membership"] == [[False, True], [True, True]]
    assert run["exact_counts"] == {
        "query_pass_at_1_count": 2,
        "query_pass_at_2_count": 3,
        "strict_task_pass_at_1_count": 1,
        "strict_task_pass_at_2_count": 2,
    }
    assert run["cumulative_score"] == 8
    assert len(run["candidate_output_sha256"]) == 64
    assert len(run["exact_membership_sha256"]) == 64


def test_parameter_dependence_run_accepts_verified_zero_violation_dale_evidence() -> (
    None
):
    result = _checkpoint_conditioned_result()
    result["model"]["neuron_typing"] = {
        "mode": "ei_dale",
        "excitatory_count": 8,
        "inhibitory_count": 2,
        "recurrent_sign_violation_count": 0,
    }

    run = _subject().parameter_dependence_run_from_result(result)

    assert run["dale_dependency"] == {
        "claimed": True,
        "status": "verified_zero_recurrent_sign_violations",
        "excitatory_count": 8,
        "inhibitory_count": 2,
        "recurrent_sign_violation_count": 0,
    }


def test_parameter_dependence_run_hashes_ordered_prediction_bytes_only() -> None:
    subject = _subject()
    result = _checkpoint_conditioned_result()
    original = subject.parameter_dependence_run_from_result(result)
    metadata_changed = copy.deepcopy(result)
    changed_candidate = metadata_changed["evaluation"]["checkpoint_queries"]["60"][0][
        "candidates"
    ][0]
    changed_candidate["checkpoint_path"] = "metadata-only-change.npz"
    metadata_only = subject.parameter_dependence_run_from_result(metadata_changed)
    reranked = copy.deepcopy(result)
    candidates = reranked["evaluation"]["checkpoint_queries"]["60"][0]["candidates"]
    candidates.reverse()
    candidates[0]["rank"] = 1
    candidates[1]["rank"] = 2
    reranked_run = subject.parameter_dependence_run_from_result(reranked)

    assert metadata_only["candidate_sha256"] == original["candidate_sha256"]
    assert reranked_run["candidate_sha256"] != original["candidate_sha256"]
    assert reranked_run["query_manifest_sha256"] == original["query_manifest_sha256"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("forest_log_probability", np.nan),
        ("network_log_probability", np.inf),
        ("combined_log_probability", -np.inf),
        ("log_probability", np.nan),
        ("combined_log_probability", -999.0),
    ],
)
def test_parameter_dependence_run_rejects_nonfinite_or_inconsistent_scores(
    field: str, value: float
) -> None:
    result = _checkpoint_conditioned_result()
    candidate = result["evaluation"]["checkpoint_queries"]["60"][0]["candidates"][0]
    candidate[field] = value

    with pytest.raises(ValueError, match="finite|combined|score"):
        _subject().parameter_dependence_run_from_result(result)


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        ("raw_head", "answer_head|checkpoint"),
        ("rule_provenance", "provenance|model"),
        ("missing_dependency_class", "dependency_class|provenance"),
        ("missing_parameter_dependencies", "parameter_dependencies|dependency"),
        ("wrong_answer_path", "answer_head_version|checkpoint"),
        ("wrong_checkpoint", "source_checkpoint|checkpoint"),
        ("parameter_mismatch", "parameter|frozen"),
    ],
)
def test_parameter_dependence_run_rejects_noncausal_or_mismatched_result(
    corrupt: str, message: str
) -> None:
    result = _checkpoint_conditioned_result()
    candidate = result["evaluation"]["checkpoint_queries"]["60"][0]["candidates"][0]
    if corrupt == "raw_head":
        result["configuration"]["answer_head"] = "demonstration_fitted"
    elif corrupt == "rule_provenance":
        candidate["provenance"] = "rule"
    elif corrupt == "missing_dependency_class":
        candidate.pop("dependency_class")
    elif corrupt == "missing_parameter_dependencies":
        candidate["parameter_dependencies"] = []
    elif corrupt == "wrong_answer_path":
        candidate["answer_head_version"] = "demonstration_fitted_v1"
    elif corrupt == "wrong_checkpoint":
        candidate["source_checkpoint"] = 30
    else:
        result["evaluation"]["parameter_sha256_after"] = _digest("2")

    with pytest.raises(ValueError, match=message):
        _subject().parameter_dependence_run_from_result(result)


@pytest.mark.parametrize(
    "corrupt",
    ["duplicate_query", "missing_rank", "bad_rank", "score_identity", "invalid_exact"],
)
def test_parameter_dependence_run_rejects_incomplete_query_evidence(
    corrupt: str,
) -> None:
    result = _checkpoint_conditioned_result()
    queries = result["evaluation"]["checkpoint_queries"]["60"]
    if corrupt == "duplicate_query":
        queries[1]["task_id"] = "task-a"
        queries[1]["query_index"] = 0
        queries[1]["score"]["task_id"] = "task-a"
        queries[1]["score"]["query_index"] = 0
    elif corrupt == "missing_rank":
        queries[0]["candidates"].pop()
    elif corrupt == "bad_rank":
        queries[0]["candidates"][1]["rank"] = 1
    elif corrupt == "score_identity":
        queries[0]["score"]["query_index"] = 9
    else:
        queries[0]["score"]["pass_at_1"] = True
        queries[0]["score"]["pass_at_2"] = False

    with pytest.raises(ValueError):
        _subject().parameter_dependence_run_from_result(result)


@pytest.mark.parametrize(
    "corrupt",
    [
        "configuration_hash",
        "coupling",
        "source_revision",
        "source_dirty",
        "manifest",
        "training_parameter",
        "binding_checkpoint",
        "binding_parameter",
        "binding_topology",
        "model_topology",
        "leaf_order",
        "leaf_hash",
        "candidate_dependencies",
        "reported_metrics",
        "reported_completion",
        "dale_claim_without_evidence",
    ],
)
def test_parameter_dependence_run_rejects_unbound_artifact_fields(
    corrupt: str,
) -> None:
    result = _checkpoint_conditioned_result()
    if corrupt == "configuration_hash":
        result["configuration_sha256"] = _digest("f")
    elif corrupt == "coupling":
        result["configuration"]["checkpoint_conditioning_coupling"] = 0.5
        result["configuration_sha256"] = _configuration_sha256(result["configuration"])
    elif corrupt == "source_revision":
        result["implementation"]["source_revision"] = "unknown"
    elif corrupt == "source_dirty":
        result["implementation"]["source_dirty"] = "yes"
    elif corrupt == "manifest":
        result["data_summary"]["manifest_sha256"] = "bad"
    elif corrupt == "training_parameter":
        result["training"]["parameter_sha256_after"] = _digest("f")
    elif corrupt == "binding_checkpoint":
        result["evaluation"]["parameter_binding"]["checkpoint_sha256"] = _digest("f")
    elif corrupt == "binding_parameter":
        result["evaluation"]["parameter_binding"]["parameter_sha256"] = _digest("f")
    elif corrupt == "binding_topology":
        result["evaluation"]["parameter_binding"]["topology_sha256"] = _digest("f")
    elif corrupt == "model_topology":
        result["model"]["topology_sha256"] = _digest("f")
    elif corrupt == "leaf_order":
        result["evaluation"]["parameter_binding"]["ordered_leaves"].reverse()
    elif corrupt == "leaf_hash":
        result["evaluation"]["parameter_binding"]["ordered_leaves"][0]["sha256"] = "bad"
    elif corrupt == "candidate_dependencies":
        result["evaluation"]["checkpoint_queries"]["60"][0]["candidates"][0][
            "parameter_dependencies"
        ].reverse()
    elif corrupt == "reported_metrics":
        result["evaluation"]["metrics_by_effort"]["60"]["query_pass_at_1"] = 0.0
    elif corrupt == "reported_completion":
        result["evaluation"]["submitted_completion"]["exact_task_count_at_2"] = 0
    else:
        result["model"]["neuron_typing"] = {
            "mode": "ei_dale",
            "recurrent_sign_violation_count": 1,
        }

    with pytest.raises(ValueError):
        _subject().parameter_dependence_run_from_result(result)


@pytest.mark.parametrize(
    "corrupt",
    [
        "empty",
        "nonmapping",
        "duplicate_name",
        "shape_type",
        "zero_dimension",
        "dtype_invalid",
        "dtype_noncanonical",
        "dtype_nonfloating",
    ],
)
def test_parameter_dependence_run_rejects_malformed_ordered_leaf_schema(
    corrupt: str,
) -> None:
    result = _checkpoint_conditioned_result()
    leaves = result["evaluation"]["parameter_binding"]["ordered_leaves"]
    if corrupt == "empty":
        leaves.clear()
    elif corrupt == "nonmapping":
        leaves[0] = "leaf"
    elif corrupt == "duplicate_name":
        leaves[1]["name"] = leaves[0]["name"]
    elif corrupt == "shape_type":
        leaves[0]["shape"] = "1"
    elif corrupt == "zero_dimension":
        leaves[0]["shape"] = [0]
    elif corrupt == "dtype_invalid":
        leaves[0]["dtype"] = object()
    elif corrupt == "dtype_noncanonical":
        leaves[0]["dtype"] = "float32"
    else:
        leaves[0]["dtype"] = "<i4"

    with pytest.raises(ValueError):
        _subject().parameter_dependence_run_from_result(result)


@pytest.mark.parametrize("corrupt", ["missing", "mode", "empty_population"])
def test_parameter_dependence_run_rejects_incomplete_dale_status(
    corrupt: str,
) -> None:
    result = _checkpoint_conditioned_result()
    if corrupt == "missing":
        result["model"].pop("neuron_typing")
    elif corrupt == "mode":
        result["model"]["neuron_typing"] = {"mode": "unknown"}
    else:
        result["model"]["neuron_typing"] = {
            "mode": "ei_dale",
            "excitatory_count": 0,
            "inhibitory_count": 0,
            "recurrent_sign_violation_count": 0,
        }

    with pytest.raises(ValueError):
        _subject().parameter_dependence_run_from_result(result)


@pytest.mark.parametrize(
    "corrupt",
    ["metrics_type", "missing_checkpoint", "tasks_manifest", "completion_type"],
)
def test_parameter_dependence_run_rejects_malformed_reported_aggregates(
    corrupt: str,
) -> None:
    result = _checkpoint_conditioned_result()
    if corrupt == "metrics_type":
        result["evaluation"]["metrics_by_effort"] = []
    elif corrupt == "missing_checkpoint":
        result["evaluation"]["metrics_by_effort"] = {}
    elif corrupt == "tasks_manifest":
        result["evaluation"]["metrics_by_effort"]["60"]["tasks"] = {}
    else:
        result["evaluation"]["submitted_completion"] = []

    with pytest.raises(ValueError):
        _subject().parameter_dependence_run_from_result(result)


def _checkpoint_leaf_binding(path: pathlib.Path) -> list[dict[str, object]]:
    values = _loaded_checkpoint(path)
    return [
        {
            "name": name,
            "shape": list(values[f"arr_{index}"].shape),
            "dtype": values[f"arr_{index}"].dtype.str,
            "sha256": _leaf_sha256(values[f"arr_{index}"]),
        }
        for index, name in enumerate(("decoder.weight", "reservoir.weight"))
    ]


def _bind_result_to_checkpoint(
    path: pathlib.Path,
    *,
    arm: str,
) -> dict[str, object]:
    parameter_arm = "baseline" if arm == "repeat_intact" else arm
    result = _checkpoint_conditioned_result(
        checkpoint_sha256=_file_sha256(path),
        parameter_sha256=hashlib.sha256(
            f"parameter-{parameter_arm}".encode()
        ).hexdigest(),
    )
    result["configuration"]["parameter_checkpoint"] = str(path)
    result["configuration"]["output_dir"] = f"out-{arm}"
    result["configuration_sha256"] = _configuration_sha256(result["configuration"])
    result["evaluation"]["parameter_binding"]["ordered_leaves"] = (
        _checkpoint_leaf_binding(path)
    )
    result["evaluation"].pop("metrics_by_effort")
    result["evaluation"].pop("submitted_completion")
    if arm != "baseline" and arm != "repeat_intact":
        query_index = {"scale": 0, "swap": 1, "reseed": 2}[arm]
        candidates = result["evaluation"]["checkpoint_queries"]["60"][query_index][
            "candidates"
        ]
        candidates.reverse()
        candidates[0]["rank"] = 1
        candidates[1]["rank"] = 2
        score = result["evaluation"]["checkpoint_queries"]["60"][query_index]["score"]
        score["pass_at_1"] = not score["pass_at_1"]
        score["pass_at_2"] = True
    return result


def _training_origin(path: pathlib.Path, *, seed: int) -> dict[str, object]:
    configuration = {
        "seed": seed,
        "parameter_checkpoint": str(path),
        "training_updates": 700,
    }
    return {
        "configuration": configuration,
        "configuration_sha256": _configuration_sha256(configuration),
        "training": {
            "performed": True,
            "parameter_checkpoint_sha256": _file_sha256(path),
            "parameter_sha256_after": hashlib.sha256(
                f"origin-{seed}".encode()
            ).hexdigest(),
        },
    }


def _matrix_inputs(tmp_path):
    subject = _subject()
    baseline = tmp_path / "baseline.npz"
    scale = tmp_path / "scale.npz"
    swap = tmp_path / "swap.npz"
    reseed = tmp_path / "reseed.npz"
    original = _write_small_checkpoint(baseline)
    subject.write_checkpoint_perturbation(baseline, scale, kind="scale", factor=0.5)
    swapped = {
        name: (
            value.copy()
            if name == "__architecture__"
            else np.asarray(value * -0.25, dtype=value.dtype)
        )
        for name, value in original.items()
    }
    np.savez(swap, **swapped)
    subject.write_checkpoint_perturbation(baseline, reseed, kind="reseed", seed=73)
    paths = {
        "baseline": baseline,
        "scale": scale,
        "swap": swap,
        "reseed": reseed,
    }
    results = {
        name: _bind_result_to_checkpoint(
            baseline if name == "repeat_intact" else paths[name], arm=name
        )
        for name in ("baseline", "repeat_intact", "scale", "swap", "reseed")
    }
    origins = {
        "baseline": _training_origin(baseline, seed=31337),
        "swap": _training_origin(swap, seed=73),
    }
    return results, origins, paths


def test_matrix_builder_audits_checkpoint_bytes_and_training_origins(
    tmp_path,
) -> None:
    results, origins, paths = _matrix_inputs(tmp_path)

    artifact = _subject().build_parameter_dependence_matrix(
        results,
        training_origins=origins,
        checkpoints=paths,
        scale_factor=0.5,
        reseed_seed=73,
        minimum_cumulative_score=0,
        require_complete_arc=False,
    )

    assert artifact["schema_version"] == 2
    assert artifact["qualification"]["passed"] is True
    assert artifact["qualification"]["checks"]["accepted_full_profile"] is True
    assert artifact["accepted_full_profile"]["required"]["checkpoints"] == [
        0,
        30,
        60,
    ]
    assert all(
        arm["passed"] is True
        for arm in artifact["accepted_full_profile"]["arms"].values()
    )
    assert artifact["origins"]["baseline"]["training_performed"] is True
    assert artifact["origins"]["swap"]["seed"] == 73
    assert artifact["arms"]["scale"]["verified_intervention"] == {
        "kind": "scale",
        "factor": 0.5,
        "exact_all_leaves": True,
    }
    assert artifact["arms"]["reseed"]["verified_intervention"] == {
        "kind": "reseed",
        "seed": 73,
        "exact_all_leaves": True,
        "generator": "brainstate.random.RandomState",
    }
    assert (
        artifact["arms"]["swap"]["verified_intervention"]["independently_trained"]
        is True
    )
    assert (
        artifact["arms"]["repeat_intact"]["checkpoint"]["sha256"]
        == artifact["arms"]["baseline"]["checkpoint"]["sha256"]
    )


@pytest.mark.parametrize(
    "corrupt",
    [
        "reduced_neurons",
        "reduced_edges",
        "wrong_latent_steps",
        "wrong_checkpoints",
        "wrong_seed",
        "cpu",
        "smoke",
    ],
)
def test_matrix_builder_never_promotes_a_nonaccepted_full_profile(
    tmp_path, corrupt: str
) -> None:
    results, origins, paths = _matrix_inputs(tmp_path)
    for result in results.values():
        configuration = result["configuration"]
        if corrupt == "reduced_neurons":
            configuration["neuron_count"] = 2048
            result["model"]["neuron_count"] = 2048
        elif corrupt == "reduced_edges":
            configuration["recurrent_edges"] = 16_384
            result["model"]["recurrent_edge_count"] = 16_384
        elif corrupt == "wrong_latent_steps":
            configuration["latent_steps"] = 32
        elif corrupt == "wrong_checkpoints":
            configuration["checkpoints"] = [0, 8, 16, 32]
        elif corrupt == "wrong_seed":
            configuration["seed"] = 9999
        elif corrupt == "cpu":
            configuration["device"] = "cpu"
            result["device"]["platform"] = "cpu"
        else:
            configuration["smoke"] = True
        result["configuration_sha256"] = _configuration_sha256(configuration)

    artifact = _subject().build_parameter_dependence_matrix(
        results,
        training_origins=origins,
        checkpoints=paths,
        scale_factor=0.5,
        reseed_seed=73,
        minimum_cumulative_score=0,
        require_complete_arc=False,
    )

    assert artifact["qualification"]["checks"]["accepted_full_profile"] is False
    assert artifact["qualification"]["passed"] is False
    assert artifact["approved_completion_target_passed"] is False
    assert all(
        arm["passed"] is False
        for arm in artifact["accepted_full_profile"]["arms"].values()
    )


@pytest.mark.parametrize(
    "corrupt",
    [
        "scale_value",
        "reseed_value",
        "swap_schema",
        "result_checkpoint_binding",
        "origin_checkpoint_binding",
        "origin_performed",
        "same_training_seed",
        "origin_config_hash",
        "nonfinite_checkpoint",
    ],
)
def test_matrix_builder_rejects_asserted_or_corrupt_checkpoint_provenance(
    tmp_path, corrupt: str
) -> None:
    results, origins, paths = _matrix_inputs(tmp_path)
    if corrupt in {"scale_value", "reseed_value", "nonfinite_checkpoint"}:
        target_name = "scale" if corrupt == "scale_value" else "reseed"
        target = paths[target_name]
        values = _loaded_checkpoint(target)
        values["arr_0"] = values["arr_0"].copy()
        values["arr_0"].flat[0] = (
            np.nan if corrupt == "nonfinite_checkpoint" else values["arr_0"].flat[0] + 1
        )
        np.savez(target, **values)
        results[target_name] = _bind_result_to_checkpoint(target, arm=target_name)
    elif corrupt == "swap_schema":
        values = _loaded_checkpoint(paths["swap"])
        values["arr_0"] = values["arr_0"].reshape(-1)
        np.savez(paths["swap"], **values)
        results["swap"] = _bind_result_to_checkpoint(paths["swap"], arm="swap")
        origins["swap"] = _training_origin(paths["swap"], seed=73)
    elif corrupt == "result_checkpoint_binding":
        results["scale"]["training"]["parameter_checkpoint_sha256"] = _digest("f")
    elif corrupt == "origin_checkpoint_binding":
        origins["baseline"]["training"]["parameter_checkpoint_sha256"] = _digest("f")
    elif corrupt == "origin_performed":
        origins["swap"]["training"]["performed"] = False
    elif corrupt == "same_training_seed":
        origins["swap"]["configuration"]["seed"] = 31337
        origins["swap"]["configuration_sha256"] = _configuration_sha256(
            origins["swap"]["configuration"]
        )
    else:
        origins["swap"]["configuration_sha256"] = _digest("f")

    with pytest.raises(ValueError):
        _subject().build_parameter_dependence_matrix(
            results,
            training_origins=origins,
            checkpoints=paths,
            scale_factor=0.5,
            reseed_seed=73,
            minimum_cumulative_score=0,
            require_complete_arc=False,
        )


def test_matrix_cli_requires_and_audits_origins_and_checkpoint_files(
    tmp_path,
) -> None:
    results, origins, paths = _matrix_inputs(tmp_path)
    json_paths = {}
    for name, value in {
        **results,
        **{f"{k}_origin": v for k, v in origins.items()},
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        json_paths[name] = path
    output = tmp_path / "artifact.json"
    argv = ["assess-matrix"]
    for name in ("baseline", "repeat_intact", "scale", "swap", "reseed"):
        argv.extend((f"--{name.replace('_', '-')}", str(json_paths[name])))
    argv.extend(
        (
            "--baseline-origin",
            str(json_paths["baseline_origin"]),
            "--swap-origin",
            str(json_paths["swap_origin"]),
            "--baseline-checkpoint",
            str(paths["baseline"]),
            "--scale-checkpoint",
            str(paths["scale"]),
            "--swap-checkpoint",
            str(paths["swap"]),
            "--reseed-checkpoint",
            str(paths["reseed"]),
            "--scale-factor",
            "0.5",
            "--reseed-seed",
            "73",
            "--minimum-cumulative-score",
            "0",
            "--allow-incomplete-arc",
            "--output",
            str(output),
        )
    )

    assert _subject().main(argv) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["qualification"]["passed"] is True
    assert artifact["arms"]["scale"]["checkpoint"]["sha256"] == _file_sha256(
        paths["scale"]
    )


def test_matrix_cli_fails_before_writing_when_origin_is_not_trained(tmp_path) -> None:
    results, origins, paths = _matrix_inputs(tmp_path)
    origins["swap"]["training"]["performed"] = False
    json_paths = {}
    values = {**results, **{f"{key}_origin": value for key, value in origins.items()}}
    for name, value in values.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        json_paths[name] = path
    output = tmp_path / "should-not-exist.json"
    argv = ["assess-matrix"]
    for name in ("baseline", "repeat_intact", "scale", "swap", "reseed"):
        argv.extend((f"--{name.replace('_', '-')}", str(json_paths[name])))
    argv.extend(
        (
            "--baseline-origin",
            str(json_paths["baseline_origin"]),
            "--swap-origin",
            str(json_paths["swap_origin"]),
            "--baseline-checkpoint",
            str(paths["baseline"]),
            "--scale-checkpoint",
            str(paths["scale"]),
            "--swap-checkpoint",
            str(paths["swap"]),
            "--reseed-checkpoint",
            str(paths["reseed"]),
            "--scale-factor",
            "0.5",
            "--reseed-seed",
            "73",
            "--allow-incomplete-arc",
            "--output",
            str(output),
        )
    )

    with pytest.raises(ValueError, match="performed"):
        _subject().main(argv)
    assert not output.exists()
