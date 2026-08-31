import hashlib
import importlib.util
import json
import sys

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location("arc_contracts", __file__.replace("_test.py", ".py"))
arc = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = arc
SPEC.loader.exec_module(arc)


def task(target=True):
    source = np.asarray([[0, 1], [2, 3]], dtype=np.uint8)
    answer = np.asarray([[4, 5]], dtype=np.uint8) if target else np.asarray([[6]], dtype=np.uint8)
    return arc.ARCTask("d631b094", ((source, np.asarray([[4, 5]], dtype=np.uint8)),), (source.copy(),), (answer,), "practice")


def _write_corpus(root, directory, count=400):
    path = root / "data" / directory
    path.mkdir(parents=True)
    payload = json.dumps({
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[2]], "output": [[3]]}],
    })
    for index in reversed(range(count)):
        (path / f"{index:08x}.json").write_text(payload, encoding="utf-8")
    return path


def _checkpoint_arrays(neuron_count, input_connections):
    input_counts = np.zeros(arc.EVENT_WIDTH, dtype=np.int32)
    input_counts[0] = input_connections
    return {
        "neuron_ids": np.arange(neuron_count, dtype=np.int32),
        "dale_codes": np.zeros(neuron_count, dtype=np.int8),
        "owner_codes": np.full(neuron_count, -1, dtype=np.int16),
        "mechanism_codes": np.zeros(neuron_count, dtype=np.uint8),
        "neuron_count": np.asarray(neuron_count, dtype=np.int32),
        "integration_substeps": np.asarray(1, dtype=np.int32),
        "input_indptr": np.concatenate((
            np.zeros(1, dtype=np.int32), np.cumsum(input_counts, dtype=np.int32),
        )),
        "input_indices": np.zeros(input_connections, dtype=np.int32),
        "input_values": np.zeros(input_connections, dtype=np.float32),
        "input_m1": np.zeros(input_connections, dtype=np.float32),
        "input_m2": np.zeros(input_connections, dtype=np.float32),
        "recurrent_indptr": np.zeros(neuron_count + 1, dtype=np.int32),
        "recurrent_indices": np.zeros(0, dtype=np.int32),
        "recurrent_values": np.zeros(0, dtype=np.float32),
        "recurrent_m1": np.zeros(0, dtype=np.float32),
        "recurrent_m2": np.zeros(0, dtype=np.float32),
        "readout_weight": np.zeros((neuron_count, 360), dtype=np.float32),
        "readout_bias": np.zeros(360, dtype=np.float32),
        "readout_weight_m1": np.zeros((neuron_count, 360), dtype=np.float32),
        "readout_weight_m2": np.zeros((neuron_count, 360), dtype=np.float32),
        "readout_bias_m1": np.zeros(360, dtype=np.float32),
        "readout_bias_m2": np.zeros(360, dtype=np.float32),
        "input_step": np.asarray(0, dtype=np.int64),
        "recurrent_step": np.asarray(0, dtype=np.int64),
        "readout_step": np.asarray(0, dtype=np.int64),
    }


def _manifest_sources():
    return tuple(
        arc.ARCCorpusSource(
            f"{index:08x}", f"data/training/{index:08x}.json", "0" * 64,
        )
        for index in range(400)
    )


def test_event_round_trip_and_target_isolation():
    first, advance = arc.encode_episode(task(True))
    second, _ = arc.encode_episode(task(False))
    assert first.shape == (705, 441)
    assert advance.shape == (705,)
    assert np.array_equal(first, second)
    pairs, query = arc.decode_episode(first)
    assert np.array_equal(pairs[0][0], task().demonstrations[0][0])
    assert np.array_equal(query, task().queries[0])


def test_event_and_loss_validation_edges():
    events, _ = arc.encode_episode(task())
    with pytest.raises(IndexError):
        arc.encode_episode(task(), 2)
    broken = events.copy()
    broken[1, 7] = False
    with pytest.raises(ValueError, match="header"):
        arc.decode_episode(broken)
    broken = events.copy()
    broken[2, 111:141] = False
    with pytest.raises(ValueError, match="declared grid"):
        arc.decode_episode(broken)
    with pytest.raises(ValueError, match="shape request"):
        arc.request_loss(np.zeros(59), np.zeros(2, dtype=int), request="shape")
    with pytest.raises(ValueError, match="row request"):
        arc.request_loss(np.zeros((30, 9)), np.zeros(30, dtype=int), request="row")
    with pytest.raises(ValueError, match="valid_mask"):
        arc.request_loss(np.zeros((30, 10)), np.zeros(30, dtype=int), request="row", valid_mask=np.zeros(29))


def test_loader_rejects_evaluation_and_invalid_grids(tmp_path):
    path = tmp_path / "data" / "training"
    path.mkdir(parents=True)
    (path / "d631b094.json").write_text(json.dumps({"train": [], "test": [{"input": [[1]], "output": [[2]]}]}))
    loaded = arc.load_task(tmp_path, "d631b094")
    assert loaded.queries[0].dtype == np.uint8
    with pytest.raises(ValueError, match="evaluation"):
        arc.load_task(tmp_path, "d631b094", "evaluation")
    (path / "d631b094.json").write_text(json.dumps({"train": [], "test": [{"input": [[10]], "output": [[2]]}]}))
    with pytest.raises(ValueError, match="integer colors"):
        arc.load_task(tmp_path, "d631b094")


def test_loader_rejects_bad_roles_paths_demo_counts_and_empty_tests(tmp_path):
    with pytest.raises(ValueError, match="role"):
        arc.load_task(tmp_path, "d631b094", "other")
    with pytest.raises(ValueError, match="declared"):
        arc.load_task(tmp_path, "unknown")
    with pytest.raises(ValueError, match="cannot be read"):
        arc.load_task(tmp_path, "d631b094")
    path = tmp_path / "data" / "training"
    path.mkdir(parents=True)
    (path / "d631b094.json").write_text(json.dumps({"train": [], "test": []}))
    with pytest.raises(ValueError, match="at least one"):
        arc.load_task(tmp_path, "d631b094")
    pairs = [{"input": [[1]], "output": [[1]]}] * 11
    (path / "d631b094.json").write_text(json.dumps({"train": pairs, "test": [{"input": [[1]]}]}))
    with pytest.raises(ValueError, match="maximum is 10"):
        arc.load_task(tmp_path, "d631b094")
    with pytest.raises(ValueError, match="rectangular"):
        arc.ARCTask("x", ((([[1], [2, 3]]), [[1]]),), ([[1]],), (None,), "practice")


def test_manifest_declares_exactly_400_sorted_training_sources(tmp_path):
    directory = _write_corpus(tmp_path, "training")
    evaluation = tmp_path / "data" / "evaluation"
    evaluation.mkdir()
    (evaluation / "must-not-be-read.json").write_text("not JSON", encoding="utf-8")
    manifest = arc.load_corpus_manifest(tmp_path)

    assert manifest.role == "practice"
    assert len(manifest.sources) == 400
    assert manifest.task_ids == tuple(sorted(manifest.task_ids))
    assert manifest.task_ids[0] == "00000000"
    assert manifest.task_ids[-1] == "0000018f"
    for source in manifest.sources:
        path = tmp_path / source.source_path
        assert source.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()

    task_id = manifest.task_ids[-1]
    with pytest.raises(ValueError, match="declared direct ARC task"):
        arc.load_task(tmp_path, task_id)
    loaded = arc.load_task(tmp_path, task_id, manifest=manifest)
    assert loaded.task_id == task_id
    assert loaded.role == "practice"

    (directory / f"{task_id}.json").write_text(
        json.dumps({"train": [], "test": [{"input": [[4]], "output": [[5]]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source digest"):
        arc.load_task(tmp_path, task_id, manifest=manifest)


def test_manifest_requires_exact_corpus_and_explicit_evaluation_access(tmp_path):
    _write_corpus(tmp_path, "training", count=399)
    with pytest.raises(ValueError, match="exactly 400"):
        arc.load_corpus_manifest(tmp_path)

    evaluation_root = tmp_path / "evaluation-root"
    _write_corpus(evaluation_root, "evaluation")
    with pytest.raises(ValueError, match="evaluation"):
        arc.load_corpus_manifest(evaluation_root, role="evaluation")
    manifest = arc.load_corpus_manifest(
        evaluation_root, role="evaluation", allow_evaluation=True,
    )
    with pytest.raises(ValueError, match="evaluation"):
        arc.load_task(
            evaluation_root, manifest.task_ids[0], role="evaluation",
            manifest=manifest,
        )
    loaded = arc.load_task(
        evaluation_root, manifest.task_ids[0], role="evaluation",
        manifest=manifest, allow_evaluation=True,
    )
    assert loaded.role == "evaluation"


def test_manifest_rejects_invalid_source_names_and_wrong_root(tmp_path):
    directory = _write_corpus(tmp_path, "training")
    (directory / "0000018f.json").rename(directory / "INVALID.json")
    with pytest.raises(ValueError, match="eight lowercase hexadecimal"):
        arc.load_corpus_manifest(tmp_path)

    other_root = tmp_path / "other"
    _write_corpus(other_root, "training")
    manifest = arc.load_corpus_manifest(other_root)
    with pytest.raises(ValueError, match="different ARC root"):
        arc.load_task(tmp_path, manifest.task_ids[0], manifest=manifest)
    with pytest.raises(ValueError, match="role does not match"):
        arc.load_task(
            other_root, manifest.task_ids[0], role="evaluation",
            allow_evaluation=True, manifest=manifest,
        )
    with pytest.raises(ValueError, match="not declared by"):
        arc.load_task(other_root, "ffffffff", manifest=manifest)
    with pytest.raises(TypeError, match="ARCCorpusManifest"):
        arc.load_task(other_root, manifest.task_ids[0], manifest=object())
    (other_root / manifest.sources[0].source_path).unlink()
    with pytest.raises(ValueError, match="source cannot be read"):
        arc.load_task(other_root, manifest.task_ids[0], manifest=manifest)


def test_manifest_value_objects_reject_forged_declarations(tmp_path):
    digest = "0" * 64
    with pytest.raises(ValueError, match="task IDs"):
        arc.ARCCorpusSource("INVALID", "data/training/INVALID.json", digest)
    with pytest.raises(ValueError, match="SHA-256"):
        arc.ARCCorpusSource("00000000", "data/training/00000000.json", "bad")
    with pytest.raises(ValueError, match="relative"):
        arc.ARCCorpusSource("00000000", "../00000000.json", digest)

    sources = _manifest_sources()
    with pytest.raises(ValueError, match="manifest role"):
        arc.ARCCorpusManifest(str(tmp_path), "other", sources)
    with pytest.raises(ValueError, match="exactly 400"):
        arc.ARCCorpusManifest(str(tmp_path), "practice", sources[:-1])
    with pytest.raises(ValueError, match="unique and sorted"):
        arc.ARCCorpusManifest(
            str(tmp_path), "practice", (sources[1], sources[0], *sources[2:]),
        )
    mismatched_paths = (
        arc.ARCCorpusSource("00000000", sources[1].source_path, digest),
        arc.ARCCorpusSource("00000001", sources[0].source_path, digest),
        *sources[2:],
    )
    with pytest.raises(ValueError, match="source paths must be sorted"):
        arc.ARCCorpusManifest(str(tmp_path), "practice", mismatched_paths)
    with pytest.raises(ValueError, match="paths do not match"):
        arc.ARCCorpusManifest(str(tmp_path), "evaluation", sources)


def test_manifest_rejects_missing_root_and_role_directory(tmp_path):
    with pytest.raises(ValueError, match="root cannot be read"):
        arc.load_corpus_manifest(tmp_path / "missing")
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(ValueError, match="directory cannot be read"):
        arc.load_corpus_manifest(tmp_path)


def test_full_decoder_has_independent_cell_colors_and_result_recomputes_flags(tmp_path):
    values = np.zeros((31, 360))
    values[0, 0], values[0, 30 + 1] = 1, 2
    values[1, 60 + 0 * 10 + 3] = 5
    values[1, 60 + 1 * 10 + 4] = 5
    prediction = arc.decode_prediction(values)
    assert prediction.tolist() == [[3, 4]]
    result = tmp_path / "result.json"
    arc.write_result(result, [{"task_id": "d631b094", "queries": [{"query_index": 0, "prediction": [[1]], "target": [[2]], "exact": True}], "strict_pass_at_1": True}])
    assert json.loads(result.read_text())["tasks"][0]["queries"][0]["exact"] is False
    assert json.loads(result.read_text())["tasks"][0]["strict_pass_at_1"] is False
    with pytest.raises(ValueError, match="integer color"):
        arc.write_result(result, [{"task_id": "x", "queries": [{"query_index": 0, "prediction": [[1.5]], "target": [[2]], "exact": False}], "strict_pass_at_1": False}])


def test_decoder_consumes_shape_and_thirty_row_request_states():
    values = np.zeros((31, 360))
    values[0, 2], values[0, 30 + 3] = 1, 2
    values[1, 60 + 0 * 10 + 4] = 5
    values[1, 60 + 1 * 10 + 5] = 5
    values[2, 60 + 0 * 10 + 6] = 5
    values[2, 60 + 1 * 10 + 7] = 5
    prediction = arc.decode_prediction(values)
    assert prediction.tolist() == [[4, 5, 0, 0], [6, 7, 0, 0], [0, 0, 0, 0]]
    with pytest.raises(ValueError, match="one shape and 30 row"):
        arc.decode_prediction(np.zeros((30, 360)))


def test_checkpoint_rejects_nan_and_same_parent(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"
    with pytest.raises(ValueError, match="schema"):
        arc.write_checkpoint(checkpoint, {"weights": np.asarray([np.nan])})
    with pytest.raises(ValueError, match="differ"):
        arc.write_checkpoint(checkpoint, {"weights": np.ones(1)}, parent=checkpoint)


def test_loss_decoder_and_strict_score():
    assert arc.request_loss(np.zeros(60), np.asarray([0, 1]), request="shape") > 0
    assert arc.request_loss(np.zeros((30, 10)), np.zeros(30, dtype=int), request="row", valid_mask=np.zeros(30)) == 0
    values = np.zeros(360)
    values[1], values[30 + 2] = 1, 2
    values[60 + 3] = 4
    prediction = arc.decode_prediction(values)
    assert prediction.shape == (2, 3)
    assert prediction.dtype == np.uint8
    assert arc.strict_task_pass_at_1([prediction], [prediction.copy()])
    assert not arc.query_exact(prediction.astype(float), prediction)


def test_non_request_and_invalid_row_loss_are_zero_and_strict_terms_change():
    logits = np.zeros(60)
    target = np.asarray([1, 2])
    assert arc.request_loss(logits, target, request=None) == 0.0
    assert arc.request_loss(logits, target, request="non-request") == 0.0
    assert arc.request_loss(logits, target, request="invalid-row") == 0.0
    shape_loss = arc.request_loss(logits, target, request="shape")
    changed = logits.copy()
    changed[target[0]] = 3.0
    assert arc.request_loss(changed, target, request="shape") < shape_loss
    row_logits = np.zeros((30, 10))
    row_target = np.zeros(30, dtype=int)
    row_loss = arc.request_loss(row_logits, row_target, request="row")
    row_logits[0, 0] = 3.0
    assert arc.request_loss(row_logits, row_target, request="row") < row_loss


def test_result_and_checkpoint_are_bounded_and_round_trip(tmp_path):
    result = tmp_path / "result.json"
    arc.write_result(result, [{"task_id": "x", "queries": [{"query_index": 0, "prediction": np.zeros((1, 1), dtype=np.uint8), "target": np.zeros((1, 1), dtype=np.uint8), "exact": True}], "strict_pass_at_1": True}])
    assert json.loads(result.read_text())["strict_task_pass_at_1_count"] == 1
    checkpoint = tmp_path / "checkpoint.npz"
    arrays = {
        "neuron_ids": np.arange(2, dtype=np.int32), "dale_codes": np.zeros(2, dtype=np.int8),
        "owner_codes": np.full(2, -1, dtype=np.int16), "mechanism_codes": np.zeros(2, dtype=np.uint8),
        "neuron_count": np.asarray(2, dtype=np.int32), "integration_substeps": np.asarray(1, dtype=np.int32),
        "input_indptr": np.zeros(arc.EVENT_WIDTH + 1, dtype=np.int32), "input_indices": np.zeros(0, dtype=np.int32),
        "input_values": np.zeros(0, dtype=np.float32), "input_m1": np.zeros(0, dtype=np.float32), "input_m2": np.zeros(0, dtype=np.float32),
        "recurrent_indptr": np.zeros(3, dtype=np.int32), "recurrent_indices": np.zeros(0, dtype=np.int32),
        "recurrent_values": np.zeros(0, dtype=np.float32), "recurrent_m1": np.zeros(0, dtype=np.float32), "recurrent_m2": np.zeros(0, dtype=np.float32),
        "readout_weight": np.zeros((2, 360), dtype=np.float32), "readout_bias": np.zeros(360, dtype=np.float32),
        "readout_weight_m1": np.zeros((2, 360), dtype=np.float32), "readout_weight_m2": np.zeros((2, 360), dtype=np.float32),
        "readout_bias_m1": np.zeros(360, dtype=np.float32), "readout_bias_m2": np.zeros(360, dtype=np.float32),
        "input_step": np.asarray(0, dtype=np.int64), "recurrent_step": np.asarray(0, dtype=np.int64), "readout_step": np.asarray(0, dtype=np.int64),
    }
    arc.write_checkpoint(checkpoint, arrays)
    loaded = arc.load_checkpoint(checkpoint)
    assert np.array_equal(loaded["neuron_ids"], arrays["neuron_ids"])
    with pytest.raises(ValueError, match="schema"):
        arc.write_checkpoint(tmp_path / "bad.npz", {}, format=2)
    with pytest.raises(ValueError, match="schema"):
        arc.write_checkpoint(tmp_path / "bad-schema.npz", {"weights": np.ones(1, dtype=np.float32)})


def test_result_schema_and_checkpoint_file_validation(tmp_path):
    result = tmp_path / "result.json"
    with pytest.raises(TypeError, match="sequence"):
        arc.write_result(result, "bad")
    record = {"task_id": "x", "queries": [], "strict_pass_at_1": False}
    with pytest.raises(ValueError, match="task result fields"):
        arc.write_result(result, [{**record, "extra": 1}])
    with pytest.raises(TypeError, match="task_id"):
        arc.write_result(result, [{**record, "task_id": 1}])
    with pytest.raises(TypeError, match="queries"):
        arc.write_result(result, [{**record, "queries": "bad"}])
    query = {"query_index": 0, "prediction": [[1]], "target": [[1]], "exact": True}
    with pytest.raises(ValueError, match="query result fields"):
        arc.write_result(result, [{**record, "queries": [{**query, "extra": 1}]}])
    with pytest.raises(ValueError, match="nonnegative"):
        arc.write_result(result, [{**record, "queries": [{**query, "query_index": -1}]}])
    with pytest.raises(TypeError, match="exact"):
        arc.write_result(result, [{**record, "queries": [{**query, "exact": 1}]}])
    malformed = tmp_path / "malformed.npz"
    np.savez(malformed, format=np.asarray([1]))
    with pytest.raises(ValueError, match="scalar"):
        arc.load_checkpoint(malformed)
    oversized = tmp_path / "oversized.npz"
    oversized.write_bytes(b"x" * (arc.MAX_CHECKPOINT_BYTES + 1))
    with pytest.raises(ValueError, match="32 MiB"):
        arc.load_checkpoint(oversized)


def test_checkpoint_enforces_input_destination_and_fixed_readout(tmp_path):
    arrays = {
        "neuron_ids": np.arange(2, dtype=np.int32), "dale_codes": np.zeros(2, dtype=np.int8),
        "owner_codes": np.full(2, -1, dtype=np.int16), "mechanism_codes": np.zeros(2, dtype=np.uint8),
        "neuron_count": np.asarray(2, dtype=np.int32), "integration_substeps": np.asarray(1, dtype=np.int32),
        "input_indptr": np.concatenate((np.array([0, 1]), np.ones(arc.EVENT_WIDTH - 1))).astype(np.int32), "input_indices": np.asarray([1], dtype=np.int32),
        "input_values": np.zeros(1, dtype=np.float32), "input_m1": np.zeros(1, dtype=np.float32), "input_m2": np.zeros(1, dtype=np.float32),
        "recurrent_indptr": np.zeros(3, dtype=np.int32), "recurrent_indices": np.zeros(0, dtype=np.int32),
        "recurrent_values": np.zeros(0, dtype=np.float32), "recurrent_m1": np.zeros(0, dtype=np.float32), "recurrent_m2": np.zeros(0, dtype=np.float32),
        "readout_weight": np.zeros((2, 360), dtype=np.float32), "readout_bias": np.zeros(360, dtype=np.float32),
        "readout_weight_m1": np.zeros((2, 360), dtype=np.float32), "readout_weight_m2": np.zeros((2, 360), dtype=np.float32),
        "readout_bias_m1": np.zeros(360, dtype=np.float32), "readout_bias_m2": np.zeros(360, dtype=np.float32),
        "input_step": np.asarray(0, dtype=np.int64), "recurrent_step": np.asarray(0, dtype=np.int64), "readout_step": np.asarray(0, dtype=np.int64),
    }
    arc.write_checkpoint(tmp_path / "valid.npz", arrays)
    arrays["input_indptr"] = np.zeros(2, dtype=np.int32)
    with pytest.raises(ValueError, match="input CSR structure"):
        arc.write_checkpoint(tmp_path / "bad-input.npz", arrays)
    arrays["input_indptr"] = np.concatenate((np.array([0, 1]), np.ones(arc.EVENT_WIDTH - 1))).astype(np.int32)
    arrays["input_indices"] = np.asarray([2], dtype=np.int32)
    with pytest.raises(ValueError, match="input CSR endpoint"):
        arc.write_checkpoint(tmp_path / "bad-endpoint.npz", arrays)
    arrays["input_indices"] = np.asarray([1], dtype=np.int32)
    arrays["readout_bias"] = np.zeros(359, dtype=np.float32)
    with pytest.raises(ValueError, match="readout parameters"):
        arc.write_checkpoint(tmp_path / "bad-readout.npz", arrays)


def test_checkpoint_uses_dynamic_per_neuron_connection_limit(tmp_path):
    above_legacy_limit = _checkpoint_arrays(31, 30_497)
    path = tmp_path / "dynamic-limit.npz"
    arc.write_checkpoint(path, above_legacy_limit)
    assert len(arc.load_checkpoint(path)["input_indices"]) == 30_497

    above_dynamic_limit = _checkpoint_arrays(2, 2_049)
    with pytest.raises(ValueError, match="1,024 connections per neuron"):
        arc.write_checkpoint(tmp_path / "above-dynamic-limit.npz", above_dynamic_limit)
