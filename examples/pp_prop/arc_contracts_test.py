import json
import importlib.util
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
    with pytest.raises(ValueError, match="sequence"):
        arc.write_result(result, "bad")
    record = {"task_id": "x", "queries": [], "strict_pass_at_1": False}
    with pytest.raises(ValueError, match="task result fields"):
        arc.write_result(result, [{**record, "extra": 1}])
    with pytest.raises(ValueError, match="task_id"):
        arc.write_result(result, [{**record, "task_id": 1}])
    with pytest.raises(ValueError, match="queries"):
        arc.write_result(result, [{**record, "queries": "bad"}])
    query = {"query_index": 0, "prediction": [[1]], "target": [[1]], "exact": True}
    with pytest.raises(ValueError, match="query result fields"):
        arc.write_result(result, [{**record, "queries": [{**query, "extra": 1}]}])
    with pytest.raises(ValueError, match="nonnegative"):
        arc.write_result(result, [{**record, "queries": [{**query, "query_index": -1}]}])
    with pytest.raises(ValueError, match="exact"):
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
