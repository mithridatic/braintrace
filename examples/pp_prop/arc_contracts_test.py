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


def test_full_decoder_has_independent_cell_colors_and_result_recomputes_flags(tmp_path):
    values = np.zeros((30, 360))
    values[0, 0], values[0, 30 + 1] = 1, 2
    values[0, 60 + 0 * 10 + 3] = 5
    values[0, 60 + 1 * 10 + 4] = 5
    prediction = arc.decode_prediction(values)
    assert prediction.tolist() == [[3, 4]]
    result = tmp_path / "result.json"
    arc.write_result(result, [{"task_id": "d631b094", "queries": [{"query_index": 0, "prediction": [[1]], "target": [[2]], "exact": True}], "strict_pass_at_1": True}])
    assert json.loads(result.read_text())["tasks"][0]["queries"][0]["exact"] is False
    assert json.loads(result.read_text())["tasks"][0]["strict_pass_at_1"] is False


def test_checkpoint_rejects_nan_and_same_parent(tmp_path):
    checkpoint = tmp_path / "checkpoint.npz"
    with pytest.raises(ValueError, match="finite"):
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


def test_result_and_checkpoint_are_bounded_and_round_trip(tmp_path):
    result = tmp_path / "result.json"
    arc.write_result(result, [{"task_id": "x", "queries": [{"query_index": 0, "prediction": np.zeros((1, 1), dtype=np.uint8), "target": np.zeros((1, 1), dtype=np.uint8), "exact": True}], "strict_pass_at_1": True}])
    assert json.loads(result.read_text())["strict_task_pass_at_1_count"] == 1
    checkpoint = tmp_path / "checkpoint.npz"
    arc.write_checkpoint(checkpoint, {"weights": np.ones(3, dtype=np.float32)})
    loaded = arc.load_checkpoint(checkpoint)
    assert np.array_equal(loaded["weights"], np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="format"):
        arc.write_checkpoint(tmp_path / "bad.npz", {}, format=2)
