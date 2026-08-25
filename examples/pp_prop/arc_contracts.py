"""Direct ARC data and artifact contracts used by Example 21."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

MAX_GRID = 30
MAX_DEMONSTRATIONS = 10
EVENTS = 705
EVENT_WIDTH = 441
MAX_RESULT_BYTES = 256 * 1024
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024
MAX_BIOLOGICAL_CONNECTIONS = 30_496
TRAINING_ORDER = ("d631b094", "dc433765", "b782dc8a", "d06dbe63", "aedd82e4", "0b148d64", "b2862040", "150deff5")
VALIDATION_ORDER = ("46f33fce", "3428a4f5", "d8c310e9", "09629e4f")
PROOF_ORDER = ("d631b094", "46f33fce")


def _copy_grid(value: object, task_id: str, name: str) -> np.ndarray:
    try:
        rows = list(value)  # type: ignore[arg-type]
        if not rows or any(not isinstance(row, (list, tuple, np.ndarray)) for row in rows):
            raise ValueError
        width = len(rows[0])
        if not width or any(len(row) != width for row in rows):
            raise ValueError
        array = np.asarray(rows)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"task {task_id} {name} must be a rectangular grid") from exc
    if array.ndim != 2 or array.shape[0] > MAX_GRID or array.shape[1] > MAX_GRID:
        raise ValueError(f"task {task_id} {name} must be 1 through 30 by 1 through 30")
    if not np.issubdtype(array.dtype, np.integer) or np.any((array < 0) | (array > 9)):
        raise ValueError(f"task {task_id} {name} must contain integer colors from 0 through 9")
    result = np.array(array, dtype=np.uint8, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ARCTask:
    """Immutable ARC task with isolated input and target grids."""

    task_id: str
    demonstrations: tuple[tuple[np.ndarray, np.ndarray], ...]
    queries: tuple[np.ndarray, ...]
    targets: tuple[np.ndarray | None, ...]
    role: str

    def __post_init__(self) -> None:
        demos = tuple((_copy_grid(x, self.task_id, "demonstration input"), _copy_grid(y, self.task_id, "demonstration output")) for x, y in self.demonstrations)
        queries = tuple(_copy_grid(x, self.task_id, "query input") for x in self.queries)
        targets = tuple(None if y is None else _copy_grid(y, self.task_id, "query target") for y in self.targets)
        if len(queries) != len(targets):
            raise ValueError("queries and targets must have equal length")
        object.__setattr__(self, "demonstrations", demos)
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "targets", targets)


def load_task(root: str | os.PathLike[str], task_id: str, role: str = "practice", *, allow_evaluation: bool = False) -> ARCTask:
    """Load one declared raw ARC task directly from its role directory."""
    if role not in ("practice", "evaluation"):
        raise ValueError("role must be practice or evaluation")
    if role == "evaluation" and not allow_evaluation:
        raise ValueError("evaluation data is not allowed in ordinary runs; pass allow_evaluation=True")
    if task_id not in TRAINING_ORDER + VALIDATION_ORDER + PROOF_ORDER:
        raise ValueError(f"task {task_id} is not a declared direct ARC task")
    path = Path(root) / "data" / ("training" if role == "practice" else "evaluation") / f"{task_id}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"task {task_id} cannot be read from {path}") from exc
    pairs = payload.get("train", [])
    if len(pairs) > MAX_DEMONSTRATIONS:
        raise ValueError(f"task {task_id} has {len(pairs)} demonstrations; maximum is 10")
    demonstrations = tuple((_copy_grid(pair["input"], task_id, "demonstration input"), _copy_grid(pair["output"], task_id, "demonstration output")) for pair in pairs)
    tests = payload.get("test", [])
    if not tests:
        raise ValueError(f"task {task_id} must contain at least one test query")
    queries = tuple(_copy_grid(item["input"], task_id, "query input") for item in tests)
    targets = tuple(_copy_grid(item["output"], task_id, "query target") if "output" in item else None for item in tests)
    return ARCTask(task_id, demonstrations, queries, targets, role)


def _one_hot(index: int | None, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=bool)
    if index is not None:
        if not 0 <= index < size:
            raise ValueError("event category is out of range")
        result[index] = True
    return result


def _event(kind: int, role: int, slot: int | None, height: int | None, width: int | None, row: int | None, grid_row: np.ndarray | None) -> np.ndarray:
    result = np.concatenate((_one_hot(kind, 7), _one_hot(role, 4), _one_hot(slot, 10), _one_hot(height, 30), _one_hot(width, 30), _one_hot(row, 30)))
    valid = np.zeros(30, dtype=bool)
    colors = np.zeros((30, 10), dtype=bool)
    if grid_row is not None:
        valid[:len(grid_row)] = True
        colors[np.arange(len(grid_row)), grid_row] = True
    return np.concatenate((result, valid, colors.reshape(-1)))


def _grid_events(grid: np.ndarray, role: int, slot: int) -> list[np.ndarray]:
    height, width = grid.shape
    events = [_event(1, role, slot, height - 1, width - 1, None, None)]
    events.extend(_event(2, role, slot, height - 1, width - 1, row, grid[row] if row < height else None) for row in range(MAX_GRID))
    events.append(_event(3, role, slot, height - 1, width - 1, None, None))
    return events


def encode_episode(task: ARCTask, query_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Encode inference-only events and their Boolean advance mask."""
    if not 0 <= query_index < len(task.queries):
        raise IndexError("query_index is out of range")
    events = [_event(0, 0, None, None, None, None, None)]
    for slot in range(MAX_DEMONSTRATIONS):
        if slot < len(task.demonstrations):
            source, target = task.demonstrations[slot]
            events.extend(_grid_events(source, 0, slot))
            events.extend(_grid_events(target, 1, slot))
        else:
            events.extend(np.zeros((64, EVENT_WIDTH), dtype=bool))
    events.extend(_grid_events(task.queries[query_index], 2, 0))
    events.extend((_event(4, 0, None, None, None, None, None), _event(5, 0, None, None, None, None, None)))
    events.extend(_event(6, 0, None, None, None, row, None) for row in range(MAX_GRID))
    encoded = np.asarray(events, dtype=bool)
    if encoded.shape != (EVENTS, EVENT_WIDTH):
        raise AssertionError("internal ARC event schedule has the wrong shape")
    return encoded, np.any(encoded, axis=1)


def _read_grid(events: np.ndarray, start: int, role: int, slot: int) -> tuple[np.ndarray, int]:
    header = events[start]
    if not header[7 + role] or not header[11 + slot]:
        raise ValueError("event stream has an invalid grid header")
    height = int(np.argmax(header[21:51])) + 1
    width = int(np.argmax(header[51:81])) + 1
    rows = []
    for offset in range(1, 31):
        event = events[start + offset]
        if event[7 + role] and event[11 + slot] and np.any(event[111:141]):
            rows.append(np.argmax(event[141:].reshape(30, 10), axis=1)[:width])
    if len(rows) != height:
        raise ValueError("event stream does not contain the declared grid")
    return np.asarray(rows, dtype=np.uint8), start + 32


def decode_episode(events: np.ndarray, task_id: str = "decoded") -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], np.ndarray]:
    """Decode all demonstration inputs, outputs, and the query input."""
    del task_id
    events = np.asarray(events, dtype=bool)
    if events.shape != (EVENTS, EVENT_WIDTH):
        raise ValueError("events must have shape (705, 441)")
    position = 1
    pairs = []
    for slot in range(MAX_DEMONSTRATIONS):
        if np.any(events[position]):
            first, position = _read_grid(events, position, 0, slot)
            second, position = _read_grid(events, position, 1, slot)
            pairs.append((first, second))
        else:
            position += 64
    query, _ = _read_grid(events, position, 2, 0)
    return tuple(pairs), query


def request_loss(logits: np.ndarray, target: np.ndarray, *, request: str, valid_mask: np.ndarray | None = None) -> float:
    """Return strict shape or valid-cell row cross-entropy."""
    values, labels = np.asarray(logits, dtype=np.float64), np.asarray(target, dtype=np.int64)
    if request == "shape":
        if values.shape != (60,) or labels.shape != (2,) or np.any((labels < 0) | (labels >= 30)):
            raise ValueError("shape request needs 60 logits and two target dimensions")
        return float(sum(-x[int(y)] + np.logaddexp.reduce(x) for x, y in zip((values[:30], values[30:]), labels)))
    if request != "row" or values.shape != (30, 10) or labels.shape != (30,) or np.any((labels < 0) | (labels >= 10)):
        raise ValueError("row request needs 30 by 10 logits and 30 target colors")
    mask = np.ones(30, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if mask.shape != (30,):
        raise ValueError("valid_mask must have 30 entries")
    if not np.any(mask):
        return 0.0
    selected, chosen = values[mask], labels[mask]
    return float(np.mean(-selected[np.arange(len(chosen)), chosen] + np.logaddexp.reduce(selected, axis=1)))


def decode_prediction(voltage: np.ndarray) -> np.ndarray:
    """Decode shape logits and one color distribution for each output cell."""
    values = np.asarray(voltage)
    if values.shape == (360,):
        height, width = np.argmax(values[:30]) + 1, np.argmax(values[30:60]) + 1
        colors = np.argmax(values[60:].reshape(30, 10), axis=1)
        return np.asarray(colors[:height, None].repeat(width, axis=1), dtype=np.uint8)
    if values.shape != (30, 360):
        raise ValueError("voltage must contain 30 rows of 360 values")
    height, width = np.argmax(values[0, :30]) + 1, np.argmax(values[0, 30:60]) + 1
    colors = np.argmax(values[:, 60:].reshape(30, 30, 10), axis=2)
    return np.asarray(colors[:height, :width], dtype=np.uint8)


def query_exact(prediction: np.ndarray, target: np.ndarray) -> bool:
    """Return true only for equal integer grids."""
    prediction, target = np.asarray(prediction), np.asarray(target)
    return bool(_is_result_grid(prediction) and _is_result_grid(target) and np.array_equal(prediction, target))


def _is_result_grid(value: np.ndarray) -> bool:
    return bool(
        value.ndim == 2
        and 0 < value.shape[0] <= MAX_GRID
        and 0 < value.shape[1] <= MAX_GRID
        and np.issubdtype(value.dtype, np.integer)
        and np.all((value >= 0) & (value <= 9))
    )


def strict_task_pass_at_1(predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray]) -> bool:
    """Return true only when every non-empty query is exact."""
    return len(predictions) == len(targets) and bool(predictions) and all(query_exact(p, t) for p, t in zip(predictions, targets))


def write_result(path: str | os.PathLike[str], records: Sequence[Mapping[str, object]]) -> None:
    """Validate and atomically write the bounded result schema."""
    tasks = []
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a sequence")
    for record in records:
        if set(record) != {"task_id", "queries", "strict_pass_at_1"}:
            raise ValueError("task result fields must be task_id, queries, and strict_pass_at_1")
        if not isinstance(record["task_id"], str) or not isinstance(record["strict_pass_at_1"], (bool, np.bool_)):
            raise ValueError("task_id must be text and strict_pass_at_1 must be Boolean")
        queries = []
        predictions, targets = [], []
        if not isinstance(record["queries"], Sequence) or isinstance(record["queries"], (str, bytes)):
            raise ValueError("queries must be a sequence")
        for query in record["queries"]:
            if set(query) != {"query_index", "prediction", "target", "exact"}:
                raise ValueError("query result fields must be query_index, prediction, target, and exact")
            if isinstance(query["query_index"], (bool, np.bool_)) or not isinstance(query["query_index"], (int, np.integer)) or query["query_index"] < 0:
                raise ValueError("query_index must be a nonnegative integer")
            if not isinstance(query["exact"], (bool, np.bool_)):
                raise ValueError("exact must be Boolean")
            prediction, target = np.asarray(query["prediction"]), np.asarray(query["target"])
            if not _is_result_grid(prediction) or not _is_result_grid(target):
                raise ValueError("prediction and target must be integer color grids from 0 through 9")
            predictions.append(prediction)
            targets.append(target)
            queries.append({"query_index": int(query["query_index"]), "prediction": prediction.tolist(), "target": target.tolist(), "exact": query_exact(prediction, target)})
        strict = strict_task_pass_at_1(predictions, targets)
        tasks.append({"task_id": str(record["task_id"]), "queries": queries, "strict_pass_at_1": strict})
    payload = {"tasks": tasks, "strict_task_pass_at_1_count": sum(item["strict_pass_at_1"] for item in tasks)}
    _atomic_bytes(Path(path), json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(), MAX_RESULT_BYTES)


def _atomic_bytes(path: Path, data: bytes, limit: int) -> None:
    if len(data) > limit:
        raise ValueError(f"file exceeds the {limit // 1024} KiB limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_checkpoint(path: str | os.PathLike[str], arrays: Mapping[str, np.ndarray], *, format: int = 1, parent: str | os.PathLike[str] | None = None) -> None:
    """Write a compressed, array-only format-1 checkpoint atomically."""
    path = Path(path)
    if parent is not None and path.resolve() == Path(parent).resolve():
        raise ValueError("checkpoint child path must differ from parent")
    _validate_checkpoint_arrays(arrays, format)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as directory:
        temporary = Path(directory) / "checkpoint.npz"
        np.savez_compressed(temporary, format=np.asarray(1, dtype=np.int64), **dict(arrays))
        _atomic_bytes(path, temporary.read_bytes(), MAX_CHECKPOINT_BYTES)


def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Load a validated format-1 array-only checkpoint without pickle."""
    path = Path(path)
    if path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("checkpoint exceeds the 32 MiB limit")
    with np.load(path, allow_pickle=False) as archive:
        if "format" not in archive or archive["format"].shape != () or int(archive["format"]) != 1:
            raise ValueError("checkpoint format must be scalar value 1")
        result = {name: np.array(archive[name], copy=True) for name in archive.files if name != "format"}
    _validate_checkpoint_arrays(result, 1)
    return result


_CHECKPOINT_DTYPES = {
    "neuron_ids": np.dtype("int32"), "dale_codes": np.dtype("int8"),
    "owner_codes": np.dtype("int16"), "mechanism_codes": np.dtype("uint8"),
    "neuron_count": np.dtype("int32"), "integration_substeps": np.dtype("int32"),
    "input_indptr": np.dtype("int32"), "input_indices": np.dtype("int32"),
    "input_values": np.dtype("float32"), "input_m1": np.dtype("float32"), "input_m2": np.dtype("float32"),
    "recurrent_indptr": np.dtype("int32"), "recurrent_indices": np.dtype("int32"),
    "recurrent_values": np.dtype("float32"), "recurrent_m1": np.dtype("float32"), "recurrent_m2": np.dtype("float32"),
    "readout_weight": np.dtype("float32"), "readout_bias": np.dtype("float32"),
    "readout_weight_m1": np.dtype("float32"), "readout_weight_m2": np.dtype("float32"),
    "readout_bias_m1": np.dtype("float32"), "readout_bias_m2": np.dtype("float32"),
    "input_step": np.dtype("int64"), "recurrent_step": np.dtype("int64"), "readout_step": np.dtype("int64"),
}


def _validate_checkpoint_arrays(arrays: Mapping[str, np.ndarray], format: int) -> None:
    if format != 1 or set(arrays) != set(_CHECKPOINT_DTYPES):
        raise ValueError("checkpoint format 1 requires the exact array schema")
    for name, dtype in _CHECKPOINT_DTYPES.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray) or value.dtype != dtype or value.dtype.hasobject:
            raise ValueError(f"checkpoint {name} has invalid dtype")
        if value.dtype.kind == "f" and not np.all(np.isfinite(value)):
            raise ValueError(f"checkpoint {name} must be finite")
    n = _scalar_count(arrays["neuron_count"], "neuron_count")
    substeps = _scalar_count(arrays["integration_substeps"], "integration_substeps")
    if n <= 0 or substeps <= 0 or arrays["neuron_ids"].shape != (n,):
        raise ValueError("checkpoint neuron counts are inconsistent")
    if any(arrays[name].shape != (n,) for name in ("dale_codes", "owner_codes", "mechanism_codes")):
        raise ValueError("checkpoint label shapes must match neuron_count")
    if not np.all(np.isin(arrays["dale_codes"], (-1, 0, 1))) or np.any(arrays["owner_codes"] < -2):
        raise ValueError("checkpoint labels contain an invalid code")
    if np.any(arrays["mechanism_codes"] < 0):
        raise ValueError("checkpoint mechanism code is invalid")
    for prefix, rows, endpoint_limit in (("input", 1, n), ("recurrent", n, n)):
        indptr = arrays[f"{prefix}_indptr"]
        indices = arrays[f"{prefix}_indices"]
        if indptr.shape != (rows + 1,) or indptr[0] != 0 or np.any(np.diff(indptr) < 0) or indptr[-1] != len(indices):
            raise ValueError(f"checkpoint {prefix} CSR structure is invalid")
        if np.any(indices < 0) or np.any(indices >= endpoint_limit):
            raise ValueError(f"checkpoint {prefix} CSR endpoint is invalid")
        if len(indices) > MAX_BIOLOGICAL_CONNECTIONS:
            raise ValueError("checkpoint exceeds the biological connection limit")
        for suffix in ("indices", "values", "m1", "m2"):
            if arrays[f"{prefix}_{suffix}"].shape != indices.shape:
                raise ValueError(f"checkpoint {prefix} arrays have inconsistent lengths")
    if len(arrays["input_indices"]) + len(arrays["recurrent_indices"]) > MAX_BIOLOGICAL_CONNECTIONS:
        raise ValueError("checkpoint exceeds the biological connection limit")
    for name in ("input_step", "recurrent_step", "readout_step"):
        if arrays[name].shape != () or int(arrays[name]) < 0:
            raise ValueError(f"checkpoint {name} must be a nonnegative scalar")
    if arrays["readout_weight"].shape != (n, 360) or arrays["readout_bias"].shape != (360,):
        raise ValueError("checkpoint readout parameters have invalid shapes")
    for suffix in ("m1", "m2"):
        if arrays[f"readout_weight_{suffix}"].shape != arrays["readout_weight"].shape or arrays[f"readout_bias_{suffix}"].shape != arrays["readout_bias"].shape:
            raise ValueError("checkpoint readout moments have inconsistent shapes")


def _scalar_count(value: np.ndarray, name: str) -> int:
    if value.shape != ():
        raise ValueError(f"checkpoint {name} must be a scalar")
    return int(value)
