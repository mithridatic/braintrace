"""Deterministic training-only synthetic ARC curriculum for the direct model."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from numbers import Integral

import brainstate
import numpy as np

try:
    from examples.pp_prop.latent_workspace_task import (
        ArcGrid,
        ArcPair,
        ArcTask,
        canonical_task_fingerprint,
    )
except ModuleNotFoundError:  # pragma: no cover - direct-script import fallback.
    from latent_workspace_task import (  # pyright: ignore[reportImplicitRelativeImport]
        ArcGrid,
        ArcPair,
        ArcTask,
        canonical_task_fingerprint,
    )

CURRICULUM_SCHEMA_VERSION = "direct_synthetic_curriculum_v1"
FAMILIES = (
    "copy",
    "recolor",
    "dihedral",
    "crop",
    "upscale",
    "count",
    "pattern_label",
)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer.")
    integer = int(value)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


@dataclass(frozen=True)
class SyntheticCurriculumConfig:
    """Configure a balanced training-only synthetic ARC curriculum.

    Parameters
    ----------
    task_count : int
        Positive number of tasks generated in family round-robin order.
    demonstrations : int, default=4
        Number of supervised demonstrations per task, from 2 through 10.
    max_grid_size : int, default=12
        Maximum generated input or output side length, from 6 through 30.
    """

    task_count: int
    demonstrations: int = 4
    max_grid_size: int = 12

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_count", _positive_integer(self.task_count, "task_count")
        )
        demonstrations = _positive_integer(self.demonstrations, "demonstrations")
        if not 2 <= demonstrations <= 10:
            raise ValueError("demonstrations must be in 2..10.")
        object.__setattr__(self, "demonstrations", demonstrations)
        max_grid_size = _positive_integer(self.max_grid_size, "max_grid_size")
        if not 6 <= max_grid_size <= 30:
            raise ValueError("max_grid_size must be in 6..30.")
        object.__setattr__(self, "max_grid_size", max_grid_size)


@dataclass(frozen=True)
class SyntheticCurriculum:
    """Hold generated tasks and stable provenance.

    Parameters
    ----------
    tasks : tuple of ArcTask
        Ordered generated tasks.
    family_counts : dict of str to int
        Exact generated count for every curriculum family.
    task_sha256 : str
        Digest of ordered canonical task fingerprints.
    schema_version : str
        Fixed generator contract identifier.
    """

    tasks: tuple[ArcTask, ...]
    family_counts: dict[str, int]
    task_sha256: str
    schema_version: str = CURRICULUM_SCHEMA_VERSION


def _randint(
    rng: brainstate.random.RandomState,
    low: int,
    high: int,
    size: tuple[int, ...] | None = None,
) -> np.ndarray:
    return np.asarray(rng.randint(low, high, size=size), dtype=np.int32)


def _grid(array: np.ndarray) -> ArcGrid:
    return ArcGrid(tuple(tuple(int(cell) for cell in row) for row in array))


def _random_grid(
    rng: brainstate.random.RandomState,
    max_grid_size: int,
    *,
    square: bool = False,
    foreground_color: int | None = None,
) -> np.ndarray:
    side_limit = min(max_grid_size, 9)
    height = int(_randint(rng, 3, side_limit + 1))
    width = height if square else int(_randint(rng, 3, side_limit + 1))
    mask = np.asarray(rng.random((height, width)), dtype=np.float32) < 0.28
    row = int(_randint(rng, 0, height))
    column = int(_randint(rng, 0, width))
    mask[row, column] = True
    if foreground_color is None:
        colors = _randint(rng, 1, 10, size=(height, width))
    else:
        colors = np.full((height, width), foreground_color, dtype=np.int32)
    return np.where(mask, colors, 0).astype(np.int32)


def _dihedral(array: np.ndarray, index: int) -> np.ndarray:
    transformed = np.fliplr(array) if index >= 4 else array
    return np.rot90(transformed, k=index % 4).copy()


def _crop_foreground(array: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(array != 0)
    return array[
        int(rows.min()) : int(rows.max()) + 1,
        int(columns.min()) : int(columns.max()) + 1,
    ].copy()


def _random_count_grid(
    rng: brainstate.random.RandomState, max_grid_size: int
) -> tuple[np.ndarray, int]:
    side = min(max_grid_size, 9)
    array = np.zeros((side, side), dtype=np.int32)
    candidates = np.asarray(
        [(row, column) for row in range(0, side, 2) for column in range(0, side, 2)],
        dtype=np.int32,
    )
    count = int(_randint(rng, 1, min(9, len(candidates)) + 1))
    order = np.asarray(rng.permutation(len(candidates)), dtype=np.int32)[:count]
    color = int(_randint(rng, 1, 10))
    selected = candidates[order]
    array[selected[:, 0], selected[:, 1]] = color
    return array, count


def _pair(input_array: np.ndarray, output_array: np.ndarray) -> ArcPair:
    return ArcPair(_grid(input_array), _grid(output_array))


def _ordinary_task(
    family: str,
    task_index: int,
    config: SyntheticCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> ArcTask:
    pair_count = config.demonstrations + 1
    pairs: list[ArcPair] = []
    if family == "copy":
        for _ in range(pair_count):
            array = _random_grid(rng, config.max_grid_size)
            pairs.append(_pair(array, array))
    elif family == "recolor":
        source = int(_randint(rng, 1, 10))
        target = int(_randint(rng, 1, 10))
        if target == source:
            target = target % 9 + 1
        for _ in range(pair_count):
            array = _random_grid(rng, config.max_grid_size, foreground_color=source)
            pairs.append(_pair(array, np.where(array == source, target, 0)))
    elif family == "dihedral":
        transform = int(_randint(rng, 1, 8))
        for _ in range(pair_count):
            array = _random_grid(rng, config.max_grid_size)
            pairs.append(_pair(array, _dihedral(array, transform)))
    elif family == "crop":
        for _ in range(pair_count):
            inner = _random_grid(rng, max(3, config.max_grid_size - 2))
            padded = np.pad(inner, ((1, 1), (1, 1)))
            pairs.append(_pair(padded, _crop_foreground(padded)))
    elif family == "upscale":
        factor = int(_randint(rng, 2, 4))
        input_limit = min(9, config.max_grid_size // factor)
        for _ in range(pair_count):
            array = _random_grid(rng, max(3, input_limit))
            output = np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)
            pairs.append(_pair(array, output))
    elif family == "count":
        for _ in range(pair_count):
            array, count = _random_count_grid(rng, config.max_grid_size)
            pairs.append(_pair(array, np.asarray([[count]], dtype=np.int32)))
    else:  # pragma: no cover - guarded by generate_synthetic_curriculum.
        raise ValueError(f"Unknown synthetic curriculum family: {family!r}.")
    return ArcTask(
        train=tuple(pairs[:-1]),
        test=(pairs[-1],),
        task_id=f"synthetic-v1:{family}:{task_index:06d}",
    )


def _pattern_label_task(
    task_index: int,
    config: SyntheticCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> ArcTask:
    class_count = max(1, min(4, config.demonstrations // 2))
    patterns = []
    labels = np.asarray(rng.permutation(np.arange(1, 10, dtype=np.int32)))[:class_count]
    for _ in range(class_count):
        pattern = np.asarray(rng.random((3, 3)), dtype=np.float32) < 0.45
        pattern[int(_randint(rng, 0, 3)), int(_randint(rng, 0, 3))] = True
        patterns.append(pattern)
    pairs = []
    for index in range(config.demonstrations):
        class_index = index % class_count
        color = int(_randint(rng, 1, 10))
        input_array = np.where(patterns[class_index], color, 0).astype(np.int32)
        pairs.append(
            _pair(input_array, np.asarray([[labels[class_index]]], dtype=np.int32))
        )
    query_class = int(_randint(rng, 0, class_count))
    query_color = int(_randint(rng, 1, 10))
    query_input = np.where(patterns[query_class], query_color, 0).astype(np.int32)
    query = _pair(query_input, np.asarray([[labels[query_class]]], dtype=np.int32))
    return ArcTask(
        train=tuple(pairs),
        test=(query,),
        task_id=f"synthetic-v1:pattern_label:{task_index:06d}",
    )


def generate_synthetic_curriculum(
    config: SyntheticCurriculumConfig,
    rng: brainstate.random.RandomState,
) -> SyntheticCurriculum:
    """Generate balanced synthetic ARC tasks for checkpoint pretraining only.

    Parameters
    ----------
    config : SyntheticCurriculumConfig
        Validated curriculum dimensions.
    rng : brainstate.random.RandomState
        Sole random source used by every family.

    Returns
    -------
    SyntheticCurriculum
        Ordered tasks, exact family counts, and canonical digest.
    """

    if not isinstance(config, SyntheticCurriculumConfig):
        raise TypeError("config must be a SyntheticCurriculumConfig instance.")
    if not isinstance(rng, brainstate.random.RandomState):
        raise TypeError("rng must be a brainstate.random.RandomState.")
    tasks = []
    family_counts = {family: 0 for family in FAMILIES}
    for task_index in range(config.task_count):
        family = FAMILIES[task_index % len(FAMILIES)]
        if family == "pattern_label":
            task = _pattern_label_task(task_index, config, rng)
        else:
            task = _ordinary_task(family, task_index, config, rng)
        tasks.append(task)
        family_counts[family] += 1
    fingerprints = "\n".join(canonical_task_fingerprint(task) for task in tasks)
    return SyntheticCurriculum(
        tasks=tuple(tasks),
        family_counts=family_counts,
        task_sha256=hashlib.sha256(fingerprints.encode("ascii")).hexdigest(),
    )
